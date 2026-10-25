"""
compare_all.py  -  run AFTER training on all three map types (rect / maze / noise).

Layout it expects (the one run_pipeline.py creates; every folder can be changed with the --*-root options)

    ROOT/data/{rect,maze,noise}/{train,val,test}.npz              make_synthetic_data.py
    ROOT/runs/guidance/{rect,maze,noise}/{best.pt,last.pt,log.csv,baselines.csv}   train.py
    ROOT/runs/heuristics/{rect,maze,noise}/member_*.pt, log_member_*.csv           train_heuristic_ensemble.py --input-mode rgb
    ROOT/comparison/                                              <- everything this script writes

What it does (every source that is missing is skipped with a warning, so it also works on partial results)
  * stochastic A*  (stochastic_astar.py)  vs  plain A* (Chebyshev, unit cost) on the test set(s)  --eval-sets
  * heuristic sources : Minkowski mean (no training) and the CNN ensembles trained on rect / maze / noise
  * guidance sources  : constant Phi = 1 and Phi = 0.5 ("no guidance") and the learned maps of the rect / maze / noise models
    -> every (heuristic x guidance) combination, so e.g. "maze-trained heuristic on the rectangle test set, with / without
       guidance map" is one row of the table and one group of bars (--main-heuristic maze).
  * metrics per map: found, path length vs the dataset's shortest length (Opt = not longer), expanded ("closed") nodes,
    reduction of expanded nodes vs plain A* (Exp), iterations, mean K, validity of the returned path.

Files written to --out-dir (open the PNGs afterwards)
    01_learning_curves_guidance.png    train/val loss, Opt, Exp, Hmean, Phi statistics of the three Neural A* runs
    02_learning_curves_heuristics.png  validation curves of the heuristic CNN members of the three ensembles
    03_dataset_examples.png            test maps of the three styles with their ground-truth shortest path
    04_guidance_maps.png               Phi of the rect / maze / noise models on one test map of each style
    05_heuristic_maps.png              true cost-to-go vs the three CNN ensembles vs Minkowski on one test map of each style
    06_headline_<set>.png              main heuristic, with / without guidance, vs plain A*   (bars)
    07_all_combinations_<set>.png      heatmaps of every heuristic x guidance combination
    08_distributions_<set>.png         per-map boxplots + scatter against plain A*
    09_search_examples_<set>.png       open / closed / path pictures: plain A* vs stochastic A* variants
    10_neural_astar_cross_eval.png     the three guidance models (and constant Phi) on all test sets, paper metrics (optional)
    results.csv / results.json / results_per_map.json / summary.txt / compare_log.txt

Kaggle notebook:
    !python compare_all.py                                   # defaults: ROOT=/kaggle/working, rect test set, 100 maps
    !python compare_all.py --eval-sets rect,maze,noise --n 100
    !python compare_all.py --reuse-results                   # only redraw the bar / heatmap / box figures
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import time
import traceback
from collections import OrderedDict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
for _p in (_HERE, os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from make_synthetic_data import bfs_dist
from stochastic_astar import (COL, CNNHeuristics, CheckpointGuidance, ConstGuidance, MinkowskiHeuristics,
                              load_npz, plain_astar, result_to_rgb, stochastic_astar, validate_path)

NAMES = ["rect", "maze", "noise"]
STYLE_COLOR = {"rect": "tab:blue", "maze": "tab:green", "noise": "tab:red"}
_LOG = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if _LOG is not None:
        _LOG.write(s + "\n")
        _LOG.flush()


# --------------------------------------------------------------------------------------
# labels / colours
# --------------------------------------------------------------------------------------
def hlabel(k):
    if k == "minkowski":
        return "Minkowski mean\n(no training)"
    return f"CNN heuristic\n(trained on {k[4:]})"


def glabel(k):
    if k.startswith("phi"):
        return f"Phi = {k[3:]}\n(constant)"
    return f"learned Phi\n({k[8:]} model)"


def gcolor(k):
    if k == "phi1":
        return "0.55"
    if k.startswith("phi"):
        return "tab:cyan"
    return STYLE_COLOR.get(k[8:], "tab:purple")


def one_line(s):
    return s.replace("\n", " ")


def default_root():
    return "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def load_test_sets(args, names):
    out = OrderedDict()
    for nm in names:
        p = os.path.join(args.data_root, nm, "test.npz")
        if not os.path.exists(p):
            log(f"[warn] missing test set {p}")
            continue
        maps, starts, goals, lengths = load_npz(p)
        out[nm] = dict(maps=maps, starts=starts, goals=goals, lengths=np.asarray(lengths, float), path=p)
    return out


def load_sources(args, device):
    """-> (heuristic sources, guidance sources) as ordered dicts key -> callable object."""
    heurs, guides = OrderedDict(), OrderedDict()
    heurs["minkowski"] = MinkowskiHeuristics()
    for nm in NAMES:
        d = os.path.join(args.heuristic_root, nm)
        if not glob.glob(os.path.join(d, "member_*.pt")):
            log(f"[warn] no heuristic members in {d} -> skipped")
            continue
        try:
            heurs[f"cnn_{nm}"] = CNNHeuristics(d, device)
            log(f"loaded heuristic ensemble {nm}: {len(heurs[f'cnn_{nm}'].nets)} members")
        except Exception as e:                                           # noqa: BLE001
            log(f"[warn] could not load heuristics from {d}: {e!r}")
    for c in args.const_phis:
        guides[f"phi{c:g}"] = ConstGuidance(c)
    for nm in NAMES:
        p = os.path.join(args.guidance_root, nm, f"{args.ckpt}.pt")
        if not os.path.exists(p):
            log(f"[warn] missing guidance checkpoint {p} -> skipped")
            continue
        try:
            guides[f"learned_{nm}"] = CheckpointGuidance(p, device)
            log(f"loaded guidance model {p}")
        except Exception as e:                                           # noqa: BLE001
            log(f"[warn] could not load {p}: {e!r}")
    return heurs, guides


def is_phi_free(hu):
    """True if the heuristic maps do not depend on Phi (so they can be computed once per map)."""
    if isinstance(hu, MinkowskiHeuristics):
        return True
    nets = getattr(hu, "nets", None)
    return bool(nets) and all(getattr(n, "input_mode", "full") == "rgb" for n in nets)


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------
def evaluate_set(name, ts, heurs, guides, kw, seed, n):
    maps, starts, goals = ts["maps"], ts["starts"], ts["goals"]
    n = min(n, len(maps)) if n else len(maps)
    lengths = np.asarray(ts["lengths"][:n], float)
    t0 = time.time()
    ref = [plain_astar(maps[i], starts[i], goals[i]) for i in range(n)]
    ref_closed = np.array([r.n_closed for r in ref], float)
    ref_len = np.array([r.path_len for r in ref], float)
    log(f"\n=== test set '{name}': {n} maps | plain A*: closed {ref_closed.mean():.1f}, path {ref_len.mean():.1f} cells, "
        f"{int((ref_len <= lengths + 1e-9).sum())}/{n} as short as the dataset's shortest path [{time.time() - t0:.1f}s]")
    phis = {gk: [gd(maps[i], starts[i], goals[i]) for i in range(n)] for gk, gd in guides.items()}
    out = {"set": name, "n": n, "lengths": lengths.tolist(),
           "ref": {"closed": ref_closed.tolist(), "open": [r.n_open for r in ref], "len": ref_len.tolist()},
           "runs": OrderedDict()}
    for hk, hu in heurs.items():
        pf = is_phi_free(hu)
        cache = {}
        for gk in guides:
            rec = {k: [] for k in ("found", "closed", "open", "iters", "meanK", "len", "valid", "sec")}
            for i in range(n):
                h = cache.get(i) if pf else None
                if h is None:
                    h = hu(maps[i], starts[i], goals[i], phis[gk][i])
                    if pf:
                        cache[i] = h
                t1 = time.time()
                res = stochastic_astar(maps[i], starts[i], goals[i], hu, phis[gk][i], seed=seed + i, h_maps=h, **kw)
                rec["sec"].append(time.time() - t1)
                rec["found"].append(bool(res.found))
                rec["closed"].append(int(res.n_closed))
                rec["open"].append(int(res.n_open))
                rec["iters"].append(int(res.iterations))
                rec["meanK"].append(float(np.mean(res.ks)) if res.ks else 0.0)
                rec["len"].append(int(res.path_len) if res.found else 0)
                rec["valid"].append(bool(validate_path(res.path, maps[i], starts[i], goals[i])) if res.found else True)
            out["runs"][f"{hk}|{gk}"] = rec
            a = agg_run(rec, ref_closed, lengths)
            log(f"  {one_line(hlabel(hk)):38s} | {one_line(glabel(gk)):28s} found {a['found']:5.1f}% | Opt {a['opt']:5.1f}% | "
                f"len/opt {a['len_ratio']:.3f} | closed {a['closed']:7.1f} ({a['closed_vs_astar'] * 100:5.1f}% of A*) | "
                f"Exp {a['exp']:5.1f}% | {a['sec']:.2f}s/map" + (f" | !! {a['invalid']} invalid paths" if a["invalid"] else ""))
    return out


def agg_run(rec, ref_closed, lengths):
    found = np.array(rec["found"], bool)
    closed = np.array(rec["closed"], float)
    plen = np.array(rec["len"], float)
    n = len(found)
    opt = found & (plen <= lengths + 1e-9)
    ratio = (plen / lengths)[found]
    exp = np.maximum(0.0, 100.0 * (ref_closed - closed) / np.maximum(ref_closed, 1.0))
    return {"found": float(found.mean() * 100), "opt": float(opt.mean() * 100),
            "len_ratio": float(ratio.mean()) if ratio.size else float("nan"),
            "len_ratio_sem": float(ratio.std(ddof=1) / math.sqrt(ratio.size)) if ratio.size > 1 else 0.0,
            "closed": float(closed.mean()), "closed_sem": float(closed.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "closed_median": float(np.median(closed)), "closed_vs_astar": float(closed.mean() / max(ref_closed.mean(), 1e-9)),
            "exp": float(exp.mean()), "iters": float(np.mean(rec["iters"])), "meanK": float(np.mean(rec["meanK"])),
            "invalid": int(n - sum(rec["valid"])), "sec": float(np.mean(rec["sec"]))}


def agg_ref(res):
    ref_closed = np.array(res["ref"]["closed"], float)
    ref_len = np.array(res["ref"]["len"], float)
    lengths = np.array(res["lengths"], float)
    n = len(ref_closed)
    ratio = ref_len / lengths
    return {"found": 100.0, "opt": float(np.mean(ref_len <= lengths + 1e-9) * 100), "len_ratio": float(ratio.mean()),
            "len_ratio_sem": float(ratio.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "closed": float(ref_closed.mean()), "closed_sem": float(ref_closed.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "closed_median": float(np.median(ref_closed)), "closed_vs_astar": 1.0, "exp": 0.0, "iters": float(ref_closed.mean()),
            "meanK": 0.0, "invalid": 0, "sec": float("nan")}


def aggregate(res):
    ref_closed = np.array(res["ref"]["closed"], float)
    lengths = np.array(res["lengths"], float)
    A = OrderedDict((k, agg_run(v, ref_closed, lengths)) for k, v in res["runs"].items())
    return agg_ref(res), A


def write_tables(results, out_dir):
    rows, lines = [], []
    hdr = f"{'heuristic':46s} {'guidance':28s} {'found%':>7s} {'Opt%':>6s} {'len/opt':>8s} {'closed':>8s} {'vs A*':>7s} " \
          f"{'Exp%':>6s} {'iters':>7s} {'meanK':>6s} {'s/map':>6s}"
    for res in results.values():
        ref, A = aggregate(res)
        lines += [f"\n=== test set '{res['set']}' ({res['n']} maps) ===", hdr]

        def add(h, g, a):
            rows.append({"set": res["set"], "heuristic": h, "guidance": g, "n": res["n"], **a})
            lines.append(f"{h:46s} {g:28s} {a['found']:7.1f} {a['opt']:6.1f} {a['len_ratio']:8.3f} {a['closed']:8.1f} "
                         f"{a['closed_vs_astar'] * 100:6.1f}% {a['exp']:6.1f} {a['iters']:7.1f} {a['meanK']:6.2f} {a['sec']:6.2f}")

        add("plain A* (Chebyshev)", "-", ref)
        for k, a in A.items():
            hk, gk = k.split("|")
            add("stochastic | " + one_line(hlabel(hk)), one_line(glabel(gk)), a)
    lines.append("\nfound%: runs reaching the goal | Opt%: path not longer than the dataset's shortest path | len/opt: mean path "
                 "length / shortest | closed: expanded nodes | vs A*: closed / plain-A* closed | Exp%: mean per-map reduction of "
                 "expanded nodes vs plain A* (floored at 0)")
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    if rows:
        with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    log(text)


# --------------------------------------------------------------------------------------
# figures from curves (no torch needed)
# --------------------------------------------------------------------------------------
def read_csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fig_guidance_curves(args, path):
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    ax = ax.ravel()
    any_run = False
    for nm in NAMES:
        lp = os.path.join(args.guidance_root, nm, "log.csv")
        if not os.path.exists(lp):
            log(f"[warn] no {lp}")
            continue
        rows = read_csv_rows(lp)
        if not rows:
            continue
        any_run = True
        ep = [int(r["epoch"]) for r in rows]
        col = lambda k: [float(r[k]) for r in rows]                      # noqa: E731
        c = STYLE_COLOR[nm]
        ax[0].plot(ep, col("train_loss"), color=c, label=f"{nm} train")
        ax[0].plot(ep, col("val_loss"), color=c, ls="--", label=f"{nm} val")
        ax[1].plot(ep, col("val_opt"), color=c, label=nm)
        ax[2].plot(ep, col("val_exp"), color=c, label=nm)
        ax[3].plot(ep, col("val_hmean"), color=c, label=nm)
        ax[4].plot(ep, col("phi_mean"), color=c, label=nm)
        ax[5].plot(ep, col("phi_sstd"), color=c, label=f"{nm}: std inside a map")
        ax[5].plot(ep, col("phi_mstd"), color=c, ls=":", label=f"{nm}: std across maps")
        bp = os.path.join(args.guidance_root, nm, "baselines.csv")
        if os.path.exists(bp):
            b = read_csv_rows(bp)
            ax[3].axhline(max(float(r["hmean"]) for r in b), color=c, ls=":", lw=1, alpha=0.8)
    for a, t in zip(ax, ["L1 loss (train solid, val dashed)", "validation Opt (%)", "validation Exp (%)",
                         "validation Hmean (dotted = best constant Phi)", "mean Phi on free cells (1 = plain A*, 0 = greedy)",
                         "spread of Phi (validation)"]):
        a.set_title(t, fontsize=10)
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.suptitle("Neural A* guidance-map training (one run per map type)")
    fig.tight_layout()
    if not any_run:
        plt.close(fig)
        raise RuntimeError("no guidance log.csv found")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_heuristic_curves(args, path):
    keys = [("val_loss", "validation loss", 1.0), ("val_over_rate_on", "path cells with h > h* (%)  (lower = safer)", 100.0),
            ("val_tight_on", "mean h / h* on path cells  (1 = exact)", 1.0),
            ("val_cons_viol_rate", "consistency violations (% of edges)", 100.0),
            ("val_goal_abs", "|h(goal)| / C*", 1.0), ("val_foff_rate", "off-path cells with g* + h < C* (%)", 100.0)]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    ax = ax.ravel()
    any_run = False
    for nm in NAMES:
        files = sorted(glob.glob(os.path.join(args.heuristic_root, nm, "log_member_*.csv")))
        runs = [read_csv_rows(f) for f in files]
        runs = [r for r in runs if r]
        if not runs:
            log(f"[warn] no log_member_*.csv in {os.path.join(args.heuristic_root, nm)}")
            continue
        any_run = True
        L = min(len(r) for r in runs)
        ep = [int(x["epoch"]) for x in runs[0][:L]]
        for a, (k, _, scale) in zip(ax, keys):
            M = np.array([[float(x[k]) * scale for x in r[:L]] for r in runs])
            for row in M:
                a.plot(ep, row, color=STYLE_COLOR[nm], alpha=0.2, lw=0.8)
            a.plot(ep, M.mean(0), color=STYLE_COLOR[nm], lw=2, label=f"{nm} ensemble ({len(runs)} members, mean)")
    for a, (_, t, _) in zip(ax, keys):
        a.set_title(t, fontsize=10)
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.suptitle("Heuristic CNN ensembles (thin lines = single members)")
    fig.tight_layout()
    if not any_run:
        plt.close(fig)
        raise RuntimeError("no heuristic logs found")
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# figures: data / guidance / heuristics
# --------------------------------------------------------------------------------------
def base_img(free):
    return np.where(np.asarray(free)[..., None], 0.85, 0.0) * np.ones(free.shape + (3,))


def fig_dataset_examples(args, path, k=3):
    names = [nm for nm in NAMES if os.path.exists(os.path.join(args.data_root, nm, "test.npz"))]
    if not names:
        raise RuntimeError("no test sets found")
    fig, axes = plt.subplots(len(names), k, figsize=(3.6 * k, 3.7 * len(names)), squeeze=False)
    for r, nm in enumerate(names):
        d = np.load(os.path.join(args.data_root, nm, "test.npz"))
        for c in range(k):
            i = c
            free = d["maps"][i] > 0.5
            img = base_img(free)
            img[d["paths"][i] > 0.5] = (1.0, 0.3, 0.3)
            img[tuple(d["starts"][i])] = (0, 0.8, 0)
            img[tuple(d["goals"][i])] = (0.1, 0.3, 1.0)
            axes[r, c].imshow(img, interpolation="nearest")
            axes[r, c].set_title(f"{nm} test #{i}: shortest path {int(d['lengths'][i])} cells", fontsize=8)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.legend(handles=[Patch(color=(0.85,) * 3, label="free"), Patch(color="k", label="wall"),
                        Patch(color=(1, 0.3, 0.3), label="ground-truth path"), Patch(color=(0, 0.8, 0), label="start"),
                        Patch(color=(0.1, 0.3, 1), label="goal")], loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_guidance_gallery(tests, guides, path, idx=0):
    models = [k for k in guides if k.startswith("learned_")]
    if not models:
        raise RuntimeError("no learned guidance models loaded")
    sets = list(tests)
    fig, axes = plt.subplots(len(sets), 1 + len(models), figsize=(3.8 * (1 + len(models)), 3.8 * len(sets)), squeeze=False)
    cmap = plt.get_cmap("Oranges").copy()
    cmap.set_bad("black")
    for r, sn in enumerate(sets):
        ts = tests[sn]
        free, s, g = ts["maps"][idx], ts["starts"][idx], ts["goals"][idx]
        img = base_img(free)
        img[s], img[g] = (0, 0.8, 0), (1, 0, 0)
        axes[r, 0].imshow(img, interpolation="nearest")
        axes[r, 0].set_title(f"{sn} test map #{idx} (S green, G red)", fontsize=9)
        for c, mk in enumerate(models, 1):
            phi = guides[mk](free, s, g)
            im = axes[r, c].imshow(np.ma.masked_where(~free, phi), cmap=cmap, interpolation="nearest")
            fig.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.03)
            axes[r, c].set_title(f"Phi of the {mk[8:]} model" + ("  (in-distribution)" if mk[8:] == sn else "  (OOD)"),
                                 fontsize=9)
        for a in axes[r]:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle("guidance maps (darker = higher step cost; colour scale auto-ranged per panel)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_heuristic_gallery(tests, heurs, path, idx=0):
    cnn = [k for k in heurs if k.startswith("cnn_")]
    if not cnn:
        raise RuntimeError("no CNN heuristic ensembles loaded")
    sets = list(tests)
    cols = ["map", "true cost-to-go h*"] + [f"CNN ({k[4:]}) ensemble mean" for k in cnn] + ["Minkowski mean"]
    fig, axes = plt.subplots(len(sets), len(cols), figsize=(3.3 * len(cols), 3.4 * len(sets)), squeeze=False)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black")
    for r, sn in enumerate(sets):
        ts = tests[sn]
        free, s, g = ts["maps"][idx], ts["starts"][idx], ts["goals"][idx]
        hstar = bfs_dist(free.astype(np.uint8), g).astype(float)
        hstar[hstar < 0] = np.nan
        vmax = float(np.nanmax(hstar))
        img = base_img(free)
        img[s], img[g] = (0, 0.8, 0), (1, 0, 0)
        axes[r, 0].imshow(img, interpolation="nearest")
        axes[r, 1].imshow(np.ma.masked_invalid(np.where(free, hstar, np.nan)), cmap=cmap, vmin=0, vmax=vmax)
        panels = [np.mean(heurs[k](free, s, g, np.ones(free.shape)), axis=0) for k in cnn]
        panels.append(np.mean(heurs["minkowski"](free, s, g, np.ones(free.shape)), axis=0))
        for c, hm in enumerate(panels, 2):
            axes[r, c].imshow(np.ma.masked_where(~free, hm), cmap=cmap, vmin=0, vmax=vmax)
        for c, t in enumerate(cols):
            axes[r, c].set_title((f"{sn}: " if c == 0 else "") + t, fontsize=8)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.suptitle("heuristic maps on test maps of each style (same colour scale per row; unit step cost)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# figures: stochastic A* vs plain A*
# --------------------------------------------------------------------------------------
def annotate(ax, im, M, fmt):
    """Write the cell values into a heatmap, in black or white depending on the cell colour."""
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                r, g, b, _ = im.cmap(im.norm(M[i, j]))
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=9,
                        color="k" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "w")


def bar_panel(ax, labels, vals, errs, colors, title, ref=None, ref_label="plain A*", fmt="{:.1f}"):
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=errs, color=colors, capsize=3, edgecolor="0.3", linewidth=0.5)
    for xi, v in zip(x, vals):
        if np.isfinite(v):
            ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
    if ref is not None:
        ax.axhline(ref, color="k", ls="--", lw=1.2, label=f"{ref_label} ({fmt.format(ref)})")
        ax.legend(fontsize=8, loc="lower right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def fig_headline(res, ref, A, main_h, path):
    keys = [k for k in A if k.split("|")[0] == main_h]
    if not keys:
        raise RuntimeError(f"no runs for heuristic '{main_h}'")
    gks = [k.split("|")[1] for k in keys]
    labels = [glabel(g) for g in gks]
    cols = [gcolor(g) for g in gks]
    get = lambda m: [A[k][m] for k in keys]                            # noqa: E731
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.8))
    bar_panel(ax[0], labels, get("closed"), get("closed_sem"), cols, "expanded (closed) nodes   (lower = faster)",
              ref["closed"])
    bar_panel(ax[1], labels, get("exp"), None, cols, "Exp %: node reduction vs plain A*   (higher = better)")
    bar_panel(ax[2], labels, get("opt"), None, cols, "Opt %: path as short as the shortest path", ref["opt"])
    bar_panel(ax[3], labels, get("len_ratio"), get("len_ratio_sem"), cols, "path length / shortest   (1 = optimal)",
              ref["len_ratio"], fmt="{:.3f}")
    lr = [v for v in get("len_ratio") + [ref["len_ratio"]] if np.isfinite(v)]
    if lr:
        ax[3].set_ylim(min(0.98, min(lr) - 0.02), max(lr) + 0.1)
    fails = [(glabel(g).replace("\n", " "), A[k]["found"]) for k, g in zip(keys, gks) if A[k]["found"] < 99.99]
    sub = ("   |   goal not reached in: " + ", ".join(f"{n} ({f:.0f}% found)" for n, f in fails)) if fails else ""
    fig.suptitle(f"test set '{res['set']}' ({res['n']} maps): stochastic A* with {one_line(hlabel(main_h))} vs plain A*" + sub,
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def fig_matrix(res, ref, A, path):
    hks = list(OrderedDict.fromkeys(k.split("|")[0] for k in A))
    gks = list(OrderedDict.fromkeys(k.split("|")[1] for k in A))
    metrics = [("closed", "expanded nodes (lower = faster)", "viridis_r", "{:.0f}"),
               ("exp", "Exp % vs plain A* (higher = better)", "viridis", "{:.1f}"),
               ("opt", "Opt % (path as short as shortest)", "viridis", "{:.1f}"),
               ("len_ratio", "path length / shortest (1 = optimal)", "viridis_r", "{:.3f}")]
    fig, axes = plt.subplots(2, 2, figsize=(2 * (1.9 * len(gks) + 2.5), 2 * (0.9 * len(hks) + 2.0)), squeeze=False)
    for ax, (m, t, cm, fmt) in zip(axes.ravel(), metrics):
        M = np.full((len(hks), len(gks)), np.nan)
        for i, h in enumerate(hks):
            for j, g in enumerate(gks):
                if f"{h}|{g}" in A:
                    M[i, j] = A[f"{h}|{g}"][m]
        im = ax.imshow(M, cmap=cm, aspect="auto")
        annotate(ax, im, M, fmt)
        ax.set_xticks(range(len(gks)))
        ax.set_xticklabels([glabel(g) for g in gks], fontsize=7)
        ax.set_yticks(range(len(hks)))
        ax.set_yticklabels([hlabel(h) for h in hks], fontsize=7)
        ax.set_title(f"{t}\nplain A*: {fmt.format(ref[m])}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"test set '{res['set']}' ({res['n']} maps): stochastic A*, rows = heuristic, columns = guidance map", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_distributions(res, A, main_h, path):
    keys = [k for k in A if k.split("|")[0] == main_h]
    if not keys:
        raise RuntimeError(f"no runs for heuristic '{main_h}'")
    ref_closed = np.array(res["ref"]["closed"], float)
    lengths = np.array(res["lengths"], float)
    labels = ["plain A*"] + [glabel(k.split("|")[1]) for k in keys]
    cols = ["k"] + [gcolor(k.split("|")[1]) for k in keys]
    closed = [ref_closed] + [np.array(res["runs"][k]["closed"], float) for k in keys]
    ratios = [np.array(res["ref"]["len"], float) / lengths]
    for k in keys:
        f = np.array(res["runs"][k]["found"], bool)
        ratios.append((np.array(res["runs"][k]["len"], float) / lengths)[f])
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    for a, data, t in ((ax[0], closed, "expanded nodes per map"), (ax[2], ratios, "path length / shortest (found runs)")):
        bp = a.boxplot(data, patch_artist=True, showmeans=True)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        a.set_xticklabels(labels, fontsize=7)
        a.set_title(t, fontsize=10)
        a.grid(axis="y", alpha=0.3)
    for k, c in zip(keys, cols[1:]):
        ax[1].scatter(ref_closed, res["runs"][k]["closed"], s=12, alpha=0.6, color=c, label=one_line(glabel(k.split("|")[1])))
    top = max(ref_closed.max(), max(np.max(d) for d in closed[1:])) * 1.05
    ax[1].plot([0, top], [0, top], "k--", lw=1, label="same as plain A*")
    ax[1].set_xlabel("plain A*: expanded nodes")
    ax[1].set_ylabel("stochastic A*: expanded nodes")
    ax[1].set_title("per-map comparison (below the line = fewer expansions)", fontsize=10)
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.suptitle(f"test set '{res['set']}': stochastic A* with {one_line(hlabel(main_h))}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_examples(res, ts, main_h, heurs, guides, kw, seed, k, path):
    n = res["n"]
    maps, starts, goals = ts["maps"], ts["starts"], ts["goals"]
    ref_closed = np.array(res["ref"]["closed"], float)
    gks = [g for g in guides if g in ("phi1", "phi0.5")]
    learned = [g for g in guides if g.startswith("learned_")]
    pref = f"learned_{main_h[4:]}" if main_h.startswith("cnn_") else None
    gks += [pref] if pref in guides else learned[:1]
    pos = np.linspace(0, n - 1, min(k, n)).round().astype(int)
    idxs = [int(np.argsort(ref_closed)[p]) for p in pos]
    hu = heurs[main_h]
    ncol = 1 + len(gks)
    fig, axes = plt.subplots(len(idxs), ncol, figsize=(3.7 * ncol, 4.3 * len(idxs)), squeeze=False)
    for r, i in enumerate(idxs):
        free, s, g = maps[i], starts[i], goals[i]
        panels = [("plain A*", plain_astar(free, s, g))]
        for gk in gks:
            phi = guides[gk](free, s, g)
            panels.append((f"stochastic | {one_line(glabel(gk))}", stochastic_astar(free, s, g, hu, phi, seed=seed + i, **kw)))
        for c, (name, rr) in enumerate(panels):
            axes[r, c].imshow(result_to_rgb(free, s, g, rr), interpolation="nearest")
            axes[r, c].set_title(f"{name}\nmap {i}: closed {rr.n_closed}, open {rr.n_open}, "
                                 f"path {rr.path_len if rr.found else 'not found'} (shortest {int(res['lengths'][i])})", fontsize=7)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.legend(handles=[Patch(facecolor=np.array(COL[q]) / 255.0, edgecolor="0.5", label=lab) for q, lab in
                        (("path", "path"), ("closed", "closed = expanded"), ("open", "open = discovered only"),
                         ("free", "free, never touched"), ("wall", "wall"), ("start", "start"), ("goal", "goal"))],
               loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle(f"test set '{res['set']}': {one_line(hlabel(main_h))} (maps picked from easy to hard for plain A*)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97), h_pad=3.0)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# optional: Neural A* (differentiable search, paper metrics) for the guidance models on all test sets
# --------------------------------------------------------------------------------------
def neural_cross_eval(args, device, n, out_dir):
    import torch
    from torch.utils.data import DataLoader, Subset
    from neural_astar import ConstPhi, NeuralAStar, PlanningDataset, evaluate

    dev = torch.device(device)
    models = OrderedDict()
    for c in args.const_phis:
        models[f"phi{c:g}"] = NeuralAStar(encoder=ConstPhi(c)).to(dev)
    for nm in NAMES:
        p = os.path.join(args.guidance_root, nm, f"{args.ckpt}.pt")
        if not os.path.exists(p):
            continue
        ck = torch.load(p, map_location=dev, weights_only=False)
        cfg = ck["config"]
        m = NeuralAStar(encoder_name=cfg["encoder_name"], tau=cfg["tau"], max_steps=cfg.get("max_steps"),
                        encoder_depth=cfg.get("encoder_depth", 5), detach_g=cfg.get("detach_g", False),
                        phi_bias=cfg.get("phi_bias"), dist_channels=cfg.get("dist_channels", False)).to(dev)
        m.load_state_dict(ck["model"])
        models[f"learned_{nm}"] = m
    sets = [nm for nm in NAMES if os.path.exists(os.path.join(args.data_root, nm, "test.npz"))]
    res = OrderedDict()
    for mk, m in models.items():
        m.eval()
        for sn in sets:
            ds = PlanningDataset.from_npz(os.path.join(args.data_root, sn, "test.npz"))
            loader = DataLoader(Subset(ds, range(min(n, len(ds)))), batch_size=16, shuffle=False)
            res[f"{mk}|{sn}"] = evaluate(m, loader, dev)
            r = res[f"{mk}|{sn}"]
            log(f"  Neural A* [{one_line(glabel(mk))}] on {sn}: Opt {r['opt']:.1f} Exp {r['exp']:.1f} Hmean {r['hmean']:.1f}")
    with open(os.path.join(out_dir, "neural_astar_cross_eval.json"), "w") as f:
        json.dump(res, f, indent=2)
    fig, axes = plt.subplots(1, 3, figsize=(5.2 * 3 + 1, 0.7 * len(models) + 2.6))
    for ax, (m, t) in zip(axes, (("opt", "Opt %"), ("exp", "Exp %"), ("hmean", "Hmean"))):
        M = np.array([[res[f"{mk}|{sn}"][m] for sn in sets] for mk in models])
        im = ax.imshow(M, cmap="viridis", aspect="auto")
        annotate(ax, im, M, "{:.1f}")
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([f"{s} test" for s in sets])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([glabel(k) for k in models], fontsize=7)
        ax.set_title(t)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"Neural A* (differentiable search, Chebyshev heuristic) on {n} test maps per set; rows = guidance map", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "10_neural_astar_cross_eval.png"), dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Compare stochastic A* (with / without guidance) against plain A* and draw the figures",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", default=default_root())
    ap.add_argument("--data-root", default=None, help="default ROOT/data")
    ap.add_argument("--guidance-root", default=None, help="default ROOT/runs/guidance")
    ap.add_argument("--heuristic-root", default=None, help="default ROOT/runs/heuristics")
    ap.add_argument("--out-dir", default=None, help="default ROOT/comparison")
    ap.add_argument("--ckpt", choices=["best", "last"], default="best", help="which guidance checkpoint to use")
    ap.add_argument("--eval-sets", default="rect", help="comma list of test sets for the stochastic-A* comparison (rect,maze,noise)")
    ap.add_argument("--main-heuristic", default="maze", help="rect | maze | noise | minkowski: used for the headline / distribution / example figures")
    ap.add_argument("--const-phis", default="1,0.5", help="constant guidance maps used as 'no guidance'")
    ap.add_argument("--n", type=int, default=100, help="maps per evaluation set (0 = all)")
    ap.add_argument("--n-examples", type=int, default=4, help="rows in the search-example figure")
    ap.add_argument("--neural-eval", type=int, default=-1,
                    help="maps per test set for the Neural A* cross evaluation (0 = skip, -1 = 64 on GPU, skip on CPU)")
    ap.add_argument("--reuse-results", action="store_true", help="load results_per_map.json instead of running the searches")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--seed", type=int, default=0)
    # stochastic A* parameters (same names / defaults as stochastic_astar.py)
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--k-max", type=int, default=20)
    ap.add_argument("--ref-std", type=float, default=2.0)
    ap.add_argument("--c-sig", type=float, default=2.0)
    ap.add_argument("--c-b", type=float, default=0.1)
    ap.add_argument("--rank-power", type=float, default=1.0)
    ap.add_argument("--reduce", choices=["mean", "min", "max"], default="mean")
    ap.add_argument("--stop-on", choices=["generate", "expand"], default="generate")
    args = ap.parse_args(argv)
    args.const_phis = [float(x) for x in args.const_phis.split(",") if x.strip()]
    args.data_root = args.data_root or os.path.join(args.root, "data")
    args.guidance_root = args.guidance_root or os.path.join(args.root, "runs", "guidance")
    args.heuristic_root = args.heuristic_root or os.path.join(args.root, "runs", "heuristics")
    args.out_dir = args.out_dir or os.path.join(args.root, "comparison")
    args.main_h = "minkowski" if args.main_heuristic == "minkowski" else f"cnn_{args.main_heuristic}"
    return args


def stage(name, fn, *a, **k):
    """Run one output step; a failure in one figure must not cost the others."""
    try:
        t0 = time.time()
        fn(*a, **k)
        log(f"[ok] {name} ({time.time() - t0:.0f}s)")
    except Exception as e:                                               # noqa: BLE001
        log(f"[FAILED] {name}: {e!r}")
        log(traceback.format_exc())


def main(argv=None):
    global _LOG
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    _LOG = open(os.path.join(args.out_dir, "compare_log.txt"), "w")
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    log(f"root {args.root} | out {args.out_dir} | device {device}")
    out = lambda f: os.path.join(args.out_dir, f)                          # noqa: E731

    stage("01 learning curves (guidance)", fig_guidance_curves, args, out("01_learning_curves_guidance.png"))
    stage("02 learning curves (heuristics)", fig_heuristic_curves, args, out("02_learning_curves_heuristics.png"))
    stage("03 dataset examples", fig_dataset_examples, args, out("03_dataset_examples.png"))

    eval_names = [s.strip() for s in args.eval_sets.split(",") if s.strip()]
    tests_all = load_test_sets(args, NAMES)
    tests = OrderedDict((k, v) for k, v in tests_all.items() if k in eval_names)
    if not tests:
        log("no evaluation test set found, stopping after the curve figures")
        return
    heurs, guides = load_sources(args, device)
    if args.main_h not in heurs:
        log(f"[warn] main heuristic '{args.main_h}' is not available -> using '{next(iter(heurs))}' for the headline figures")
        args.main_h = next(iter(heurs))
    kw = dict(k_min=args.k_min, k_max=args.k_max, ref_std=args.ref_std, c_sig=args.c_sig, c_b=args.c_b,
              rank_power=args.rank_power, reduce=args.reduce, stop_on=args.stop_on)

    per_map_path = out("results_per_map.json")
    if args.reuse_results and os.path.exists(per_map_path):
        with open(per_map_path) as f:
            results = json.load(f, object_pairs_hook=OrderedDict)
        results = OrderedDict((k, v) for k, v in results.items() if k in tests)
        log(f"reusing {per_map_path}")
    else:
        results = OrderedDict()
        for sn, ts in tests.items():
            results[sn] = evaluate_set(sn, ts, heurs, guides, kw, args.seed, args.n)
            with open(per_map_path, "w") as f:                              # save after every set (long runs)
                json.dump(results, f)
    write_tables(results, args.out_dir)
    with open(out("results.json"), "w") as f:
        json.dump({sn: {"ref": aggregate(r)[0], "runs": aggregate(r)[1]} for sn, r in results.items()}, f, indent=2)

    for sn, res in results.items():
        ref, A = aggregate(res)
        stage(f"06 headline ({sn})", fig_headline, res, ref, A, args.main_h, out(f"06_headline_{sn}.png"))
        stage(f"07 all combinations ({sn})", fig_matrix, res, ref, A, out(f"07_all_combinations_{sn}.png"))
        stage(f"08 distributions ({sn})", fig_distributions, res, A, args.main_h, out(f"08_distributions_{sn}.png"))
        stage(f"09 search examples ({sn})", fig_examples, res, tests[sn], args.main_h, heurs, guides, kw, args.seed,
              args.n_examples, out(f"09_search_examples_{sn}.png"))

    stage("04 guidance maps", fig_guidance_gallery, tests_all, guides, out("04_guidance_maps.png"))
    stage("05 heuristic maps", fig_heuristic_gallery, tests_all, heurs, out("05_heuristic_maps.png"))

    n_neural = args.neural_eval
    if n_neural < 0:
        try:
            import torch
            n_neural = 64 if torch.cuda.is_available() else 0
        except ImportError:
            n_neural = 0
    if n_neural > 0:
        stage("10 Neural A* cross evaluation", neural_cross_eval, args, device, n_neural, args.out_dir)
    else:
        log("[skip] Neural A* cross evaluation (use --neural-eval N to run it; slow without a GPU)")
    log(f"\nall files are in {args.out_dir}")
    _LOG.close()
    _LOG = None


if __name__ == "__main__":
    main()
