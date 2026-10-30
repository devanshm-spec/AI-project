"""
Stochastic multi-CNN-heuristic A* on 64x64 maps (maps from make_synthetic_data.py), compared against plain A*
with the Chebyshev and Euclidean heuristics on the same maps.

WEIGHTS: the member_*.pt files written by train_heuristic_ensemble.py in --weights-dir ({"model", "config"} with
    config = width / depth / h_scale / input_mode).  The network class (HeuristicNet) is copied into this file, so
    train_heuristic_ensemble.py is not needed at run time.  The nets output h in cell units
    (h = Chebyshev + h_scale * softplus(z), zero on obstacles).

BEFORE THE SEARCH (once per map)
    The map is turned into the RGB input  R = goal (one-hot), G = start (one-hot), B = free cells  and passed through
    every heuristic CNN (input_mode "full" nets additionally get Phi, see below).
    h_mean = mean over the n nets, h_std = std over the n nets (per pixel).
    --invert-map: the maps in the .npz use 1 = obstacle (same flag as in train_heuristic_ensemble.py); the nets always
    see B = free cells, exactly as in training.

THE SEARCH
    O (open) = {start}, C (closed) = {}.  Every iteration:
      1. E(n) = g(n) + h(n) - c_b * sqrt(d_max / (1 + d(n)))   for every n in O
         g(n): accumulated step cost: --g-mode const -> 1 per step; --g-mode guidance -> Phi (the Neural A* guidance
               map, needs --phi-checkpoint and neural_astar.py) is the cost of entering a cell.
         h(n) = h_scale * h_mean(n), d(n) = iteration at which n was added to O (start: 0), d_max = largest depth so far.
      2. sort O by E ascending; rank -> probability, lowest E = highest probability (--rank-dist).
      3. sigma_eff = sum_i std_i * exp(-E_i) / sum_i exp(-E_i)         (std_i = h_std at node i)
         K = floor(S * 2/pi * arctan(sigma_eff / ref_std)),  S = |O|, clipped to [1, S]
      4. draw K distinct nodes from O with those probabilities -> candidate set D (the rest stay in O).
      5. the node of D with the lowest E is expanded: it moves O -> C and its free, in-map, never-before-seen
         8-neighbours are added to O.  Stop when the goal is selected for expansion.
    Nodes are never re-opened and parents/g are fixed when a node is first discovered (as specified), so the path
    is not guaranteed to be shortest.

OUTPUT (all under --out-dir)
    examples/sample_XXXX/   overview.png (all panels) + search_ours.png, search_astar_chebyshev.png,
                            search_astar_euclidean.png, heatmap_h_mean.png, heatmap_h_std.png,
                            h_over_iterations.png, description.txt
    scatter/                scatter_nodes.png (x = plain A* nodes, y = ours; Chebyshev and Euclidean)
    tables/                 summary.csv, summary.json, report.txt (all example descriptions + aggregate stats)

USAGE
    python stochastic_astar.py --weights-dir runs/heuristics --data data/test.npz --out-dir out_vis \
        --n-samples 200 --n-examples 6 [--invert-map] [--g-mode guidance --phi-checkpoint runs/dist/last.pt]

NOTES
    * "full" input-mode nets also take Phi as input: it is the guidance map when --phi-checkpoint is given,
      otherwise a constant --phi-const (default 1; use the value the nets were trained with).
    * plain A* always uses unit step costs; with --g-mode guidance our path cost is in Phi units, so compare nodes,
      not costs.
"""
import argparse
import csv
import glob
import heapq
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
COL = {"wall": (0, 0, 0), "free": (25, 25, 90), "closed": (255, 150, 0), "open": (0, 215, 215),
       "path": (255, 255, 255), "start": (0, 255, 0), "goal": (255, 0, 0)}


# ======================================================================================
# 1. Data / RGB input
# ======================================================================================
def load_npz(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


# ======================================================================================
# 2. Heuristic CNN ensemble (HeuristicNet copied from train_heuristic_ensemble.py) and the optional Phi
# ======================================================================================
def cheb_dist(point_map):
    """(B,1,H,W) one-hot -> Chebyshev distance (in cells) to that point."""
    B, _, H, W = point_map.shape
    idx = point_map.flatten(1).argmax(dim=1)
    r, c = (idx // W).to(point_map.dtype), (idx % W).to(point_map.dtype)
    rr = torch.arange(H, device=point_map.device, dtype=point_map.dtype).view(1, H, 1)
    cc = torch.arange(W, device=point_map.device, dtype=point_map.dtype).view(1, 1, W)
    return torch.maximum((rr - r.view(B, 1, 1)).abs(), (cc - c.view(B, 1, 1)).abs()).unsqueeze(1)


def lower_bound(free, goal, phi):
    cmin = (phi + (1.0 - free) * 1e3).amin(dim=(1, 2, 3), keepdim=True)
    return cmin * cheb_dist(goal) * free


def conv_block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class HeuristicUNet(nn.Module):
    def __init__(self, in_ch, width, depth):
        super().__init__()
        ch = [width * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([conv_block(in_ch, ch[0])] + [conv_block(ch[i], ch[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([conv_block(2 * ch[i], ch[i]) for i in reversed(range(depth))])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        feats = [self.enc[0](x)]
        for e in self.enc[1:]:
            feats.append(e(F.max_pool2d(feats[-1], 2)))
        h = feats[-1]
        for up, dec, skip in zip(self.up, self.dec, reversed(feats[:-1])):
            h = dec(torch.cat([skip, up(h)], dim=1))
        return self.out(h)


class HeuristicNet(nn.Module):
    def __init__(self, width=32, depth=4, h_scale=32.0, input_mode="full"):
        super().__init__()
        assert input_mode in ("full", "rgb"), "input_mode must be 'full' or 'rgb'"
        self.input_mode = input_mode
        self.body = HeuristicUNet(3 if input_mode == "rgb" else 6, width, depth)
        self.h_scale = h_scale

    def forward(self, free, start, goal, phi):
        W = free.shape[-1]
        if self.input_mode == "rgb":
            x = torch.cat([goal, start, free], dim=1)           # R = goal, G = start, B = free
            lb = cheb_dist(goal) * free
        else:
            x = torch.cat([free, start, goal, cheb_dist(start) / W, cheb_dist(goal) / W, phi * free], dim=1)
            lb = lower_bound(free, goal, phi)
        return (lb + self.h_scale * F.softplus(self.body(x))) * free


def load_ensemble(weights_dir, n, device):
    files = sorted(glob.glob(os.path.join(weights_dir, "member_*.pt")))
    if not files:
        files = sorted(glob.glob(os.path.join(weights_dir, "*.pt")))
    if not files:
        raise FileNotFoundError(f"no member_*.pt files found in {weights_dir}")
    if n:
        files = files[:n]
    nets = []
    for f in files:
        ck = torch.load(f, map_location=device, weights_only=False)
        if not (isinstance(ck, dict) and "config" in ck and "model" in ck):
            raise ValueError(f"{f}: expected a checkpoint {{'model', 'config'}} written by train_heuristic_ensemble.py")
        net = HeuristicNet(**ck["config"]).to(device)
        net.load_state_dict(ck["model"])
        nets.append(net.eval())
    modes = {n_.input_mode for n_ in nets}
    print(f"loaded {len(nets)} heuristic CNNs {[os.path.basename(f) for f in files]} (input mode: {', '.join(modes)})")
    return nets


class PhiModel:
    """Neural A* guidance map Phi (same reconstruction as train_heuristic_ensemble.compute_phi)."""

    def __init__(self, path, device):
        from neural_astar import NeuralAStar              # must be next to this file
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = ck["config"]
        self.model = NeuralAStar(encoder_name=cfg["encoder_name"], tau=cfg["tau"], max_steps=cfg.get("max_steps"),
                                 encoder_depth=cfg.get("encoder_depth", 5), detach_g=cfg.get("detach_g", False),
                                 phi_bias=cfg.get("phi_bias"), dist_channels=cfg.get("dist_channels", False)).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, free, start, goal):
        H, W = free.shape
        m = torch.from_numpy(free.astype(np.float32)).view(1, 1, H, W).to(self.device)
        s, g = torch.zeros_like(m), torch.zeros_like(m)
        s[0, 0, start[0], start[1]] = 1.0
        g[0, 0, goal[0], goal[1]] = 1.0
        return self.model.guidance(m, s, g)[0, 0].cpu().numpy().astype(np.float64)


@torch.no_grad()
def cnn_heuristics(nets, free, start, goal, phi, device):
    """-> (n, H, W) float64, one heuristic map (in cells) per CNN."""
    H, W = free.shape
    f = torch.from_numpy(free.astype(np.float32)).view(1, 1, H, W).to(device)
    s, g = torch.zeros_like(f), torch.zeros_like(f)
    s[0, 0, start[0], start[1]] = 1.0
    g[0, 0, goal[0], goal[1]] = 1.0
    p = torch.from_numpy(phi.astype(np.float32)).view(1, 1, H, W).to(device)
    return np.stack([net(f, s, g, p)[0, 0].cpu().numpy() for net in nets]).astype(np.float64)


# ======================================================================================
# 3. Searches
# ======================================================================================
@dataclass
class Result:
    name: str
    found: bool
    path: List[tuple]
    expanded: List[tuple]            # nodes in expansion order
    discovered: int                  # nodes that were ever in the open set (incl. start)
    time: float
    open_final: List[tuple] = field(default_factory=list)
    parent: dict = field(default_factory=dict)
    hist: dict = field(default_factory=dict)   # per-iteration logs (stochastic only)

    @property
    def n_expanded(self):
        return len(self.expanded)

    @property
    def path_len(self):              # cells, start and goal included (same as `lengths` of the dataset)
        return len(self.path)


def backtrack(parent, goal):
    p, cur = [goal], goal
    while parent.get(cur) is not None:
        cur = parent[cur]
        p.append(cur)
    return p[::-1]


def rank_probs(S, kind):
    r = np.arange(S, dtype=np.float64)
    p = (S - r) if kind == "linear" else 1.0 / (r + 1.0)
    return p / p.sum()


def stochastic_astar(free, start, goal, h, h_std, cost, c_b, ref_std, rank_dist, rng, max_exp=None):
    H, W = free.shape
    max_exp = max_exp or H * W
    t0 = time.perf_counter()
    status = np.zeros(H * W, dtype=np.int8)               # 0 unseen, 1 open, 2 closed
    g = np.full(H * W, np.inf)
    depth = np.full(H * W, -1, dtype=np.int64)
    par = np.full(H * W, -1, dtype=np.int64)
    hf, sf, cf, ff = h.ravel(), h_std.ravel(), cost.ravel(), free.ravel()
    s, t = start[0] * W + start[1], goal[0] * W + goal[1]
    status[s], g[s], depth[s] = 1, 0.0, 0
    expanded, hist = [], {"h_selected": [], "h_open_mean": [], "K": [], "S": [], "sigma_eff": [], "E_selected": []}
    found, it = False, 0
    while True:
        idx = np.flatnonzero(status == 1)
        S = len(idx)
        if S == 0 or it >= max_exp:
            break
        it += 1
        d_max = depth.max()
        E = g[idx] + hf[idx] - c_b * np.sqrt(d_max / (1.0 + depth[idx]))
        order = np.argsort(E, kind="stable")              # ascending: order[0] = lowest E
        p = np.empty(S)
        p[order] = rank_probs(S, rank_dist)
        w = np.exp(-(E - E.min()))
        sigma_eff = float((sf[idx] * w).sum() / w.sum())
        K = int(math.floor(S * (2 / math.pi) * math.atan(sigma_eff / ref_std)))
        K = min(max(K, 1), S)
        D = rng.choice(S, size=K, replace=False, p=p)     # candidate set
        b = D[np.argmin(E[D])]
        n = int(idx[b])
        status[n] = 2
        expanded.append((n // W, n % W))
        hist["h_selected"].append(float(hf[n]))
        hist["h_open_mean"].append(float(hf[idx].mean()))
        hist["K"].append(K)
        hist["S"].append(S)
        hist["sigma_eff"].append(sigma_eff)
        hist["E_selected"].append(float(E[b]))
        if n == t:
            found = True
            break
        r, c = divmod(n, W)
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                m = nr * W + nc
                if ff[m] and status[m] == 0:
                    status[m], g[m], depth[m], par[m] = 1, g[n] + cf[m], it, n
    dt = time.perf_counter() - t0
    parent = {(int(i) // W, int(i) % W): ((int(par[i]) // W, int(par[i]) % W) if par[i] >= 0 else None)
              for i in np.flatnonzero(status > 0)}
    path = backtrack(parent, goal) if found else []
    return Result("stochastic A* (CNN ensemble)", found, path, expanded, int((status > 0).sum()), dt,
                  [(int(i) // W, int(i) % W) for i in np.flatnonzero(status == 1)], parent,
                  {k: np.array(v) for k, v in hist.items()})


def heuristic_map(shape, goal, kind):
    H, W = shape
    dr = np.abs(np.arange(H)[:, None] - goal[0]) * np.ones((1, W))
    dc = np.abs(np.arange(W)[None, :] - goal[1]) * np.ones((H, 1))
    return np.maximum(dr, dc) if kind == "chebyshev" else np.sqrt(dr ** 2 + dc ** 2)


def plain_astar(free, start, goal, kind):
    """Textbook A*, 8 neighbours, unit step cost, corner cutting allowed (same grid as the data generator)."""
    H, W = free.shape
    t0 = time.perf_counter()
    hm = heuristic_map(free.shape, goal, kind)
    g = {start: 0}
    parent = {start: None}
    heap = [(hm[start], hm[start], start)]
    closed, expanded, found = set(), [], False
    while heap:
        _, _, n = heapq.heappop(heap)
        if n in closed:
            continue
        closed.add(n)
        expanded.append(n)
        if n == goal:
            found = True
            break
        for dr, dc in NEIGHBORS:
            m = (n[0] + dr, n[1] + dc)
            if 0 <= m[0] < H and 0 <= m[1] < W and free[m] and m not in closed:
                ng = g[n] + 1
                if ng < g.get(m, math.inf):
                    g[m], parent[m] = ng, n
                    heapq.heappush(heap, (ng + hm[m], hm[m], m))
    dt = time.perf_counter() - t0
    open_final = [k for k in g if k not in closed]
    return Result(f"A* ({kind})", found, backtrack(parent, goal) if found else [], expanded, len(g), dt,
                  open_final, parent)


# ======================================================================================
# 4. Pictures
# ======================================================================================
def search_rgb(free, start, goal, res):
    img = np.zeros(free.shape + (3,), dtype=np.uint8)
    img[free > 0] = COL["free"]
    for k, cells in (("closed", res.expanded), ("open", res.open_final), ("path", res.path)):
        for r, c in cells:
            img[r, c] = COL[k]
    img[start], img[goal] = COL["start"], COL["goal"]
    return img


def draw_search(ax, free, start, goal, res, graph=False, extra=""):
    ax.imshow(search_rgb(free, start, goal, res), interpolation="nearest")
    if graph and res.parent:                              # search tree: parent -> child edges of expanded nodes
        segs = [((p[1], p[0]), (c[1], c[0])) for c, p in res.parent.items() if p is not None]
        ax.add_collection(LineCollection(segs, colors=(1, 1, 1, 0.25), linewidths=0.4))
    ax.set_title(f"{res.name}\nexpanded {res.n_expanded}, discovered {res.discovered}, "
                 f"path {res.path_len} cells, {res.time * 1e3:.1f} ms{extra}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def draw_heat(ax, fig, arr, free, start, goal, title):
    m = np.ma.masked_where(free == 0, arr)
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad("black")
    im = ax.imshow(m, cmap=cm, interpolation="nearest")
    ax.plot(start[1], start[0], "o", color="lime", ms=5, mec="k")
    ax.plot(goal[1], goal[0], "o", color="red", ms=5, mec="k")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def draw_hist(ax, res):
    h = res.hist
    it = np.arange(1, len(h["h_selected"]) + 1)
    ax.plot(it, h["h_selected"], label="h of expanded node", lw=1.2)
    ax.plot(it, h["h_open_mean"], label="mean h over open set", lw=1.2)
    ax.set_xlabel("iteration"); ax.set_ylabel("h (mean of CNNs)")
    ax2 = ax.twinx()
    ax2.plot(it, h["K"], color="tab:red", alpha=0.5, lw=0.9, label="K (|D|)")
    ax2.plot(it, h["S"], color="gray", alpha=0.4, lw=0.9, label="|open|")
    ax2.set_ylabel("K, |open|")
    l1, n1 = ax.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, n1 + n2, fontsize=7, loc="upper right")
    ax.set_title("h values per iteration", fontsize=8)


def describe(i, d, free, start, goal, ours, cheb, euc, h_mean, h_std, ref_std, t_cnn, opt_len, n_nets, metric):
    cheb_d = max(abs(start[0] - goal[0]), abs(start[1] - goal[1]))
    key = (lambda r: r.n_expanded) if metric == "expanded" else (lambda r: r.discovered)
    L = [f"=== sample {i:04d} ===",
         f"map {free.shape[0]}x{free.shape[1]}, free fraction {free.mean():.3f}, start {tuple(start)}, goal {tuple(goal)}, "
         f"Chebyshev start-goal distance {cheb_d}, optimal path {opt_len} cells",
         f"heuristic ensemble: {n_nets} CNNs, inference {t_cnn * 1e3:.1f} ms (not included in the search times); "
         f"h_mean on free cells: min {h_mean[free > 0].min():.2f} / mean {h_mean[free > 0].mean():.2f} / "
         f"max {h_mean[free > 0].max():.2f}; h_std: mean {h_std[free > 0].mean():.3f} / max {h_std[free > 0].max():.3f}; "
         f"ref_std {ref_std:.3f}"]
    for r in (ours, cheb, euc):
        ratio = f", path/optimal = {r.path_len / opt_len:.3f}" if r.found else ""
        L.append(f"- {r.name}: found={r.found}, nodes expanded (closed) {r.n_expanded}, nodes discovered (ever in open) "
                 f"{r.discovered}, left in open {len(r.open_final)}, path length {r.path_len} cells{ratio}, "
                 f"time {r.time * 1e3:.2f} ms")
    h = ours.hist
    if len(h.get("K", [])):
        L.append(f"- stochastic details: K mean {h['K'].mean():.1f} / max {h['K'].max()}, |open| mean {h['S'].mean():.1f} / "
                 f"max {h['S'].max()}, sigma_eff mean {h['sigma_eff'].mean():.3f}, h of expanded node first "
                 f"{h['h_selected'][0]:.2f} -> last {h['h_selected'][-1]:.2f}")
    L.append(f"- node ratio ours/A*(Chebyshev) = {key(ours) / key(cheb):.3f}, ours/A*(Euclidean) = {key(ours) / key(euc):.3f} "
             f"(metric: {metric})")
    return "\n".join(L)


def save_example(i, root, free, start, goal, ours, cheb, euc, h_mean, h_std, text):
    d = os.path.join(root, f"sample_{i:04d}")
    os.makedirs(d, exist_ok=True)
    singles = [("search_ours.png", lambda a, f: draw_search(a, free, start, goal, ours, graph=True)),
               ("search_astar_chebyshev.png", lambda a, f: draw_search(a, free, start, goal, cheb)),
               ("search_astar_euclidean.png", lambda a, f: draw_search(a, free, start, goal, euc)),
               ("heatmap_h_mean.png", lambda a, f: draw_heat(a, f, h_mean, free, start, goal, "mean h of the CNNs")),
               ("heatmap_h_std.png", lambda a, f: draw_heat(a, f, h_std, free, start, goal, "std of h across the CNNs")),
               ("h_over_iterations.png", lambda a, f: draw_hist(a, ours))]
    for name, fn in singles:
        fig, ax = plt.subplots(figsize=(6, 6) if name != "h_over_iterations.png" else (7, 4))
        fn(ax, fig)
        fig.tight_layout(); fig.savefig(os.path.join(d, name), dpi=130); plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(16, 10.5))
    for (name, fn), ax in zip(singles, axs.ravel()):
        fn(ax, fig)
    fig.suptitle(text.split("\n")[1], fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(d, "overview.png"), dpi=130); plt.close(fig)
    with open(os.path.join(d, "description.txt"), "w") as f:
        f.write(text + "\n")


def save_scatter(rows, out_dir, metric):
    os.makedirs(out_dir, exist_ok=True)
    a = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    key = "exp" if metric == "expanded" else "disc"
    ours = a[f"ours_{key}"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for ax, kind in zip(axs, ("cheb", "euc")):
        x = a[f"{kind}_{key}"]
        lim = max(x.max(), ours.max()) * 1.05
        ax.scatter(x, ours, s=14, alpha=0.6)
        ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
        ax.set_xlabel(f"nodes {metric} by plain A* ({'Chebyshev' if kind == 'cheb' else 'Euclidean'})")
        ax.set_ylabel(f"nodes {metric} by stochastic CNN A*")
        ax.set_title(f"{'Chebyshev' if kind == 'cheb' else 'Euclidean'}: ours fewer on {(ours < x).mean() * 100:.1f}% "
                     f"of maps, mean ratio {np.mean(ours / x):.2f}", fontsize=9)
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "scatter_nodes.png"), dpi=140); plt.close(fig)


# ======================================================================================
# 5. Main
# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-dir", required=True, help="folder with the member_*.pt files of train_heuristic_ensemble.py")
    ap.add_argument("--n-heuristics", type=int, default=0, help="use only the first n members (0 = all)")
    ap.add_argument("--data", default="data/test.npz", help=".npz written by make_synthetic_data.py")
    ap.add_argument("--out-dir", default="stochastic_astar_out")
    ap.add_argument("--invert-map", action="store_true", help="the maps in the .npz use 1 = obstacle (as in training)")
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--n-examples", type=int, default=5, help="how many maps get a folder with pictures")
    ap.add_argument("--g-mode", choices=["const", "guidance"], default="const",
                    help="const: every step costs 1; guidance: cost of entering a cell = Phi (needs --phi-checkpoint)")
    ap.add_argument("--phi-checkpoint", default=None, help="trained Neural A* checkpoint (guidance map Phi)")
    ap.add_argument("--phi-const", type=float, default=1.0, help="Phi fed to 'full' nets when there is no checkpoint")
    ap.add_argument("--h-scale", type=float, default=1.0, help="h = h_scale * (mean CNN output); std scaled alike")
    ap.add_argument("--c-b", type=float, default=0.1)
    ap.add_argument("--ref-std", type=float, default=0.0, help="<=0: per map, mean h_std over free cells")
    ap.add_argument("--rank-dist", choices=["linear", "inverse"], default="linear",
                    help="rank -> probability: linear (S - r) or inverse (1 / (r + 1))")
    ap.add_argument("--count-metric", choices=["expanded", "discovered"], default="expanded",
                    help="what 'nodes opened' means in the scatter plot / ratios")
    ap.add_argument("--max-expansions", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.g_mode == "guidance" and not args.phi_checkpoint:
        ap.error("--g-mode guidance needs --phi-checkpoint")

    data = load_npz(args.data)
    data["maps"] = (1 - (data["maps"] > 0.5) if args.invert_map else (data["maps"] > 0.5)).astype(np.uint8)
    n = min(args.n_samples, len(data["maps"]))
    nets = load_ensemble(args.weights_dir, args.n_heuristics, args.device)
    phi_model = PhiModel(args.phi_checkpoint, args.device) if args.phi_checkpoint else None
    ex_dir = os.path.join(args.out_dir, "examples")
    tab_dir = os.path.join(args.out_dir, "tables")
    os.makedirs(ex_dir, exist_ok=True); os.makedirs(tab_dir, exist_ok=True)

    rows, reports = [], []
    for i in range(n):
        free = data["maps"][i]
        start, goal = tuple(int(x) for x in data["starts"][i]), tuple(int(x) for x in data["goals"][i])
        cheb = plain_astar(free, start, goal, "chebyshev")
        euc = plain_astar(free, start, goal, "euclidean")
        opt_len = int(data["lengths"][i]) if "lengths" in data else cheb.path_len

        # ---- before the task: pass the map through the CNNs, mean / std per pixel --------------------------
        t0 = time.perf_counter()
        phi = phi_model(free, start, goal) if phi_model else np.full(free.shape, args.phi_const)
        stack = cnn_heuristics(nets, free, start, goal, phi, args.device)
        t_cnn = time.perf_counter() - t0
        h_mean, h_std = stack.mean(0) * args.h_scale, stack.std(0) * args.h_scale
        ref_std = args.ref_std if args.ref_std > 0 else max(float(h_std[free > 0].mean()), 1e-9)
        cost = phi if args.g_mode == "guidance" else np.ones(free.shape)

        # ---- the stochastic search -------------------------------------------------------------------------
        rng = np.random.default_rng(args.seed + i)
        ours = stochastic_astar(free, start, goal, h_mean, h_std, cost, args.c_b, ref_std, args.rank_dist, rng,
                                args.max_expansions or None)
        rows.append({"index": i, "opt_len": opt_len, "ours_found": int(ours.found),
                     "ours_exp": ours.n_expanded, "ours_disc": ours.discovered, "ours_len": ours.path_len,
                     "ours_ms": ours.time * 1e3, "cheb_exp": cheb.n_expanded, "cheb_disc": cheb.discovered,
                     "cheb_len": cheb.path_len, "cheb_ms": cheb.time * 1e3, "euc_exp": euc.n_expanded,
                     "euc_disc": euc.discovered, "euc_len": euc.path_len, "euc_ms": euc.time * 1e3,
                     "cnn_ms": t_cnn * 1e3})
        if i < args.n_examples:
            text = describe(i, data, free, start, goal, ours, cheb, euc, h_mean, h_std, ref_std, t_cnn, opt_len,
                            len(nets), args.count_metric)
            print(text + "\n")
            reports.append(text)
            save_example(i, ex_dir, free, start, goal, ours, cheb, euc, h_mean, h_std, text)

    # ---- aggregate ---------------------------------------------------------------------------------------------
    key = "exp" if args.count_metric == "expanded" else "disc"
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    ok = arr["ours_found"] > 0
    agg = [f"=== aggregate over {n} maps ({args.count_metric} nodes) ===",
           f"stochastic search found the goal on {ok.sum()}/{n} maps",
           f"mean nodes: ours {arr['ours_' + key].mean():.1f}, A*(Chebyshev) {arr['cheb_' + key].mean():.1f}, "
           f"A*(Euclidean) {arr['euc_' + key].mean():.1f}",
           f"ours uses fewer nodes than A*(Chebyshev) on {(arr['ours_' + key] < arr['cheb_' + key]).mean() * 100:.1f}% "
           f"and than A*(Euclidean) on {(arr['ours_' + key] < arr['euc_' + key]).mean() * 100:.1f}% of the maps",
           f"mean path/optimal: ours {np.mean(arr['ours_len'][ok] / arr['opt_len'][ok]) if ok.any() else float('nan'):.3f}, "
           f"Chebyshev {np.mean(arr['cheb_len'] / arr['opt_len']):.3f}, Euclidean {np.mean(arr['euc_len'] / arr['opt_len']):.3f}",
           f"mean search time (ms): ours {arr['ours_ms'].mean():.2f} (+ CNN {arr['cnn_ms'].mean():.1f}), "
           f"Chebyshev {arr['cheb_ms'].mean():.2f}, Euclidean {arr['euc_ms'].mean():.2f}"]
    print("\n".join(agg))
    save_scatter(rows, os.path.join(args.out_dir, "scatter"), args.count_metric)
    with open(os.path.join(tab_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(tab_dir, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=1)
    with open(os.path.join(tab_dir, "report.txt"), "w") as f:
        f.write("\n\n".join(reports + ["\n".join(agg)]) + "\n")
    print(f"\nfigures and tables written under {args.out_dir}/")


if __name__ == "__main__":
    main()
