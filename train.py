"""
Train Neural A* (guidance-map encoder + differentiable A*) with Adam.

Example
-------
python make_synthetic_data.py --out-dir data
python train.py --train-data data/train.npz --val-data data/val.npz --epochs 100 --batch-size 32

Your own data: save an .npz with arrays
    maps   (N, 64, 64)  1 = passable, 0 = obstacle   (use --invert-map if it is the other way round)
    starts (N, 2)       (row, col)
    goals  (N, 2)       (row, col)
    paths  (N, 64, 64)  1 along the ground-truth path (incl. start and goal)
    lengths (N,)        optional: cells on a shortest path (needed if `paths` is not a single shortest path)

SUPERVISED MODE (--supervised): train the guidance map Phi DIRECTLY so that it is LOW on optimal-path cells and HIGH
everywhere else (Phi is used as the step cost g).  No differentiable A* is run during training (fast), the loss is
    w_bce * weighted-BCE(1 - Phi, target) + w_dice * Dice + w_contrast * hinge(margin - (mean Phi off path - mean Phi on path))
where target = (1 - phi_lo) on path cells and 0 elsewhere, i.e. Phi -> phi_lo on the path and Phi -> 1 off the path.
Optionally + w_hist * (original Neural A* history L1).  Per-epoch metrics: mean Phi on / off the path, contrast,
precision / recall / F1 of "Phi < 0.5" against the path.  Use data made with  make_synthetic_data.py --target union.

What changed vs. the first version (all aimed at the flip-flopping "all Phi ~ 0 / all Phi ~ 1" behaviour)
-----------------------------------------------------------------------------------------------------
* vgg16_bn encoder, lr 3e-4 with linear warm-up + cosine decay, gradient clipping at 1.0 by default.
* Before training, A* with a constant Phi is evaluated on the validation set. Those numbers are the
  "no learning" frontier: a learned guidance map has to beat them to count as learning anything.
* Per-epoch Phi statistics (phi_mean / phi_sstd / phi_mstd) are logged and plotted. If phi_sstd and
  phi_mstd sit near 0 while phi_mean jumps around, the encoder is outputting a constant map.
* Model selection (--select) defaults to "maximise Exp subject to Opt >= --opt-min". Hmean is dominated by
  Exp when Opt is high, so it happily picks the greedy snapshot.
"""
import argparse
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neural_astar import NeuralAStar, PlanningDataset, evaluate, constant_phi_baselines


def parse_args():
    p = argparse.ArgumentParser(description="Train Neural A* with Adam")
    p.add_argument("--train-data", required=True, help=".npz with maps/starts/goals/paths")
    p.add_argument("--val-data", required=True, help=".npz for validation (model selection)")
    p.add_argument("--invert-map", action="store_true", help="your maps use 1 = obstacle")
    p.add_argument("--encoder", default="vgg16_bn", help="smp encoder name, e.g. vgg16_bn or vgg16")
    p.add_argument("--encoder-depth", type=int, default=5, help="U-Net depth (3-5); smaller = less downsampling")
    p.add_argument("--phi-bias", type=float, default=1.4,
                   help="mean-centred guidance map: global level of Phi = sigmoid(phi_bias) (1.4 -> ~0.80). "
                        "Pick it from baselines.csv (a level with Opt ~85-90%%)")
    p.add_argument("--no-center", action="store_true", help="old behaviour: plain sigmoid output, no centering")
    p.add_argument("--no-dist-channels", action="store_true",
                   help="do not feed the Chebyshev distance to start / goal as extra input channels")
    p.add_argument("--no-augment", action="store_true",
                   help="disable random rotation/flip augmentation of the training set (8x more variety)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16, help="paper: 100; lower it if you run out of GPU memory")
    p.add_argument("--train-max-steps", type=int, default=1024,
                   help="cap on A* iterations during TRAINING only (memory of the autograd graph grows with it; "
                        "1024 = a quarter of a 64x64 map). Evaluation always runs the full search. 0 = no cap")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sched", choices=["cosine", "none"], default="cosine", help="lr schedule after warm-up")
    p.add_argument("--warmup-epochs", type=float, default=2.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm, 0 = off")
    p.add_argument("--tau", type=float, default=None, help="temperature; default sqrt(map width)")
    p.add_argument("--detach-g", action="store_true",
                   help="old behaviour: cut the gradient through g(v*) (Phi(v) only gets gradient via its own G)")
    p.add_argument("--select", choices=["constrained", "hmean", "loss", "f1", "contrast"], default="constrained",
                   help="best.pt criterion: constrained = max Exp s.t. Opt >= --opt-min (fallback: max Opt); "
                        "hmean = paper; loss = min val loss (tends to pick the greedy snapshot); "
                        "f1 / contrast = how well Phi singles out the optimal path (supervised mode)")
    p.add_argument("--opt-min", type=float, default=90.0, help="Opt (%%) floor for --select constrained")
    p.add_argument("--no-baselines", action="store_true", help="skip the constant-Phi baseline evaluation")
    p.add_argument("--supervised", action="store_true",
                   help="train Phi directly (low on the optimal path, high elsewhere) instead of through the A* history loss")
    p.add_argument("--phi-lo", type=float, default=0.2, help="supervised: target Phi on optimal-path cells (off-path target is 1)")
    p.add_argument("--margin", type=float, default=0.4, help="supervised: required (mean Phi off path) - (mean Phi on path)")
    p.add_argument("--w-bce", type=float, default=1.0, help="supervised: weight of the class-balanced BCE")
    p.add_argument("--w-dice", type=float, default=1.0, help="supervised: weight of the Dice loss (overlap of path)")
    p.add_argument("--w-contrast", type=float, default=0.5, help="supervised: weight of the on/off-path contrast hinge")
    p.add_argument("--w-hist", type=float, default=0.0,
                   help="supervised: weight of the original Neural A* history L1 (0 = never run the differentiable A* in training)")
    p.add_argument("--pos-weight-max", type=float, default=30.0, help="supervised: cap of the per-map path-cell weight in the BCE")
    p.add_argument("--eval-search-every", type=int, default=1,
                   help="run the (slow) full differentiable-A* validation (Opt / Exp) every N epochs and at the end; "
                        "cheap Phi metrics are computed every epoch")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-dir", default="runs/neural_astar")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-vis", type=int, default=6, help="validation examples to plot at the end (0 = skip)")
    p.add_argument("--log-every", type=int, default=10, help="print batch progress every N batches (0 = off)")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_score(val, mode, opt_min):
    if mode == "f1":
        return val["f1"]
    if mode == "contrast":
        return val["contrast"]
    if math.isnan(val["opt"]):                 # the A* validation was skipped this epoch
        return -float("inf")
    if mode == "hmean":
        return val["hmean"]
    if mode == "loss":
        return -val["loss"]
    return val["exp"] if val["opt"] >= opt_min else val["opt"] - 1000.0   # any model meeting the floor wins


LOG_COLS = ["epoch", "lr", "train_loss", "val_loss", "val_opt", "val_exp", "val_hmean",
            "phi_mean", "phi_sstd", "phi_mstd", "seconds",
            "phi_on", "phi_off", "contrast", "precision", "recall", "f1", "sup_loss"]


def save_curves(log_path, out_path, ref_hmean=None):
    """Redraw loss / metric / guidance-map curves from log.csv (called every epoch, so you can watch it live)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    ep = [int(r["epoch"]) for r in rows]
    col = lambda k: [float(r[k]) for r in rows]
    fig, ax = plt.subplots(2, 3, figsize=(17, 7))
    ax = ax.ravel()
    ax[0].plot(ep, col("train_loss"), label="train")
    ax[0].plot(ep, col("val_loss"), label="val")
    ax[0].set_title("L1 loss"); ax[0].legend()
    ax[1].plot(ep, col("val_opt"), label="Opt")
    ax[1].plot(ep, col("val_exp"), label="Exp")
    ax[1].set_title("validation Opt / Exp (%)"); ax[1].legend()
    ax[2].plot(ep, col("val_hmean"), color="tab:green", label="learned")
    if ref_hmean is not None:
        ax[2].axhline(ref_hmean, color="k", ls="--", lw=1, label="best constant Phi")
    ax[2].set_title("validation Hmean"); ax[2].legend()
    ax[3].plot(ep, col("phi_mean"), label="mean Phi")
    ax[3].plot(ep, col("phi_sstd"), label="std inside a map")
    ax[3].plot(ep, col("phi_mstd"), label="std across maps")
    ax[3].set_title("guidance map statistics (validation)"); ax[3].legend()
    if "phi_on" in rows[0]:
        ax[4].plot(ep, col("phi_on"), label="Phi on optimal path")
        ax[4].plot(ep, col("phi_off"), label="Phi elsewhere (free)")
        ax[4].set_title("mean Phi: path vs. rest (want low / high)"); ax[4].legend()
        ax[5].plot(ep, col("precision"), label="precision")
        ax[5].plot(ep, col("recall"), label="recall")
        ax[5].plot(ep, col("f1"), label="F1")
        ax[5].set_title("path cells found by (Phi < 0.5)"); ax[5].legend()
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def save_examples(model, dataset, device, path, n):
    """Plot: input | guidance map | search history + path | ground truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from neural_astar import plan_single

    n = min(n, len(dataset))
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.6 * n), squeeze=False)
    for i in range(n):
        sample = dataset[i]
        free = sample["map"][0].numpy()
        start = tuple(int(x) for x in dataset.starts[i])
        goal = tuple(int(x) for x in dataset.goals[i])
        phi, hist, path_ = plan_single(model, free, start, goal, device=device)

        def base_rgb():
            img = np.where(free[..., None] > 0.5, 0.8, 0.0) * np.ones((*free.shape, 3))
            return img

        inp = base_rgb()
        inp[start] = (0, 0.8, 0)
        inp[goal] = (1, 0, 0)

        res = base_rgb()
        res[hist > 0.5] = (0.2, 0.8, 0.2)
        res[path_ > 0.5] = (1, 0, 0)

        gt = base_rgb()
        gt[sample["path"][0].numpy() > 0.5] = (1, 0, 0)

        axes[i, 0].imshow(inp)
        axes[i, 1].imshow(phi, cmap="gray_r", vmin=0, vmax=1)
        ov = np.zeros((*free.shape, 4))
        ov[sample["path"][0].numpy() > 0.5] = (1, 0, 0, 0.55)           # ground-truth path in red
        axes[i, 1].imshow(ov)
        axes[i, 2].imshow(res)
        axes[i, 3].imshow(gt)
        for j, t in enumerate(["input (S green, G red)", "guidance map (dark = costly), GT path red",
                               f"Neural A*: {int(hist.sum())} expanded", "ground truth"]):
            axes[i, j].set_title(t, fontsize=8)
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def supervised_phi_loss(phi, path, free, args):
    """Direct loss on the guidance map.  phi (B,1,H,W) in (0,1) is the step cost; path = optimal cells; free = free cells.
    Target: phi = phi_lo on path cells, phi = 1 on every other free cell.
    Returns (total, bce, dice, hinge)."""
    eps = 1e-6
    q = (1.0 - phi).clamp(eps, 1.0 - eps)                       # 'probability' that a cell is on the optimal path
    pos, neg = path * free, (1.0 - path) * free
    n_pos = pos.sum((1, 2, 3)).clamp(min=1.0)
    n_neg = neg.sum((1, 2, 3)).clamp(min=1.0)
    pw = (n_neg / n_pos).clamp(max=args.pos_weight_max)         # path cells are only a few % of a map
    t = path * (1.0 - args.phi_lo)                              # soft target: q = 1 - phi_lo on the path, 0 elsewhere
    el = -(t * torch.log(q) + (1.0 - t) * torch.log(1.0 - q))
    w = pos * pw.view(-1, 1, 1, 1) + neg
    bce = ((el * w).sum((1, 2, 3)) / w.sum((1, 2, 3)).clamp(min=1.0)).mean()
    qn = (q / (1.0 - args.phi_lo)).clamp(max=1.0) * free
    inter = (qn * pos).sum((1, 2, 3))
    dice = (1.0 - (2.0 * inter + 1.0) / (qn.sum((1, 2, 3)) + pos.sum((1, 2, 3)) + 1.0)).mean()
    on = (phi * pos).sum((1, 2, 3)) / n_pos
    off = (phi * neg).sum((1, 2, 3)) / n_neg
    hinge = F.relu(args.margin - (off - on)).mean()
    total = args.w_bce * bce + args.w_dice * dice + args.w_contrast * hinge
    return total, bce, dice, hinge


@torch.no_grad()
def phi_diagnostics(model, loader, device, args):
    """Cheap per-epoch metrics of the guidance map on the validation set (no A* search)."""
    model.eval()
    on_s = off_s = tp = n_pred = n_gt = 0.0
    sup, n = 0.0, 0
    for batch in loader:
        m, s, g, p = (batch[k].to(device) for k in ("map", "start", "goal", "path"))
        phi = model.guidance(m, s, g)
        pos, neg = p * m, (1.0 - p) * m
        on_s += ((phi * pos).sum((1, 2, 3)) / pos.sum((1, 2, 3)).clamp(min=1.0)).sum().item()
        off_s += ((phi * neg).sum((1, 2, 3)) / neg.sum((1, 2, 3)).clamp(min=1.0)).sum().item()
        pred = (phi < 0.5).float() * m
        tp += (pred * pos).sum().item(); n_pred += pred.sum().item(); n_gt += pos.sum().item()
        sup += supervised_phi_loss(phi, p, m, args)[0].item() * m.shape[0]
        n += m.shape[0]
    prec, rec = tp / max(n_pred, 1.0), tp / max(n_gt, 1.0)
    return {"phi_on": on_s / n, "phi_off": off_s / n, "contrast": (off_s - on_s) / n, "precision": prec,
            "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9), "sup_loss": sup / n}


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    train_ds = PlanningDataset.from_npz(args.train_data, invert_map=args.invert_map,
                                        augment=not args.no_augment)
    val_ds = PlanningDataset.from_npz(args.val_data, invert_map=args.invert_map)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    print(f"train samples: {len(train_ds)}   val samples: {len(val_ds)}   device: {device}")
    print(f"input channels: map, start+goal" + ("" if args.no_dist_channels else ", dist-to-start, dist-to-goal")
          + f"   | augmentation (rot/flip): {'off' if args.no_augment else 'on'}   | weight decay {args.weight_decay}")

    # ---- the "no learning" frontier: A* with a constant guidance map ---------------------------
    ref_hmean = None
    if not args.no_baselines:
        base = constant_phi_baselines(val_loader, device, tau=args.tau)
        print("constant-Phi baselines on validation (Phi = 1 is vanilla A*, Phi = 0 is greedy):")
        with open(os.path.join(args.out_dir, "baselines.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phi", "loss", "opt", "exp", "hmean"])
            for c, r in base:
                print(f"  Phi={c:4.2f} | val loss {r['loss']:.4f} | Opt {r['opt']:5.1f} | Exp {r['exp']:5.1f} | "
                      f"Hmean {r['hmean']:5.1f}")
                w.writerow([c, r["loss"], r["opt"], r["exp"], r["hmean"]])
        ref_hmean = max(r["hmean"] for _, r in base)
        best_c = max(base, key=lambda cr: cr[1]["hmean"])[0]
        print(f"  best constant Hmean {ref_hmean:.1f} (Phi={best_c}); a useful learned map should beat this "
              f"while keeping Opt high")

    phi_bias = None if args.no_center else args.phi_bias
    print("guidance map: " + ("plain sigmoid" if phi_bias is None else
                              f"mean-centred, level sigmoid({phi_bias}) = {1 / (1 + math.exp(-phi_bias)):.2f}"))
    model = NeuralAStar(encoder_name=args.encoder, tau=args.tau, encoder_depth=args.encoder_depth,
                        detach_g=args.detach_g, phi_bias=phi_bias,
                        dist_channels=not args.no_dist_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params / 1e6:.2f}M")

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warm_steps = int(args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warm_steps:
            return (step + 1) / max(warm_steps, 1)
        if args.sched == "none":
            return 1.0
        t = (step - warm_steps) / max(total_steps - warm_steps, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    config = {"encoder_name": args.encoder, "encoder_depth": args.encoder_depth, "tau": args.tau,
              "detach_g": args.detach_g, "phi_bias": phi_bias,
              "dist_channels": not args.no_dist_channels}
    log_path = os.path.join(args.out_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(LOG_COLS)

    best_score = -float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        model.astar.max_steps = args.train_max_steps if args.train_max_steps > 0 else None
        running, seen = 0.0, 0
        n_batches = len(train_loader)
        for bi, batch in enumerate(train_loader, 1):
            m, s, g, p = (batch[k].to(device) for k in ("map", "start", "goal", "path"))
            if args.supervised:
                phi = model.guidance(m, s, g)                         # (B,1,H,W) step cost, differentiable
                if not phi.requires_grad:
                    raise RuntimeError("model.guidance() returned a tensor without grad; cannot train Phi directly")
                loss, bce_, dice_, hinge_ = supervised_phi_loss(phi, p, m, args)
                if args.w_hist > 0:                                   # optional: keep the original A* history loss
                    out, _ = model(m, s, g)
                    loss = loss + args.w_hist * F.l1_loss(out.histories, p)
            else:
                out, _ = model(m, s, g)
                # Eq. (7): mean L1 between the search history C and the ground-truth path map
                loss = F.l1_loss(out.histories, p)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            running += loss.item() * m.shape[0]
            seen += m.shape[0]
            if args.log_every and (bi == 1 or bi % args.log_every == 0 or bi == n_batches):
                el = time.time() - t0
                eta = el / bi * (n_batches - bi)
                print(f"  epoch {epoch:3d} [{bi:4d}/{n_batches}] running loss {running / seen:.4f} | "
                      f"{el:5.0f}s elapsed | epoch ETA {eta:4.0f}s", flush=True)
        train_loss = running / max(seen, 1)

        model.astar.max_steps = None            # full search for validation / checkpoint selection
        diag = phi_diagnostics(model, val_loader, device, args)
        if (epoch % args.eval_search_every == 0) or epoch == args.epochs:
            val = evaluate(model, val_loader, device)
        else:                                   # skip the slow full-search evaluation this epoch
            nan = float("nan")
            val = {"loss": nan, "opt": nan, "exp": nan, "hmean": nan, "phi_mean": nan,
                   "phi_sstd": nan, "phi_mstd": nan}
        val.update(diag)
        secs = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d} | lr {lr_now:.1e} | train loss {train_loss:.4f} | val loss {val['loss']:.4f} | "
              f"Opt {val['opt']:.1f} | Exp {val['exp']:.1f} | Hmean {val['hmean']:.1f} | "
              f"Phi mean {val['phi_mean']:.2f} sstd {val['phi_sstd']:.3f} mstd {val['phi_mstd']:.3f} | "
              f"{secs:.0f}s", flush=True)
        print(f"          Phi on path {val['phi_on']:.3f} | elsewhere {val['phi_off']:.3f} | contrast {val['contrast']:+.3f} | "
              f"precision {val['precision']:.3f} recall {val['recall']:.3f} F1 {val['f1']:.3f}", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, lr_now, train_loss, val["loss"], val["opt"], val["exp"], val["hmean"],
                                    val["phi_mean"], val["phi_sstd"], val["phi_mstd"], secs,
                                    val["phi_on"], val["phi_off"], val["contrast"], val["precision"],
                                    val["recall"], val["f1"], val["sup_loss"]])

        save_curves(log_path, os.path.join(args.out_dir, "curves.png"), ref_hmean)

        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "config": config, "val": val}
        torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))
        score = select_score(val, args.select, args.opt_min)
        if score > best_score:
            best_score = score
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> new best ({args.select}): Opt {val['opt']:.1f} Exp {val['exp']:.1f} "
                  f"Hmean {val['hmean']:.1f}, saved best.pt")

    if args.n_vis > 0:
        try:
            best = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device, weights_only=False)
            model.load_state_dict(best["model"])
            fig_path = os.path.join(args.out_dir, "examples.png")
            save_examples(model, val_ds, device, fig_path, args.n_vis)
            print(f"saved visualization to {fig_path}")
        except ImportError:
            print("matplotlib not installed, skipping visualization")


if __name__ == "__main__":
    main()
