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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32, help="paper: 100; lower it if you run out of GPU memory")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sched", choices=["cosine", "none"], default="cosine", help="lr schedule after warm-up")
    p.add_argument("--warmup-epochs", type=float, default=2.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm, 0 = off")
    p.add_argument("--tau", type=float, default=None, help="temperature; default sqrt(map width)")
    p.add_argument("--detach-g", action="store_true",
                   help="old behaviour: cut the gradient through g(v*) (Phi(v) only gets gradient via its own G)")
    p.add_argument("--select", choices=["constrained", "hmean", "loss"], default="constrained",
                   help="best.pt criterion: constrained = max Exp s.t. Opt >= --opt-min (fallback: max Opt); "
                        "hmean = paper; loss = min val loss (tends to pick the greedy snapshot)")
    p.add_argument("--opt-min", type=float, default=90.0, help="Opt (%%) floor for --select constrained")
    p.add_argument("--no-baselines", action="store_true", help="skip the constant-Phi baseline evaluation")
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
    if mode == "hmean":
        return val["hmean"]
    if mode == "loss":
        return -val["loss"]
    return val["exp"] if val["opt"] >= opt_min else val["opt"] - 1000.0   # any model meeting the floor wins


LOG_COLS = ["epoch", "lr", "train_loss", "val_loss", "val_opt", "val_exp", "val_hmean",
            "phi_mean", "phi_sstd", "phi_mstd", "seconds"]


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
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
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
        axes[i, 2].imshow(res)
        axes[i, 3].imshow(gt)
        for j, t in enumerate(["input (S green, G red)", "guidance map (dark = costly)",
                               f"Neural A*: {int(hist.sum())} expanded", "ground truth"]):
            axes[i, j].set_title(t, fontsize=8)
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    train_ds = PlanningDataset.from_npz(args.train_data, invert_map=args.invert_map)
    val_ds = PlanningDataset.from_npz(args.val_data, invert_map=args.invert_map)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    print(f"train samples: {len(train_ds)}   val samples: {len(val_ds)}   device: {device}")

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
                        detach_g=args.detach_g, phi_bias=phi_bias).to(device)
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
              "detach_g": args.detach_g, "phi_bias": phi_bias}
    log_path = os.path.join(args.out_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(LOG_COLS)

    best_score = -float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running, seen = 0.0, 0
        n_batches = len(train_loader)
        for bi, batch in enumerate(train_loader, 1):
            m, s, g, p = (batch[k].to(device) for k in ("map", "start", "goal", "path"))
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

        val = evaluate(model, val_loader, device)
        secs = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d} | lr {lr_now:.1e} | train loss {train_loss:.4f} | val loss {val['loss']:.4f} | "
              f"Opt {val['opt']:.1f} | Exp {val['exp']:.1f} | Hmean {val['hmean']:.1f} | "
              f"Phi mean {val['phi_mean']:.2f} sstd {val['phi_sstd']:.3f} mstd {val['phi_mstd']:.3f} | "
              f"{secs:.0f}s", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, lr_now, train_loss, val["loss"], val["opt"], val["exp"], val["hmean"],
                                    val["phi_mean"], val["phi_sstd"], val["phi_mstd"], secs])

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
            best = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
            model.load_state_dict(best["model"])
            fig_path = os.path.join(args.out_dir, "examples.png")
            save_examples(model, val_ds, device, fig_path, args.n_vis)
            print(f"saved visualization to {fig_path}")
        except ImportError:
            print("matplotlib not installed, skipping visualization")


if __name__ == "__main__":
    main()
