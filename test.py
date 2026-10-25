"""
Evaluate a trained Neural A* checkpoint, or inspect one test map.
 
Whole test set (Opt / Exp / Hmean + a few example plots):
    python test.py --checkpoint runs/neural_astar/best.pt --data data/test.npz
 
One test map (picture of the map and its guidance map):
    python test.py --checkpoint runs/neural_astar/best.pt --data data/test.npz --index 5
    python test.py --checkpoint runs/neural_astar/best.pt --data data/test.npz --random
Left panel : the map, R = goal, G = start, B = free (walls black).
Right panel: the guidance map in a single colour; darker = higher cost.
"""
import argparse
import json
import os
 
import numpy as np
import torch
from torch.utils.data import DataLoader
 
from neural_astar import NeuralAStar, PlanningDataset, evaluate, plan_single
from train import save_examples
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help=".npz with maps/starts/goals/paths")
    ap.add_argument("--invert-map", action="store_true", help="your maps use 1 = obstacle")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-vis", type=int, default=8, help="example plots in whole-set mode")
    ap.add_argument("--out-dir", default=None, help="default: folder of the checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # single-map mode
    ap.add_argument("--index", type=int, default=None, help="show this test map and its guidance map")
    ap.add_argument("--random", action="store_true", help="show a random test map")
    ap.add_argument("--pure", action="store_true", help="start pure green, goal pure red")
    ap.add_argument("--auto-range", action="store_true",
                    help="scale guidance colours to [min, max] of this map instead of [0, 1]")
    ap.add_argument("--mask-walls", action="store_true", help="paint obstacles black in the guidance panel")
    args = ap.parse_args()
 
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    # rebuild the model exactly as it was trained (depth / centred guidance map are stored in the checkpoint)
    model = NeuralAStar(encoder_name=cfg["encoder_name"], tau=cfg["tau"],
                        max_steps=cfg.get("max_steps"),
                        encoder_depth=cfg.get("encoder_depth", 5),
                        detach_g=cfg.get("detach_g", False),
                        phi_bias=cfg.get("phi_bias"),
                        dist_channels=cfg.get("dist_channels", False)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt['epoch']})")
 
    ds = PlanningDataset.from_npz(args.data, invert_map=args.invert_map)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    if not os.access(out_dir, os.W_OK):               # e.g. /kaggle/input is read-only
        fallback = os.path.join(os.getcwd(), "test_outputs")
        print(f"{out_dir} is read-only -> writing results to {fallback}")
        out_dir = fallback
    os.makedirs(out_dir, exist_ok=True)
 
    # ---------------- single-map mode ----------------
    if args.index is not None or args.random:
        from view_example import render_pair          # only needed here (your own helper file)
        i = int(np.random.randint(len(ds))) if args.random else args.index
        if not 0 <= i < len(ds):
            raise SystemExit(f"--index must be in [0, {len(ds) - 1}], got {i}")
        free = ds.maps[i].numpy()                      # already 1 = free (invert_map handled by the dataset)
        start = tuple(int(x) for x in ds.starts[i])
        goal = tuple(int(x) for x in ds.goals[i])
        phi, hist, path = plan_single(model, free, start, goal, device=device)
        out_path = os.path.join(out_dir, f"test_map_{i}.png")
        render_pair(free, start, goal, phi, out_path, pure=args.pure, auto_range=args.auto_range,
                    mask_walls=args.mask_walls, title=f"test map {i}")
        print(f"test map {i} of {len(ds)}: start {start}, goal {goal}")
        print(f"guidance values: min {phi.min():.3f}, mean {phi.mean():.3f}, max {phi.max():.3f}")
        print(f"Neural A*: {int(hist.sum())} nodes expanded, path length {int(path.sum())} "
              f"(ground truth {int(ds.paths[i].sum())})")
        print(f"saved {out_path}")
        return
 
    # ---------------- whole test set ----------------
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    res = evaluate(model, loader, device)
    print(f"test samples: {len(ds)}")
    print(f"Opt {res['opt']:.2f} | Exp {res['exp']:.2f} | Hmean {res['hmean']:.2f} | L1 loss {res['loss']:.4f}")
    # Is Phi lower on the ground-truth path than on other free cells? (guidance only, no search -> fast)
    on, off = [], []
    with torch.no_grad():
        for batch in loader:
            m, s, g, p = (batch[k].to(device) for k in ("map", "start", "goal", "path"))
            phi = model.guidance(m, s, g)
            on.append(phi[(p > 0.5) & (m > 0.5)].cpu())
            off.append(phi[(p < 0.5) & (m > 0.5)].cpu())
    on, off = torch.cat(on), torch.cat(off)
    print(f"mean Phi on ground-truth path cells {on.mean():.3f} | on all other free cells {off.mean():.3f}")
    res["phi_on_path"], res["phi_off_path"] = on.mean().item(), off.mean().item()
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    if args.n_vis > 0:
        fig_path = os.path.join(out_dir, "test_examples.png")
        save_examples(model, ds, device, fig_path, args.n_vis)
        print(f"saved {fig_path}")
 
 
if __name__ == "__main__":
    main()
