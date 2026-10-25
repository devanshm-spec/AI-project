"""
run_pipeline.py - steps 1-3 of the pipeline for the three map types (rect / maze / noise).

    1. data        make_synthetic_data.py   -> ROOT/data/<name>/{train,val,test}.npz
    2. guidance    train.py                 -> ROOT/runs/guidance/<name>/{best.pt,last.pt,log.csv,curves.png,examples.png,...}
    3. heuristics  train_heuristic_ensemble.py --input-mode rgb --phi-const 1
                   (N CNNs; input = 3 channels R = goal, G = start, B = free)
                                            -> ROOT/runs/heuristics/<name>/member_*.pt, log_member_*.csv, ...

Steps whose outputs already exist are skipped (so a restarted Kaggle session continues where it stopped); --force redoes them.
Afterwards run compare_all.py (steps 4-5).

Kaggle:
    !python run_pipeline.py --guidance-epochs 60 --heuristic-epochs 30 --n-members 5
    !python run_pipeline.py --steps data --n-train 3000            # only generate the data
    !python run_pipeline.py --steps guidance,heuristics --styles maze
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# name -> extra make_synthetic_data.py arguments
STYLES = {
    "rect": ["--style", "rectangles"],
    "maze": ["--style", "maze", "--maze-loops", "0.02"],
    "noise": ["--style", "noise", "--density", "0.30", "--density-jitter", "0.05"],
}


def guidance_done(folder, epochs):
    """Finished = best.pt + last.pt exist AND log.csv has all epochs (best/last.pt already exist after epoch 1)."""
    log = os.path.join(folder, "log.csv")
    if not (os.path.exists(os.path.join(folder, "best.pt")) and os.path.exists(log)):
        return False
    with open(log) as f:
        return sum(1 for _ in f) - 1 >= epochs


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=HERE)
    print(f"-- done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd())
    ap.add_argument("--styles", default="rect,maze,noise")
    ap.add_argument("--steps", default="data,guidance,heuristics")
    ap.add_argument("--force", action="store_true", help="redo steps even if their outputs exist")
    ap.add_argument("--seed", type=int, default=0)
    # data
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--data-args", default="", help="extra arguments for make_synthetic_data.py")
    # guidance (Neural A*)
    ap.add_argument("--guidance-epochs", type=int, default=60)
    ap.add_argument("--guidance-batch-size", type=int, default=16)
    ap.add_argument("--guidance-args", default="", help="extra arguments for train.py, e.g. '--phi-bias 1.4 --opt-min 90'")
    # heuristics
    ap.add_argument("--n-members", type=int, default=5)
    ap.add_argument("--heuristic-epochs", type=int, default=30)
    ap.add_argument("--heuristic-batch-size", type=int, default=32)
    ap.add_argument("--heuristic-workers", type=int, default=4, help="processes for the exact Dijkstra targets")
    ap.add_argument("--heuristic-args", default="", help="extra arguments for train_heuristic_ensemble.py, e.g. '--bootstrap --vary-width'")
    args = ap.parse_args()

    py = sys.executable
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    for s in styles:
        if s not in STYLES:
            raise SystemExit(f"unknown style '{s}' (choose from {', '.join(STYLES)})")
    data = lambda s, f="": os.path.join(args.root, "data", s, f)                                   # noqa: E731
    gdir = lambda s, f="": os.path.join(args.root, "runs", "guidance", s, f)                       # noqa: E731
    hdir = lambda s, f="": os.path.join(args.root, "runs", "heuristics", s, f)                     # noqa: E731

    if "data" in steps:
        for s in styles:
            if not args.force and all(os.path.exists(data(s, f"{p}.npz")) for p in ("train", "val", "test")):
                print(f"[skip] data for {s} exists")
                continue
            run([py, "make_synthetic_data.py", "--out-dir", data(s), "--n-train", str(args.n_train), "--n-val", str(args.n_val),
                 "--n-test", str(args.n_test), "--seed", str(args.seed)] + STYLES[s] + args.data_args.split())

    if "guidance" in steps:
        for s in styles:
            if not args.force and guidance_done(gdir(s), args.guidance_epochs):
                print(f"[skip] guidance model for {s} is complete")
                continue
            run([py, "train.py", "--train-data", data(s, "train.npz"), "--val-data", data(s, "val.npz"),
                 "--epochs", str(args.guidance_epochs), "--batch-size", str(args.guidance_batch_size),
                 "--seed", str(args.seed), "--out-dir", gdir(s)] + args.guidance_args.split())

    if "heuristics" in steps:
        for s in styles:
            if not args.force and os.path.exists(hdir(s, f"member_{args.n_members - 1}.pt")) \
                    and os.path.exists(hdir(s, "metrics.json")):
                print(f"[skip] heuristic ensemble for {s} exists")
                continue
            run([py, "train_heuristic_ensemble.py", "--train-data", data(s, "train.npz"), "--val-data", data(s, "val.npz"),
                 "--test-data", data(s, "test.npz"), "--input-mode", "rgb", "--phi-const", "1",
                 "--n-members", str(args.n_members), "--epochs", str(args.heuristic_epochs),
                 "--batch-size", str(args.heuristic_batch_size), "--workers", str(args.heuristic_workers),
                 "--seed", str(args.seed), "--out-dir", hdir(s)] + args.heuristic_args.split())

    print("\npipeline finished. Next: python compare_all.py")


if __name__ == "__main__":
    main()
