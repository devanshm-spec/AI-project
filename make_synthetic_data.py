"""
Generate synthetic 64x64 path-planning data for train.py.

Maps (--style):
  rectangles  (default) random rectangles plus several U-shaped dead ends. Concave obstacles are what
              makes the Chebyshev heuristic misleading, i.e. what gives a learned guidance map something to learn.
  maze        random perfect maze (1-pixel corridors); --maze-loops opens a fraction of the walls to add loops.
  noise       every cell is an obstacle independently with probability ~density. NOTE: on these maps plain A*
              is already close to greedy (little to gain), so they are a poor benchmark for Neural A*.
  mixed       a mixture of TWO of the styles above (--mix-styles A B, default: rectangles maze).
              --mix-ratio R = fraction of the maps (in every split) that come from the FIRST style, so
              R=0.5 is 50/50, R=0.7 is 70% rectangles + 30% maze.  The counts are exact (not random per map)
              and the samples are shuffled.  Each file also stores `style_ids` (index into `style_names`).
              Unless --no-per-style-files is given, val/test are additionally written per style
              (val_rectangles.npz, val_maze.npz, ...) so every style can be evaluated on its own.

Ground truth: shortest paths on the same 8-neighbour, unit-cost, corner-cutting-allowed grid that the
differentiable A* uses. Two BFS runs (from start and from goal) give every cell on *some* shortest path.
  --target single   (default, as in the paper) one shortest path, tie-broken randomly (--tie-break first = deterministic)
  --target union    every cell that lies on any shortest path (removes the arbitrary tie-break from the labels)
The file also stores `lengths` (cells on a shortest path), which Opt uses, so both targets are evaluated the same way.

python make_synthetic_data.py --out-dir data --n-train 3000 --n-val 300 --n-test 300
python make_synthetic_data.py --out-dir data_maze --style maze --maze-loops 0.02
python make_synthetic_data.py --out-dir data_noise --style noise --density 0.30 --density-jitter 0.05
python make_synthetic_data.py --out-dir data_mix --style mixed --mix-styles rectangles maze --mix-ratio 0.5 \\
    --maze-loops 0.02 --pairs-per-map 4
"""
import argparse
import os
from collections import deque

import numpy as np

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# --------------------------------------------------------------------------------------
# Map generators (1 = free, 0 = wall)
# --------------------------------------------------------------------------------------
def random_map(rng, size, n_rect, n_u):
    m = np.ones((size, size), dtype=np.uint8)
    for _ in range(n_rect):
        h = int(rng.integers(2, size // 4 + 1))
        w = int(rng.integers(2, size // 4 + 1))
        r = int(rng.integers(0, size - h + 1))
        c = int(rng.integers(0, size - w + 1))
        m[r:r + h, c:c + w] = 0
    for _ in range(n_u):                                  # U-shaped dead ends
        s = int(rng.integers(10, 23))
        ring = np.ones((s, s), dtype=bool)
        ring[2:s - 2, 2:s - 2] = False                    # hollow square, wall thickness 2
        ring[0:2, s // 2 - 2:s // 2 + 2] = False          # opening on the top side
        ring = np.rot90(ring, int(rng.integers(0, 4)))
        r = int(rng.integers(0, size - s + 1))
        c = int(rng.integers(0, size - s + 1))
        block = m[r:r + s, c:c + s]
        block[ring] = 0
    return m


def noise_map(rng, size, density, jitter):
    """Independent random obstacles; the density is jittered per map so maps differ in difficulty."""
    d = float(np.clip(density + rng.uniform(-jitter, jitter), 0.0, 0.6))
    return (rng.random((size, size)) >= d).astype(np.uint8)


def maze_map(rng, size, loops):
    """Perfect maze (randomised DFS) on the odd-coordinate cell grid, optionally with extra openings."""
    m = np.zeros((size, size), dtype=np.uint8)
    n_cell = (size - 1) // 2
    visited = np.zeros((n_cell, n_cell), dtype=bool)
    start = (int(rng.integers(n_cell)), int(rng.integers(n_cell)))
    visited[start] = True
    m[2 * start[0] + 1, 2 * start[1] + 1] = 1
    stack = [start]
    while stack:
        r, c = stack[-1]
        nbrs = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= r + dr < n_cell and 0 <= c + dc < n_cell and not visited[r + dr, c + dc]]
        if not nbrs:
            stack.pop()
            continue
        nr, nc = nbrs[int(rng.integers(len(nbrs)))]
        visited[nr, nc] = True
        m[r + nr + 1, c + nc + 1] = 1                      # knock down the wall between the two cells
        m[2 * nr + 1, 2 * nc + 1] = 1
        stack.append((nr, nc))
    if loops > 0:
        cand = np.zeros((size, size), dtype=bool)
        cand[1:2 * n_cell:2, 2:2 * n_cell - 1:2] = True    # walls between horizontally adjacent cells
        cand[2:2 * n_cell - 1:2, 1:2 * n_cell:2] = True    # walls between vertically adjacent cells
        cand &= (m == 0)
        m[cand & (rng.random((size, size)) < loops)] = 1
    return m


# --------------------------------------------------------------------------------------
# Shortest paths
# --------------------------------------------------------------------------------------
def bfs_dist(m, src):
    """Distance (in moves) from src to every free cell on the 8-neighbour grid, -1 if unreachable."""
    H, W = m.shape
    W2 = W + 2
    pad = np.zeros((H + 2, W2), dtype=bool)               # zero border => no bounds checks in the loop
    pad[1:-1, 1:-1] = (m == 1)
    free = pad.ravel().tolist()
    offs = [dr * W2 + dc for dr, dc in NEIGHBORS]
    dist = [-1] * len(free)
    s = (src[0] + 1) * W2 + src[1] + 1
    dist[s] = 0
    q = deque([s])
    while q:
        cur = q.popleft()
        d = dist[cur] + 1
        for o in offs:
            nb = cur + o
            if free[nb] and dist[nb] < 0:
                dist[nb] = d
                q.append(nb)
    return np.array(dist, dtype=np.int32).reshape(H + 2, W2)[1:-1, 1:-1]


def sample_path(ds, dg, start, D, rng, random_tb):
    """Walk from start to goal along cells with ds == k and dg == D - k (i.e. always on a shortest path)."""
    H, W = ds.shape
    path, cur = [start], start
    for k in range(1, D + 1):
        cands = []
        for dr, dc in NEIGHBORS:
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < H and 0 <= nc < W and ds[nr, nc] == k and dg[nr, nc] == D - k:
                cands.append((nr, nc))
        cur = cands[int(rng.integers(len(cands)))] if random_tb else cands[0]
        path.append(cur)
    return path


def make_split(n, size, rng, min_dist, make_map, target, random_tb, pairs_per_map):
    maps = np.zeros((n, size, size), dtype=np.uint8)
    paths = np.zeros((n, size, size), dtype=np.uint8)
    starts = np.zeros((n, 2), dtype=np.int64)
    goals = np.zeros((n, 2), dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    rr, cc = np.indices((size, size))
    i = 0
    while i < n:
        m = make_map(rng)
        free = np.argwhere(m == 1)
        if len(free) < 2:
            continue
        got = 0
        for _ in range(20 * pairs_per_map):               # try several start/goal pairs on this map
            if got >= pairs_per_map or i >= n:
                break
            s = tuple(int(x) for x in free[rng.integers(len(free))])
            ds = bfs_dist(m, s)
            cheb = np.maximum(np.abs(rr - s[0]), np.abs(cc - s[1]))
            cand = np.argwhere((ds > 0) & (cheb >= min_dist))     # reachable and far enough
            if len(cand) == 0:
                continue
            g = tuple(int(x) for x in cand[rng.integers(len(cand))])
            dg = bfs_dist(m, g)
            D = int(ds[g])
            if target == "union":
                mask = (ds >= 0) & (dg >= 0) & (ds + dg == D)
                paths[i][mask] = 1
            else:
                for r, c in sample_path(ds, dg, s, D, rng, random_tb):
                    paths[i, r, c] = 1
            maps[i] = m
            starts[i], goals[i] = s, g
            lengths[i] = D + 1                            # cells on a shortest path, start and goal included
            i += 1
            got += 1
    return maps, starts, goals, paths, lengths


def build_map_fn(style, args):
    """Map generator (rng -> 64x64 uint8, 1 = free) for one style."""
    if style == "noise":
        return lambda r: noise_map(r, args.size, args.density, args.density_jitter)
    if style == "maze":
        return lambda r: maze_map(r, args.size, args.maze_loops)
    return lambda r: random_map(r, args.size, n_rect=int(r.integers(8, 21)), n_u=int(r.integers(3, 7)))


def make_split_styles(n, args, rng, styles, ratio):
    """n samples drawn from one style, or from two styles with an exact ratio (share `ratio` from styles[0]).
    The result is shuffled; the last array is the style index of every sample."""
    if len(styles) == 1:
        counts = [n]
    else:
        n0 = int(round(n * ratio))
        counts = [n0, n - n0]
    parts = []
    for sid, (style, cnt) in enumerate(zip(styles, counts)):
        if cnt <= 0:
            continue
        maps, starts, goals, paths, lengths = make_split(
            cnt, args.size, rng, args.min_dist, build_map_fn(style, args), args.target,
            args.tie_break == "random", args.pairs_per_map)
        parts.append((maps, starts, goals, paths, lengths, np.full(cnt, sid, dtype=np.int8)))
    cat = [np.concatenate(x) for x in zip(*parts)]
    perm = rng.permutation(n)
    return [a[perm] for a in cat]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--min-dist", type=int, default=20, help="min Chebyshev distance between start and goal")
    ap.add_argument("--style", choices=["rectangles", "maze", "noise", "mixed"], default="rectangles")
    ap.add_argument("--mix-styles", nargs=2, choices=["rectangles", "maze", "noise"], default=["rectangles", "maze"],
                    metavar=("A", "B"), help="mixed style: the two map types (A is the one --mix-ratio refers to)")
    ap.add_argument("--mix-ratio", type=float, default=0.5,
                    help="mixed style: fraction of the maps taken from the first --mix-styles entry (0..1)")
    ap.add_argument("--no-per-style-files", action="store_true",
                    help="mixed style: do not additionally write val/test files per style")
    ap.add_argument("--density", type=float, default=0.25, help="noise style: obstacle probability per cell")
    ap.add_argument("--density-jitter", type=float, default=0.05, help="noise style: +/- variation per map")
    ap.add_argument("--maze-loops", type=float, default=0.02, help="maze style: fraction of inner walls opened")
    ap.add_argument("--target", choices=["single", "union"], default="single",
                    help="single: one shortest path (paper); union: all cells on any shortest path")
    ap.add_argument("--tie-break", choices=["random", "first"], default="random",
                    help="single target: how to choose among equally short paths")
    ap.add_argument("--pairs-per-map", type=int, default=1, help="start/goal pairs sampled on each generated map")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    if args.style == "mixed":
        styles = list(args.mix_styles)
        if styles[0] == styles[1]:
            ap.error("--mix-styles needs two different styles")
        if not 0.0 <= args.mix_ratio <= 1.0:
            ap.error("--mix-ratio must be in [0, 1]")
    else:
        styles = [args.style]
    for name, n in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
        maps, starts, goals, paths, lengths, style_ids = make_split_styles(n, args, rng, styles, args.mix_ratio)
        out = os.path.join(args.out_dir, f"{name}.npz")
        data = dict(maps=maps, starts=starts, goals=goals, paths=paths, lengths=lengths,
                    style_ids=style_ids, style_names=np.array(styles))
        np.savez_compressed(out, **data)
        print(f"{name}: {n} samples -> {out}  (mean shortest-path length {lengths.mean():.1f} cells, "
              f"mean target size {paths.sum((1, 2)).mean():.1f}, free fraction {maps.mean():.2f})")
        if len(styles) > 1:
            for sid, st in enumerate(styles):
                k = style_ids == sid
                if not k.any():
                    continue
                print(f"    {st:11s}: {int(k.sum()):5d} maps ({k.mean() * 100:.0f}%), mean path length "
                      f"{lengths[k].mean():.1f}, free fraction {maps[k].mean():.2f}")
                if name != "train" and not args.no_per_style_files:
                    sub = {key: v[k] for key, v in data.items() if key != "style_names"}
                    sub["style_names"] = np.array(styles)
                    p = os.path.join(args.out_dir, f"{name}_{st}.npz")
                    np.savez_compressed(p, **sub)
                    print(f"      -> {p}")


if __name__ == "__main__":
    main()
