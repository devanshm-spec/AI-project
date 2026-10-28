"""
Stochastic multi-heuristic A* on a 64x64 grid, plus plain A* as the reference, plus an evaluation harness.

THE ALGORITHM (as specified)
----------------------------
Input: a 64x64 RGB image. Green = start, red = goal, blue = free space, walls = black (no colour).
Moves: the 8 neighbours, each step enters a cell that costs Phi(cell). With Phi = 1 everywhere (the default) this is
exactly the specification: g(n) = number of steps from the start = depth(n).

State: open set (starts as {start}), closed set, D set (both empty).  Repeat until the goal is found:

  1. h(n)  = reduce over the N heuristics of h_k(n)     (reduce = mean by default; the 5 Minkowski distances
                                                          p = 1, 2, 5, 10, 100, or N heuristic CNNs)
  2. value(n) = h(n) + g(n) + c_b * sqrt(d_max) / (1 + d(n))       d = depth, d_max = max depth in the open set,
             c_b = 0.1.   Rank the open set ascending by value (rank 1 = lowest value).
  3. K     = floor(K_min + (K_max - K_min) / (1 + exp(+std/ref_std - c))),   K_max = min(k_max, |open|)  (K_min = 1,
             k_max = 20).  std = standard deviation ACROSS THE N HEURISTIC MEMBERS h_k at the cell of the node with
             the lowest value (rank 1) of step 2, i.e. the disagreement of the ensemble at the most promising point.
  4. Rank -> probability (lowest value = highest probability). Draw K DISTINCT nodes from the open set with these
     probabilities; they form D.  The node of D with the LOWEST value is the one that gets expanded.
  5. Expand it: its free 8-neighbours that are not in the closed set join the open set (if one is already open,
     its g / parent are updated when the new route is cheaper). If one of the neighbours is the goal, stop and
     return the path. Then move the expanded node from open to closed, and empty D.

Choices the specification leaves open (all are arguments, defaults in brackets):
  * ref_std [2.0] and c [2.0]: with c > 0 members that agree (std ~ 0) give K close to K_max (near-greedy) and members
    that disagree give K close to K_min (random exploration). c = 0 would give K ~ (K_min + K_max)/2 at std = 0.
    --k-dir high-std-large-k restores the previous, opposite direction.
  * std is the (population, ddof = 0) std of the N member heuristics at the lowest-value open node (--std-source
    best, default). --std-source open-mean averages that member std over the whole open set, --std-source
    open-spatial is the OLD behaviour (std of the reduced h across different open cells; not a disagreement measure).
  * the member std is in heuristic units (cells), typically much smaller than the spatial std the old default
    ref_std = 2 was meant for. Look at the meanStd column of `eval` and set --ref-std near the typical std.
  * rank -> probability: p(rank) proportional to rank^-rank_power [1.0]  (rank_power = 0 is uniform, large = greedy).
  * "stop when a neighbour is the goal" is applied literally (--stop-on generate). --stop-on expand uses the
    textbook rule instead (stop when the goal itself is selected for expansion).
  * --heuristic-reduce {mean,min,max} (alias --reduce) combines the members into h, exactly as in test.py.
  * Because nodes are not always expanded in best-first order and closed nodes are never re-opened, the path is
    NOT guaranteed to be shortest. That is a property of the method, so Opt (below) is a measured quantity.

OUTPUT / PICTURES (render_search): 4 panels - the result in the colours below, plain A* in the same colours, the order
in which nodes were expanded (purple = early, yellow = late, path in white), and K and the member std per iteration.
`eval` also writes such pictures for the first --n-vis maps per setting, plus a scatter of expanded nodes (plain A* vs
stochastic) per setting, into --vis-dir.
COLOURS (render_search):  start green, goal red, path white, closed (expanded = "searched") orange,
open (discovered, not expanded) cyan, free space dark blue, walls black. The legend gives the counts.

Heuristic sources
  minkowski        the 5 Minkowski distances p = 1, 2, 5, 10, 100 to the goal  (no training needed)
  cnn:DIR          the member_*.pt networks written by train_heuristic_ensemble.py (needs torch, and that file next
                   to this one). They take Phi as an input, so pick the same guidance they were trained with.
Guidance (Phi) sources
  const:C          Phi = C everywhere (const:1 = plain unit cost, const:0.5 ...)
  ckpt:PATH        guidance map from a Neural A* checkpoint written by train.py (needs torch, neural_astar.py)

Examples
  # one map from a dataset, picture with the open / closed / path colours
  python stochastic_astar.py single --data data/test.npz --index 5 --out search.png
  # an RGB image of your own
  python stochastic_astar.py single --image map.png --out search.png
  # compare against plain A* on in-distribution and OOD sets, several settings at once
  python stochastic_astar.py eval --sets rect=data/test.npz,maze=data_maze/test.npz \
      --heuristics minkowski,cnn:runs/heuristics --guidance const:1,ckpt:runs/dist/best.pt --n 200
"""
import argparse
import heapq
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
MINKOWSKI_PS = (1, 2, 5, 10, 100)

# colours of the result picture
COL = {"wall": (0, 0, 0), "free": (25, 25, 90), "closed": (255, 150, 0), "open": (0, 215, 215),
       "path": (255, 255, 255), "start": (0, 255, 0), "goal": (255, 0, 0)}


# ======================================================================================
# 1. RGB map  <->  (free, start, goal)
# ======================================================================================
def parse_rgb_map(img):
    """(H, W, 3) uint8 -> (free bool (H, W), start (r, c), goal (r, c)).

    green  = start (a green + blue pixel, as view_example.py draws it, is accepted too)
    red    = goal  (red + blue accepted too)
    any other non-black pixel = free space (blue); black = wall.
    """
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("expected an (H, W, 3) RGB image")
    r, g, b = (img[..., i].astype(int) for i in range(3))
    start_m = (g > 127) & (r < 128)
    goal_m = (r > 127) & (g < 128)
    if start_m.sum() != 1 or goal_m.sum() != 1:
        raise ValueError(f"need exactly one green (start) and one red (goal) pixel, found "
                         f"{int(start_m.sum())} and {int(goal_m.sum())}")
    free = (r + g + b > 0)
    start = tuple(int(x) for x in np.argwhere(start_m)[0])
    goal = tuple(int(x) for x in np.argwhere(goal_m)[0])
    free[start] = free[goal] = True
    return free, start, goal


def map_to_rgb(free, start, goal):
    """Inverse of parse_rgb_map with pure colours: blue free space, pure green start, pure red goal."""
    free = np.asarray(free) > 0.5
    rgb = np.zeros(free.shape + (3,), dtype=np.uint8)
    rgb[..., 2] = free.astype(np.uint8) * 255
    rgb[start] = (0, 255, 0)
    rgb[goal] = (255, 0, 0)
    return rgb


# ======================================================================================
# 2. Heuristic sources: each returns a (K, H, W) stack, one map per heuristic, computed once per problem
# ======================================================================================
def minkowski_map(shape, goal, p):
    """(sum |d|^p)^(1/p) to the goal, computed in a numerically safe way (p = 100 would overflow otherwise)."""
    H, W = shape
    dy = np.abs(np.arange(H, dtype=np.float64)[:, None] - goal[0]) * np.ones((1, W))
    dx = np.abs(np.arange(W, dtype=np.float64)[None, :] - goal[1]) * np.ones((H, 1))
    m = np.maximum(dx, dy)
    safe = np.where(m > 0, m, 1.0)
    out = m * ((dy / safe) ** p + (dx / safe) ** p) ** (1.0 / p)
    return np.where(m > 0, out, 0.0)


class MinkowskiHeuristics:
    def __init__(self, ps: Sequence[float] = MINKOWSKI_PS):
        self.ps = tuple(ps)
        self.name = "minkowski(p=" + ",".join(f"{p:g}" for p in self.ps) + ")"

    def __call__(self, free, start, goal, phi):
        return np.stack([minkowski_map(free.shape, goal, p) for p in self.ps])


class CNNHeuristics:
    """Ensemble of HeuristicNet members from train_heuristic_ensemble.py (one heuristic map per member)."""

    def __init__(self, source, device="cpu"):
        import torch
        self.torch = torch
        if isinstance(source, (str, os.PathLike)):
            try:
                from train_heuristic_ensemble import load_ensemble
            except ImportError as err:
                raise ImportError("train_heuristic_ensemble.py must be in the same folder as this file") from err
            self.nets = load_ensemble(str(source), device)
            self.name = f"cnn({len(self.nets)} members, {os.path.basename(str(source).rstrip('/'))})"
        else:
            self.nets = list(source)
            self.name = f"cnn({len(self.nets)} members)"
        self.device = device

    def __call__(self, free, start, goal, phi):
        t = self.torch
        H, W = free.shape
        f = t.from_numpy(free.astype(np.float32)).view(1, 1, H, W).to(self.device)
        s = t.zeros(1, 1, H, W, device=self.device)
        g = t.zeros(1, 1, H, W, device=self.device)
        s[0, 0, start[0], start[1]] = 1.0
        g[0, 0, goal[0], goal[1]] = 1.0
        p = t.from_numpy(phi.astype(np.float32)).view(1, 1, H, W).to(self.device)
        with t.no_grad():
            return np.stack([net.eval()(f, s, g, p)[0, 0].cpu().numpy() for net in self.nets]).astype(np.float64)


# ======================================================================================
# 3. Guidance (Phi) sources: callables (free, start, goal) -> (H, W) float array
# ======================================================================================
class ConstGuidance:
    def __init__(self, c: float):
        self.c = float(c)
        self.name = f"const({self.c:g})"

    def __call__(self, free, start, goal):
        return np.full(free.shape, self.c, dtype=np.float64)


class CheckpointGuidance:
    """Guidance map of a Neural A* checkpoint written by train.py (rebuilt exactly as test.py does)."""

    def __init__(self, path, device="cpu"):
        import torch
        self.torch = torch
        try:
            from neural_astar import NeuralAStar
        except ImportError as err:
            raise ImportError("neural_astar.py must be in the same folder as this file") from err
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = ck["config"]
        self.model = NeuralAStar(encoder_name=cfg["encoder_name"], tau=cfg["tau"], max_steps=cfg.get("max_steps"),
                                 encoder_depth=cfg.get("encoder_depth", 5), detach_g=cfg.get("detach_g", False),
                                 phi_bias=cfg.get("phi_bias"), dist_channels=cfg.get("dist_channels", False)).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.device = device
        self.name = f"learned({os.path.basename(path)})"

    def __call__(self, free, start, goal):
        t = self.torch
        H, W = free.shape
        m = t.from_numpy(free.astype(np.float32)).view(1, 1, H, W).to(self.device)
        s = t.zeros(1, 1, H, W, device=self.device)
        g = t.zeros(1, 1, H, W, device=self.device)
        s[0, 0, start[0], start[1]] = 1.0
        g[0, 0, goal[0], goal[1]] = 1.0
        with t.no_grad():
            return self.model.guidance(m, s, g)[0, 0].cpu().numpy().astype(np.float64)


# ======================================================================================
# 4. The searches
# ======================================================================================
@dataclass
class SearchResult:
    found: bool
    path: List[Tuple[int, int]]                 # start ... goal (empty if not found)
    closed: np.ndarray                          # (H, W) bool, expanded nodes
    open: np.ndarray                            # (H, W) bool, discovered but not expanded (final open set)
    iterations: int = 0
    cost: float = math.inf                      # sum of Phi over the cells entered along the path
    ks: List[int] = field(default_factory=list)  # K used in every iteration (stochastic search only)
    stds: List[float] = field(default_factory=list)  # std that produced that K (stochastic search only)
    order: Optional[np.ndarray] = None          # (H, W) int, iteration at which a cell was expanded, -1 = never

    @property
    def n_closed(self):
        return int(self.closed.sum())

    @property
    def n_open(self):
        return int(self.open.sum())

    @property
    def path_len(self):                         # in cells, start and goal included (same as the dataset's `lengths`)
        return len(self.path)


def _backtrack(parent, goal):
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    return path[::-1]


def sigmoid_k(std, k_min, k_max_eff, ref_std, c, k_dir="low-std-large-k"):
    """K = floor(K_min + (K_max - K_min) * frac), clipped to [K_min, K_max].

    k_dir = 'low-std-large-k' (default):  frac = 1 / (1 + exp( std/ref_std - c))
        members agree (std ~ 0)  -> K near K_max -> D almost surely holds the best node -> near-greedy best-first
        members disagree (std big) -> K near K_min -> D is a rank-weighted random draw   -> more exploration
    k_dir = 'high-std-large-k' (previous behaviour): frac = 1 / (1 + exp(-std/ref_std + c))
        the mirror image: disagreement -> large K (greedy), agreement -> small K (random).
    In both cases c > 0 pushes the std = 0 end of the curve towards its extreme (frac = 1/(1+e^-c) or 1/(1+e^c))."""
    k_lo = min(k_min, k_max_eff)
    if k_dir == "low-std-large-k":
        z = std / ref_std - c
    elif k_dir == "high-std-large-k":
        z = -std / ref_std + c
    else:
        raise ValueError(f"k_dir must be low-std-large-k or high-std-large-k, got {k_dir}")
    frac = 0.0 if z > 700 else 1.0 / (1.0 + math.exp(max(z, -700.0)))
    return int(min(max(math.floor(k_lo + (k_max_eff - k_lo) * frac), k_lo), k_max_eff))


def rank_probabilities(values, rank_power=1.0):
    """Rank the values ascending (rank 1 = lowest value) and map ranks to probabilities p ~ rank^-rank_power."""
    n = len(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)
    w = ranks ** (-rank_power)
    return w / w.sum()


def stochastic_astar(free, start, goal, heuristics: Callable, phi=None, *, k_min=1, k_max=20, ref_std=2.0,
                     c_sig=2.0, c_b=0.1, rank_power=1.0, reduce="mean", stop_on="generate", seed=0,
                     corner_cut=True, h_maps=None, std_source="best", k_dir="low-std-large-k") -> SearchResult:
    """Stochastic multi-heuristic A*, see the module docstring. `heuristics(free, start, goal, phi)` -> (K,H,W)."""
    free = np.asarray(free) > 0.5
    H, W = free.shape
    phi = np.ones((H, W)) if phi is None else np.asarray(phi, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if h_maps is None:
        h_maps = heuristics(free, start, goal, phi)
    hs = np.asarray(h_maps, dtype=np.float64)
    h_all = {"mean": hs.mean(0), "min": hs.min(0), "max": hs.max(0)}[reduce]
    member_std = hs.std(0)                       # (H, W): disagreement of the N heuristics at every cell
    if std_source not in ("best", "open-mean", "open-spatial"):
        raise ValueError(f"std_source must be best, open-mean or open-spatial, got {std_source}")

    N = H * W
    cap = N + 8
    rows = np.zeros(cap, dtype=np.int64)         # open set as parallel arrays; removal = swap with the last entry
    cols = np.zeros(cap, dtype=np.int64)
    g = np.zeros(cap)
    d = np.zeros(cap, dtype=np.int64)
    pos = -np.ones(N, dtype=np.int64)            # flat cell index -> slot in the open arrays, -1 = not open
    closed = np.zeros((H, W), dtype=bool)
    order = -np.ones((H, W), dtype=np.int64)
    parent = {start: None}
    n = 1
    rows[0], cols[0] = start
    pos[start[0] * W + start[1]] = 0
    ks, stds, it = [], [], 0
    found_via = None

    while n > 0:
        it += 1
        hv = h_all[rows[:n], cols[:n]]
        # --- value = h(reduced) + g + c_b * sqrt(d_max) / (1 + d), rank ascending ----------------------------
        d_max = int(d[:n].max())
        val = hv + g[:n] + c_b * math.sqrt(d_max) / (1.0 + d[:n])
        # --- std of the member heuristics at the lowest-value open node -> K ------------------------------
        if std_source == "best":
            b = int(np.argmin(val))
            sd = float(member_std[rows[b], cols[b]])
        elif std_source == "open-mean":
            sd = float(member_std[rows[:n], cols[:n]].mean())
        else:                                     # legacy: spread of the reduced h over different open cells
            sd = float(hv.std())
        k_max_eff = min(k_max, n)
        K = sigmoid_k(sd, k_min, k_max_eff, ref_std, c_sig, k_dir)
        ks.append(K)
        stds.append(sd)
        # --- rank -> probability; D = K distinct nodes; expand the lowest value in D ------------------------
        # --- D = K distinct nodes drawn with those probabilities; expand the lowest value in D -------------
        if K >= n:
            D_idx = np.arange(n)
        else:
            D_idx = rng.choice(n, size=K, replace=False, p=rank_probabilities(val, rank_power))
        sel = int(D_idx[np.argmin(val[D_idx])])
        r, c = int(rows[sel]), int(cols[sel])
        node = (r, c)
        g_node, d_node = g[sel], int(d[sel])

        if stop_on == "expand" and node == goal:
            found_via = goal
            break

        # --- remove the expanded node from open, add it to closed -------------------------------------------
        last = n - 1
        if sel != last:
            rows[sel], cols[sel], g[sel], d[sel] = rows[last], cols[last], g[last], d[last]
            pos[rows[sel] * W + cols[sel]] = sel
        pos[r * W + c] = -1
        n -= 1
        closed[r, c] = True
        order[r, c] = it

        # --- expand: neighbours join the open set (not if closed); the goal ends the search -----------------
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W) or not free[nr, nc] or closed[nr, nc]:
                continue
            if not corner_cut and dr != 0 and dc != 0 and not (free[r + dr, c] and free[r, c + dc]):
                continue
            ng = g_node + phi[nr, nc]
            if (nr, nc) == goal and stop_on == "generate":
                parent[goal] = node
                found_via = goal
                break
            slot = pos[nr * W + nc]
            if slot >= 0:                         # already open: keep the cheaper route
                if ng < g[slot] - 1e-12:
                    g[slot], d[slot] = ng, d_node + 1
                    parent[(nr, nc)] = node
            else:
                rows[n], cols[n], g[n], d[n] = nr, nc, ng, d_node + 1
                pos[nr * W + nc] = n
                n += 1
                parent[(nr, nc)] = node
        if found_via is not None:
            break

    open_mask = np.zeros((H, W), dtype=bool)
    open_mask[rows[:n], cols[:n]] = True
    if found_via is None:
        return SearchResult(False, [], closed, open_mask, it, math.inf, ks, stds, order)
    path = _backtrack(parent, goal)
    cost = float(sum(phi[p] for p in path[1:]))
    return SearchResult(True, path, closed, open_mask, it, cost, ks, stds, order)


def plain_astar(free, start, goal, phi=None, heuristic="chebyshev", corner_cut=True) -> SearchResult:
    """Deterministic textbook A* (expands the lowest f, stops when the goal is expanded). This is the reference.

    heuristic: 'chebyshev' (admissible and consistent for unit costs; scaled by min Phi for other costs),
               'minkowski' (mean of the 5 Minkowski distances, the same h the stochastic search gets), 'zero'."""
    free = np.asarray(free) > 0.5
    H, W = free.shape
    phi = np.ones((H, W)) if phi is None else np.asarray(phi, dtype=np.float64)
    if heuristic == "chebyshev":
        cmin = float(phi[free].min())
        hm = cmin * minkowski_map((H, W), goal, 1000.0)
    elif heuristic == "minkowski":
        hm = np.mean([minkowski_map((H, W), goal, p) for p in MINKOWSKI_PS], axis=0)
    else:
        hm = np.zeros((H, W))
    h_l, free_l, phi_l = hm.tolist(), free.tolist(), phi.tolist()
    best = {start: 0.0}
    parent = {start: None}
    closed = np.zeros((H, W), dtype=bool)
    order = -np.ones((H, W), dtype=np.int64)
    heap = [(h_l[start[0]][start[1]], 0.0, start)]
    it, found = 0, False
    while heap:
        f, gv, node = heapq.heappop(heap)
        if closed[node] or gv > best.get(node, math.inf) + 1e-12:
            continue
        it += 1
        closed[node] = True
        order[node] = it
        if node == goal:
            found = True
            break
        r, c = node
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W) or not free_l[nr][nc] or closed[nr, nc]:
                continue
            if not corner_cut and dr != 0 and dc != 0 and not (free_l[r + dr][c] and free_l[r][c + dc]):
                continue
            ng = gv + phi_l[nr][nc]
            if ng < best.get((nr, nc), math.inf) - 1e-12:
                best[(nr, nc)] = ng
                parent[(nr, nc)] = node
                heapq.heappush(heap, (ng + h_l[nr][nc], ng, (nr, nc)))
    open_mask = np.zeros((H, W), dtype=bool)
    for k in best:
        open_mask[k] = True
    open_mask &= ~closed
    if not found:
        return SearchResult(False, [], closed, open_mask, it, order=order)
    path = _backtrack(parent, goal)
    return SearchResult(True, path, closed, open_mask, it, float(sum(phi[p] for p in path[1:])), order=order)


def validate_path(path, free, start, goal, corner_cut=True):
    """True if the path starts at start, ends at goal, only uses free cells and 8-adjacent steps."""
    if not path or path[0] != start or path[-1] != goal:
        return False
    for (r, c), (nr, nc) in zip(path[:-1], path[1:]):
        if max(abs(nr - r), abs(nc - c)) != 1 or not free[nr, nc]:
            return False
        if not corner_cut and nr != r and nc != c and not (free[nr, c] and free[r, nc]):
            return False
    return bool(free[start]) and bool(free[goal])


# ======================================================================================
# 5. Picture: open set vs closed set vs path
# ======================================================================================
def result_to_rgb(free, start, goal, res: SearchResult):
    img = np.zeros(free.shape + (3,), dtype=np.uint8)
    img[free] = COL["free"]
    img[res.closed] = COL["closed"]
    img[res.open] = COL["open"]
    for p in res.path:
        img[p] = COL["path"]
    img[start], img[goal] = COL["start"], COL["goal"]
    return img


def render_search(free, start, goal, res: SearchResult, out_path, title="", ref: Optional[SearchResult] = None,
                  ref_name="plain A*"):
    """2x2 picture. Top: the stochastic result and the reference, in the colours above (legend with counts).
    Bottom left: expansion order of the stochastic search (purple = early, yellow = late, path in white).
    Bottom right: K (steps) and the member std (line) in every iteration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11.8))
    panels = [(title or "stochastic A*", res)] + ([(ref_name, ref)] if ref is not None else [])
    for ax, (name, r) in zip(axes[0], panels):
        ax.imshow(result_to_rgb(free, start, goal, r), interpolation="nearest")
        ax.set_title(f"{name}\nexpanded {r.n_closed} | open {r.n_open} | "
                     f"path {r.path_len if r.found else 'not found'} cells", fontsize=9, pad=8)
        ax.set_xticks([]), ax.set_yticks([])
    for ax in axes[0][len(panels):]:
        ax.axis("off")

    ax = axes[1][0]                                   # expansion order
    base = np.zeros(free.shape + (3,))
    base[free] = (0.12, 0.12, 0.12)
    ax.imshow(base, interpolation="nearest")
    if res.order is not None and res.n_closed:
        om = np.ma.masked_where(res.order < 0, res.order)
        im = ax.imshow(om, cmap="viridis", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="iteration at which the node was expanded")
    if res.path:
        ax.plot([q[1] for q in res.path], [q[0] for q in res.path], color="white", lw=1.6)
    ax.plot(start[1], start[0], "o", color="lime", ms=7, mec="k")
    ax.plot(goal[1], goal[0], "o", color="red", ms=7, mec="k")
    ax.set_title("order of expansion (stochastic search)", fontsize=9, pad=8)
    ax.set_xticks([]), ax.set_yticks([])

    ax = axes[1][1]                                   # K and std per iteration
    if res.ks:
        its = np.arange(1, len(res.ks) + 1)
        ax.step(its, res.ks, where="post", color="tab:orange", label="K")
        ax.set_xlabel("iteration"), ax.set_ylabel("K (nodes drawn into D)", color="tab:orange")
        ax2 = ax.twinx()
        ax2.plot(its, res.stds, color="tab:blue", lw=1, label="std")
        ax2.set_ylabel("std of the member heuristics", color="tab:blue")
        ax.set_title(f"K and std per iteration (mean K {np.mean(res.ks):.2f}, mean std {np.mean(res.stds):.3f})",
                     fontsize=9, pad=8)
    else:
        ax.axis("off")

    handles = [Patch(facecolor=np.array(COL[k]) / 255.0, edgecolor="0.5", label=lab) for k, lab in
               (("path", "path"), ("closed", "closed = expanded (searched)"), ("open", "open = discovered only"),
                ("free", "free, never touched"), ("wall", "wall"), ("start", "start"), ("goal", "goal"))]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_scatter(ref_closed, closed, out_path, title=""):
    """Per map: nodes expanded by plain A* (x) vs stochastic search (y). Below the diagonal = fewer than plain A*."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y = np.asarray(ref_closed, float), np.asarray(closed, float)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(x, y, s=10, alpha=0.6)
    top = max(x.max(), y.max()) * 1.05
    ax.plot([0, top], [0, top], "k--", lw=1)
    ax.set_xlim(0, top), ax.set_ylim(0, top)
    ax.set_xlabel("plain A*: nodes expanded"), ax.set_ylabel("stochastic A*: nodes expanded")
    ax.set_title(f"{title}\n{np.mean(y < x) * 100:.0f}% of maps below the diagonal (fewer nodes than A*), "
                 f"{np.mean(y > x) * 100:.0f}% above", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _slug(text):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)[:60]


# ======================================================================================
# 6. Evaluation harness: stochastic A* vs plain A*, per dataset / guidance / heuristic
# ======================================================================================
def load_npz(path):
    d = np.load(path)
    maps = d["maps"] > 0.5
    starts = [tuple(int(x) for x in s) for s in d["starts"]]
    goals = [tuple(int(x) for x in g) for g in d["goals"]]
    lengths = d["lengths"] if "lengths" in d.files else d["paths"].reshape(len(maps), -1).sum(1)
    return maps, starts, goals, np.asarray(lengths)


def make_heuristics(spec, device):
    if spec == "minkowski":
        return MinkowskiHeuristics()
    if spec.startswith("cnn:"):
        return CNNHeuristics(spec[4:], device)
    raise SystemExit(f"unknown heuristic source '{spec}' (use minkowski or cnn:DIR)")


def make_guidance(spec, device):
    if spec.startswith("const:"):
        return ConstGuidance(float(spec[6:]))
    if spec.startswith("ckpt:"):
        return CheckpointGuidance(spec[5:], device)
    raise SystemExit(f"unknown guidance source '{spec}' (use const:C or ckpt:PATH)")


def run_eval(args):
    device = args.device
    sets = [s.split("=", 1) if "=" in s else (os.path.basename(s), s) for s in args.sets.split(",")]
    heurs = [make_heuristics(h.strip(), device) for h in args.heuristics.split(",")]
    guides = [make_guidance(g.strip(), device) for g in args.guidance.split(",")]
    kw = dict(k_min=args.k_min, k_max=args.k_max, ref_std=args.ref_std, c_sig=args.c_sig, c_b=args.c_b,
              rank_power=args.rank_power, reduce=args.reduce, stop_on=args.stop_on, corner_cut=not args.no_corner_cut,
              std_source=args.std_source, k_dir=args.k_dir)
    rows = []
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
    for set_name, path in sets:
        maps, starts, goals, lengths = load_npz(path)
        n = min(args.n, len(maps)) if args.n else len(maps)
        print(f"\n=== {set_name}: {n} maps from {path}")
        # plain A* reference (Chebyshev, unit cost): its path length is optimal, its closed count is the baseline
        ref = []
        t0 = time.time()
        for i in range(n):
            ref.append(plain_astar(maps[i], starts[i], goals[i], corner_cut=not args.no_corner_cut))
        ref_closed = np.array([r.n_closed for r in ref], dtype=float)
        ref_len = np.array([r.path_len for r in ref], dtype=float)
        bad_ref = int(np.sum(ref_len > lengths[:n] + 1e-9))
        print(f"plain A* (Chebyshev, unit cost): closed {ref_closed.mean():.1f} | path {ref_len.mean():.1f} cells | "
              f"{int(np.sum(ref_len <= lengths[:n] + 1e-9))}/{n} match the dataset's shortest length"
              f"{'' if bad_ref == 0 else f'  (!! {bad_ref} longer than ground truth)'}  [{time.time() - t0:.1f}s]")
        rows.append({"set": set_name, "method": "plain A* (Chebyshev)", "guidance": "-", "n": n,
                     "success": 100.0, "opt": float(np.mean(ref_len <= lengths[:n] + 1e-9) * 100),
                     "len_ratio": float(np.mean(ref_len / lengths[:n])), "closed": float(ref_closed.mean()),
                     "touched": float(np.mean([r.n_closed + r.n_open for r in ref])), "exp": 0.0, "exp_raw": 0.0, "invalid": 0})
        print(f"{'guidance':22s} {'heuristic':34s} {'found%':>7s} {'Opt%':>6s} {'len/opt':>8s} {'closed':>8s} "
              f"{'touched':>8s} {'Exp%':>6s} {'ExpRaw%':>8s} {'iters':>7s} {'meanK':>6s} {'meanStd':>8s}")
        for gd in guides:
            for hu in heurs:
                if isinstance(hu, CNNHeuristics) and isinstance(gd, ConstGuidance):
                    print(f"  note: the CNN heuristics take Phi as an input; with {gd.name} they are out of the "
                          f"distribution they were trained on unless they were trained with --phi-const")
                found, opt, ratio, closed, touched, exp, iters, mk, invalid = [], [], [], [], [], [], [], [], 0
                exp_raw, msd, n_saved = [], [], 0
                t0 = time.time()
                for i in range(n):
                    phi = gd(maps[i], starts[i], goals[i])
                    res = stochastic_astar(maps[i], starts[i], goals[i], hu, phi, seed=args.seed + i, **kw)
                    found.append(res.found)
                    closed.append(res.n_closed)
                    touched.append(res.n_closed + res.n_open)
                    iters.append(res.iterations)
                    mk.append(np.mean(res.ks) if res.ks else 0.0)
                    msd.append(np.mean(res.stds) if res.stds else 0.0)
                    exp_raw.append(100.0 * (ref_closed[i] - res.n_closed) / max(ref_closed[i], 1.0))
                    exp.append(max(0.0, 100.0 * (ref_closed[i] - res.n_closed) / max(ref_closed[i], 1.0)))
                    if res.found:
                        if not validate_path(res.path, maps[i], starts[i], goals[i], not args.no_corner_cut):
                            invalid += 1
                        opt.append(res.path_len <= lengths[i] + 1e-9)
                        ratio.append(res.path_len / lengths[i])
                    else:
                        opt.append(False)
                    if args.vis_dir and n_saved < args.n_vis:
                        tag = f"{_slug(set_name)}_{_slug(gd.name)}_{_slug(hu.name)}_{args.reduce}"
                        render_search(maps[i], starts[i], goals[i], res, os.path.join(args.vis_dir, f"{tag}_map{i}.png"),
                                      title=f"stochastic A* ({hu.name}, {gd.name}, reduce={args.reduce})  map {i}",
                                      ref=ref[i])
                        n_saved += 1
                row = {"set": set_name, "method": f"stochastic | {hu.name}", "guidance": gd.name, "n": n,
                       "success": float(np.mean(found) * 100), "opt": float(np.mean(opt) * 100),
                       "len_ratio": float(np.mean(ratio)) if ratio else float("nan"),
                       "closed": float(np.mean(closed)), "touched": float(np.mean(touched)),
                       "exp": float(np.mean(exp)), "exp_raw": float(np.mean(exp_raw)),
                       "mean_k": float(np.mean(mk)), "mean_std": float(np.mean(msd)), "reduce": args.reduce,
                       "std_source": args.std_source, "invalid": invalid}
                rows.append(row)
                if args.vis_dir:
                    sp = os.path.join(args.vis_dir, f"scatter_{_slug(set_name)}_{_slug(gd.name)}_{_slug(hu.name)}_{args.reduce}.png")
                    render_scatter(ref_closed, closed, sp, f"{set_name} | {hu.name} | {gd.name} | reduce={args.reduce}")
                print(f"{gd.name:22s} {hu.name[:34]:34s} {row['success']:7.1f} {row['opt']:6.1f} "
                      f"{row['len_ratio']:8.3f} {row['closed']:8.1f} {row['touched']:8.1f} {row['exp']:6.1f} "
                      f"{row['exp_raw']:8.1f} {np.mean(iters):7.1f} {np.mean(mk):6.2f} {np.mean(msd):8.3f}   [{time.time() - t0:.0f}s]"
                      f"{'  !! invalid paths: ' + str(invalid) if invalid else ''}")
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nsaved {args.out_json}")
    print("\nfound%: runs that reached the goal | Opt%: path no longer than the dataset's shortest path | "
          "len/opt: mean path length / shortest | closed: expanded nodes | touched: closed + open | "
          "Exp%: mean per-map reduction of expanded nodes vs plain A* (floored at 0, as in test.py) | "
          "ExpRaw%: the same without the floor (negative = more nodes than plain A* on average) | "
          "meanStd: mean member std used for K (use it to choose --ref-std)")


def run_single(args):
    device = args.device
    if args.image:
        from PIL import Image
        free, start, goal = parse_rgb_map(np.array(Image.open(args.image).convert("RGB")))
        label = os.path.basename(args.image)
    else:
        maps, starts, goals, lengths = load_npz(args.data)
        i = int(np.random.randint(len(maps))) if args.random else args.index
        free, start, goal = maps[i], starts[i], goals[i]
        label = f"{os.path.basename(args.data)} #{i} (shortest path {int(lengths[i])} cells)"
    hu = make_heuristics(args.heuristics, device)
    gd = make_guidance(args.guidance, device)
    phi = gd(free, start, goal)
    res = stochastic_astar(free, start, goal, hu, phi, seed=args.seed, k_min=args.k_min, k_max=args.k_max,
                           ref_std=args.ref_std, c_sig=args.c_sig, c_b=args.c_b, rank_power=args.rank_power,
                           reduce=args.reduce, stop_on=args.stop_on, corner_cut=not args.no_corner_cut,
                           std_source=args.std_source, k_dir=args.k_dir)
    ref = plain_astar(free, start, goal, corner_cut=not args.no_corner_cut)
    print(f"{label}: start {start}, goal {goal}, free {free.mean() * 100:.0f}%")
    print(f"stochastic A* [{hu.name} | {gd.name}]: found={res.found} path={res.path_len} cells, "
          f"closed={res.n_closed}, open={res.n_open}, iterations={res.iterations}, mean K={np.mean(res.ks):.2f}, "
          f"mean member std={np.mean(res.stds):.3f} [reduce={args.reduce}, std-source={args.std_source}]")
    print(f"plain A*: path={ref.path_len} cells, closed={ref.n_closed}, open={ref.n_open}")
    print(f"path valid: {validate_path(res.path, free, start, goal, not args.no_corner_cut) if res.found else 'n/a'}")
    render_search(free, start, goal, res, args.out, title=f"stochastic A*  ({hu.name}, {gd.name}, reduce={args.reduce})", ref=ref)
    print(f"saved {args.out}")


def main():
    ap = argparse.ArgumentParser(description="Stochastic multi-heuristic A* vs plain A*")
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--k-min", type=int, default=1)
        p.add_argument("--k-max", type=int, default=20)
        p.add_argument("--ref-std", type=float, default=2.0, help="std scale in the K sigmoid")
        p.add_argument("--c-sig", type=float, default=2.0, help="offset c in the K sigmoid (c > 0: small K when std is small)")
        p.add_argument("--c-b", type=float, default=0.1, help="weight of the sqrt(d_max)/(1+d) term")
        p.add_argument("--rank-power", type=float, default=1.0, help="p(rank) ~ rank^-power (0 = uniform)")
        p.add_argument("--heuristic-reduce", "--reduce", dest="reduce", choices=["mean", "min", "max"], default="mean",
                       help="how the ensemble members are combined into h (same choices as test.py)")
        p.add_argument("--std-source", choices=["best", "open-mean", "open-spatial"], default="best",
                       help="best: std across the members at the lowest-value open node (default); open-mean: that "
                            "std averaged over the open set; open-spatial: old behaviour, std of h over open cells")
        p.add_argument("--k-dir", choices=["low-std-large-k", "high-std-large-k"], default="low-std-large-k",
                       help="low-std-large-k (default): K large when the members agree, small when they disagree; "
                            "high-std-large-k: the previous, opposite behaviour")
        p.add_argument("--stop-on", choices=["generate", "expand"], default="generate",
                       help="generate: stop when a neighbour is the goal (spec); expand: when the goal is selected")
        p.add_argument("--no-corner-cut", action="store_true", help="forbid diagonal moves between two walls")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", default="cpu")

    s = sub.add_parser("single", help="one map, with a picture")
    common(s)
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="64x64 RGB png: green start, red goal, blue free, black walls")
    g.add_argument("--data", help=".npz from make_synthetic_data.py")
    s.add_argument("--index", type=int, default=0)
    s.add_argument("--random", action="store_true")
    s.add_argument("--heuristics", default="minkowski", help="minkowski | cnn:DIR")
    s.add_argument("--guidance", default="const:1", help="const:C | ckpt:PATH")
    s.add_argument("--out", default="search.png")

    e = sub.add_parser("eval", help="compare against plain A* on one or more datasets")
    common(e)
    e.add_argument("--sets", required=True, help="comma list of NAME=file.npz (e.g. rect=data/test.npz,maze=data_maze/test.npz)")
    e.add_argument("--heuristics", default="minkowski", help="comma list: minkowski | cnn:DIR")
    e.add_argument("--guidance", default="const:1", help="comma list: const:C | ckpt:PATH")
    e.add_argument("--n", type=int, default=200, help="maps per set (0 = all)")
    e.add_argument("--out-json", default=None)
    e.add_argument("--vis-dir", default=None, help="write pictures here (first --n-vis maps per setting + scatter)")
    e.add_argument("--n-vis", type=int, default=4, help="pictures per set / guidance / heuristic")

    args = ap.parse_args()
    (run_single if args.mode == "single" else run_eval)(args)


if __name__ == "__main__":
    main()
