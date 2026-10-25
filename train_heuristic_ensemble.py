"""
Ensemble of heuristic CNNs h_k(v) for A*, trained under the step costs Phi of a trained Neural A* guidance map.

Cost model (same as the differentiable A*): moving onto cell v' costs Phi(v'), 8-neighbour grid, corner cutting
allowed.  g(v) = sum of Phi over the cells entered so far, f = g + h.

Targets, computed once per map (exact Dijkstra on the Phi-costs, cached in --out-dir):
    h*(v) = cheapest cost from v to the goal            (true cost-to-go; "the cost of the optimal path")
    g*(v) = cheapest cost from the start to v
    C*    = g*(goal) = cost of the optimal path under Phi
    on    = cells on some Phi-optimal path   (g* + h* == C*)
    off   = every other reachable free cell  (g* + h* >  C*)  -> off-path labels exist for free
--target given-path uses instead the *provided* optimal path: h*(path cell) = sum of Phi on the rest of that path.
    That only labels path cells (no off-path labels), and the provided path is optimal for unit costs, not
    necessarily for Phi, so its cost can exceed the true cost-to-go (the script prints how big that gap is).

Loss (all errors are measured in units of C*, so short and long routes weigh the same):
    e = (h - h*) / C*
    over_on   strong, asymmetric Huber on max(e,0) on path cells    (h > h*  breaks admissibility)
    over_off  weaker, same form, on off-path cells                  (harmless for optimality, keeps h sane)
    under     weak Huber on max(-e,0) on all labelled cells         (h << h* is safe but expands more nodes)
    cons      hinge  max(0, h(v) - Phi(v') - h(v')) over all free 8-neighbour edges   (consistency)
    goal      |h(goal)| / C*                                                         (anchor)
    foff      max(0, C* - g*(v) - h(v)) / C* on off-path cells   (f = g* + h should reach C*, so A* skips them;
              this bound is never above h*(v), so it cannot conflict with admissibility)
With weights w_over_on : w_under = 20 : 1 the asymmetric loss is a pinball loss whose optimum is the ~5%
quantile of h* given the input, i.e. h is deliberately conservative.

Network: small U-Net, input = [free, start, goal, cheb-dist-to-start, cheb-dist-to-goal, Phi*free],
    h = c_min * Chebyshev(v, goal) + h_scale * softplus(z)      (the first term is an admissible, consistent
    analytic bound, so every member starts from a safe heuristic and only learns the extra cost).

Input data (--train-data / --val-data / --test-data), any of
    .npz with maps (N,H,W; 1 = free), starts (N,2), goals (N,2), paths (N,H,W)       (make_synthetic_data.py)
    .npz with images (N,H,W,3 uint8 RGB), paths (N,H,W) or (N,H,W,3)
    folder with NAME.png (RGB map) and NAME_path.png (path; any non-black pixel = path)
RGB convention (change with --wall-rgb / --start-rgb / --goal-rgb / --color-tol): walls black, start pure green,
goal pure red, every other colour is free space.  Exactly one start and one goal pixel per image.

Examples
    python train_heuristic_ensemble.py --train-data data/train.npz --val-data data/val.npz \
        --phi-checkpoint runs/dist/last.pt --n-members 5 --out-dir runs/heuristics
    python train_heuristic_ensemble.py --mode inspect --val-data data/val.npz --out-dir runs/heuristics --n-vis 6
"""
import argparse
import csv
import glob
import heapq
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ======================================================================================
# 1. Data loading (RGB images / npz / folder)
# ======================================================================================
def _rgb(s):
    v = tuple(int(x) for x in s.split(","))
    assert len(v) == 3, "colours are given as R,G,B"
    return v


def _match(img, rgb, tol):
    return np.abs(img.astype(np.int16) - np.array(rgb, dtype=np.int16)).max(axis=-1) <= tol


def parse_rgb(images, wall_rgb, start_rgb, goal_rgb, tol):
    """(N,H,W,3) RGB -> free (N,H,W) uint8, starts (N,2), goals (N,2)."""
    images = np.asarray(images)
    if images.dtype != np.uint8:
        images = (np.clip(images, 0, 1) * 255).astype(np.uint8) if images.max() <= 1.0 else images.astype(np.uint8)
    N, H, W, _ = images.shape
    start_m, goal_m = _match(images, start_rgb, tol), _match(images, goal_rgb, tol)
    wall_m = _match(images, wall_rgb, tol) & ~start_m & ~goal_m
    out = []
    for name, m in (("start", start_m), ("goal", goal_m)):
        cnt = m.reshape(N, -1).sum(1)
        bad = np.flatnonzero(cnt != 1)
        if len(bad):
            raise SystemExit(f"{len(bad)} of {N} images do not contain exactly one {name} pixel "
                             f"(first: image {bad[0]} has {cnt[bad[0]]}). Check --{name}-rgb / --color-tol.")
        out.append(np.stack(np.unravel_index(m.reshape(N, -1).argmax(1), (H, W)), axis=1))
    return (~wall_m).astype(np.uint8), out[0].astype(np.int64), out[1].astype(np.int64)


def to_binary(a):
    a = np.asarray(a)
    if a.ndim == 4:                                  # (N,H,W,3) RGB mask or (N,1,H,W)
        a = a.max(axis=-1) if a.shape[-1] == 3 else a[:, 0]
    thr = 127 if (a.dtype == np.uint8 and a.max() > 1) else 0.5
    return (a > thr).astype(np.uint8)


def _load_png_dir(d):
    from PIL import Image
    files = sorted(f for f in glob.glob(os.path.join(d, "*.png")) if not f.endswith("_path.png"))
    if not files:
        raise SystemExit(f"no NAME.png files found in {d}")
    imgs, pths = [], []
    for f in files:
        pf = f[:-4] + "_path.png"
        if not os.path.exists(pf):
            raise SystemExit(f"missing path image {pf}")
        imgs.append(np.array(Image.open(f).convert("RGB")))
        pths.append(np.array(Image.open(pf).convert("RGB")))
    return np.stack(imgs), np.stack(pths)


def load_dataset(path, args):
    if os.path.isdir(path):
        images, paths = _load_png_dir(path)
        free, starts, goals = parse_rgb(images, args.wall_rgb, args.start_rgb, args.goal_rgb, args.color_tol)
    else:
        d = np.load(path)
        paths = d["paths"]
        if "images" in d.files:
            free, starts, goals = parse_rgb(d["images"], args.wall_rgb, args.start_rgb, args.goal_rgb, args.color_tol)
        else:
            maps = np.asarray(d["maps"])
            maps = maps[:, 0] if maps.ndim == 4 else maps
            free = maps > 0.5
            free = (~free if args.invert_map else free).astype(np.uint8)
            starts, goals = np.asarray(d["starts"], dtype=np.int64), np.asarray(d["goals"], dtype=np.int64)
    paths = to_binary(paths)
    if paths.shape != free.shape:
        raise SystemExit(f"paths {paths.shape} and maps {free.shape} must have the same shape")
    N = len(free)
    ar = np.arange(N)
    if not free[ar, starts[:, 0], starts[:, 1]].all() or not free[ar, goals[:, 0], goals[:, 1]].all():
        raise SystemExit("some starts/goals lie on obstacles (check --invert-map or the RGB colour convention)")
    if (starts == goals).all(axis=1).any():
        raise SystemExit("start == goal in some samples")
    return {"free": free, "starts": starts, "goals": goals, "paths": paths}


# ======================================================================================
# 2. Phi (guidance map of the trained Neural A*)
# ======================================================================================
@torch.no_grad()
def compute_phi(free, starts, goals, args, device, bs=64):
    N, H, W = free.shape
    if args.phi_const is not None:
        return np.full((N, H, W), args.phi_const, dtype=np.float32)
    from neural_astar import NeuralAStar
    ckpt = torch.load(args.phi_checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = NeuralAStar(encoder_name=cfg["encoder_name"], tau=cfg["tau"], max_steps=cfg.get("max_steps"),
                        encoder_depth=cfg.get("encoder_depth", 5), detach_g=cfg.get("detach_g", False),
                        phi_bias=cfg.get("phi_bias"), dist_channels=cfg.get("dist_channels", False)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    out = np.empty((N, H, W), dtype=np.float32)
    for i in range(0, N, bs):
        m = torch.from_numpy(free[i:i + bs].astype(np.float32)).unsqueeze(1).to(device)
        s, g = torch.zeros_like(m), torch.zeros_like(m)
        idx = torch.arange(m.shape[0], device=device)
        st = torch.from_numpy(starts[i:i + bs]).to(device)
        gl = torch.from_numpy(goals[i:i + bs]).to(device)
        s[idx, 0, st[:, 0], st[:, 1]] = 1.0
        g[idx, 0, gl[:, 0], gl[:, 1]] = 1.0
        out[i:i + bs] = model.guidance(m, s, g)[:, 0].cpu().numpy()
    return out


# ======================================================================================
# 3. Exact targets (numpy): Dijkstra on the Phi-costs, ordered given path, reference A*
# ======================================================================================
def _dijkstra(free, phi, src, reverse):
    """reverse=False: g[v] = cheapest src -> v, summing Phi of every cell entered (src excluded).
    reverse=True : d[v] = cheapest v -> src, summing Phi of every cell entered (v excluded, src included)."""
    H, W = free.shape
    free_l, phi_l = free.tolist(), phi.tolist()
    dist = [[math.inf] * W for _ in range(H)]
    dist[src[0]][src[1]] = 0.0
    heap = [(0.0, src[0], src[1])]
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r][c]:
            continue
        base = d + phi_l[r][c] if reverse else d
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and free_l[nr][nc]:
                nd = base if reverse else base + phi_l[nr][nc]
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))
    return np.array(dist)


def order_path(path, start, goal):
    """Cells of a single shortest-path mask in walking order, or None if the mask is not one chain."""
    H, W = path.shape
    p = path.astype(bool)
    if not (p[start] and p[goal]):
        return None
    seen = {start}
    order, frontier = [start], [start]
    while frontier:
        nxt = []
        for r, c in frontier:
            for dr, dc in NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and p[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    nxt.append((nr, nc))
        if len(nxt) > 1:
            return None
        frontier = nxt
        order.extend(nxt)
    if order[-1] != goal or len(order) != int(p.sum()):
        return None
    return order


def _target_job(job):
    free, phi, start, goal, path, mode, tol = job
    start, goal = tuple(int(x) for x in start), tuple(int(x) for x in goal)
    shape = free.shape
    g = _dijkstra(free, phi, start, reverse=False)
    h = _dijkstra(free, phi, goal, reverse=True)
    C = float(g[goal])
    if not math.isfinite(C):                                   # goal unreachable: sample carries no labels
        z = np.zeros(shape, dtype=np.float32)
        zb = np.zeros(shape, dtype=bool)
        return z, z, 1.0, zb, zb, zb, 1.0, math.nan
    reach = np.isfinite(g) & np.isfinite(h)
    on = reach & (g + h <= C * (1 + tol) + 1e-6)
    order = order_path(path, start, goal)
    seq = np.array([phi[r, c] for r, c in order], dtype=np.float64) if order is not None else None
    C_given = float(seq[1:].sum()) if order is not None else math.nan
    if mode == "given-path":
        hh, gg = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
        m = np.zeros(shape, dtype=bool)
        if order is not None:
            suffix = np.concatenate([np.cumsum(seq[::-1])[::-1][1:], [0.0]])      # sum of Phi after cell i
            prefix = np.concatenate([[0.0], np.cumsum(seq[1:])])                  # sum of Phi up to cell i
            for (r, c), a, b in zip(order, suffix, prefix):
                hh[r, c], gg[r, c], m[r, c] = a, b, True
        return hh, gg, (C_given if order is not None else 1.0), m, m.copy(), np.zeros(shape, dtype=bool), C, C_given
    hh = np.where(reach, h, 0.0).astype(np.float32)
    gg = np.where(reach, g, 0.0).astype(np.float32)
    return hh, gg, C, on, reach, reach & ~on, C, C_given


def astar_np(free, phi, start, goal, h):
    """Reference A* (Phi step costs, heuristic map h, re-expansion allowed). Returns (cost, expanded nodes)."""
    H, W = free.shape
    free_l, phi_l, h_l = free.tolist(), phi.tolist(), h.tolist()
    best = {start: 0.0}
    heap = [(h_l[start[0]][start[1]], 0.0, start[0], start[1])]
    expanded = 0
    while heap:
        f, neg_g, r, c = heapq.heappop(heap)
        gv = -neg_g
        if gv > best.get((r, c), math.inf):
            continue
        expanded += 1
        if (r, c) == goal:
            return gv, expanded
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and free_l[nr][nc]:
                ng = gv + phi_l[nr][nc]
                if ng < best.get((nr, nc), math.inf) - 1e-12:
                    best[(nr, nc)] = ng
                    heapq.heappush(heap, (ng + h_l[nr][nc], -ng, nr, nc))
    return math.inf, expanded


def prepare_split(name, raw, args, device):
    """Phi + exact targets for one split, cached in out_dir/cache_<name>.npz."""
    N = len(raw["free"])
    src = "const:%s" % args.phi_const if args.phi_const is not None else \
        "%s|%s" % (os.path.abspath(args.phi_checkpoint), os.path.getmtime(args.phi_checkpoint))
    meta = f"{src}|{args.target}|{N}|{raw['paths'].sum()}|{raw['starts'].sum()}"
    cache = os.path.join(args.out_dir, f"cache_{name}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        if str(d["meta"]) == meta:
            print(f"[{name}] loaded cached targets from {cache}")
            return {k: d[k] for k in d.files if k != "meta"}
    print(f"[{name}] computing Phi and exact targets for {N} maps ...", flush=True)
    t0 = time.time()
    phi = compute_phi(raw["free"], raw["starts"], raw["goals"], args, device)
    jobs = [(raw["free"][i], phi[i], raw["starts"][i], raw["goals"][i], raw["paths"][i], args.target, 1e-4)
            for i in range(N)]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            res = []
            for i, r in enumerate(ex.map(_target_job, jobs, chunksize=8), 1):
                res.append(r)
                if i % 500 == 0:
                    print(f"   {i}/{N}  ({time.time() - t0:.0f}s)", flush=True)
    else:
        res = [_target_job(j) for j in jobs]
    h, g, C, on, lab, off, C_opt, C_given = zip(*res)
    d = {"free": raw["free"], "starts": raw["starts"], "goals": raw["goals"], "phi": phi,
         "h_star": np.stack(h), "g_star": np.stack(g), "C": np.array(C, dtype=np.float32),
         "on": np.stack(on).astype(np.uint8), "lab": np.stack(lab).astype(np.uint8),
         "off": np.stack(off).astype(np.uint8), "C_opt": np.array(C_opt, dtype=np.float32),
         "C_given": np.array(C_given, dtype=np.float32)}
    np.savez_compressed(cache, meta=np.array(meta), **d)
    print(f"[{name}] done in {time.time() - t0:.0f}s, cached to {cache}")
    return d


def report_targets(name, d, mode):
    C_opt, C_given = d["C_opt"], d["C_given"]
    ok = np.isfinite(C_given)
    msg = f"[{name}] mean optimal cost under Phi C* = {C_opt.mean():.2f}"
    if ok.any():
        gap = C_given[ok] / C_opt[ok] - 1.0
        msg += (f" | provided path is a single chain in {ok.mean() * 100:.0f}% of maps; its Phi-cost is on average "
                f"{gap.mean() * 100:.2f}% above C*, exactly optimal in {(gap < 1e-4).mean() * 100:.0f}%")
    else:
        msg += " | provided paths are not single chains (union labels?) -> --target given-path unavailable"
    print(msg)
    if mode == "given-path" and not ok.all():
        print(f"   WARNING: {int((~ok).sum())} maps have no usable ordered path; they only get the consistency/goal terms")
    print(f"   labelled cells per map: on-path {d['on'].sum((1, 2)).mean():.0f}, off-path {d['off'].sum((1, 2)).mean():.0f}")


# ======================================================================================
# 4. Dataset (torch)
# ======================================================================================
class HeuristicDataset(Dataset):
    KEYS = ["free", "start", "goal", "phi", "h_star", "g_star", "on", "lab", "off"]

    def __init__(self, d, augment=False):
        self.d, self.augment = d, augment
        if augment:
            assert d["free"].shape[1] == d["free"].shape[2], "augmentation needs square maps"

    def __len__(self):
        return len(self.d["free"])

    def __getitem__(self, i):
        d = self.d
        H, W = d["free"].shape[1:]
        start, goal = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
        start[tuple(d["starts"][i])] = 1.0
        goal[tuple(d["goals"][i])] = 1.0
        sp = torch.from_numpy(np.stack([d["free"][i], start, goal, d["phi"][i], d["h_star"][i], d["g_star"][i],
                                        d["on"][i], d["lab"][i], d["off"][i]]).astype(np.float32))
        if self.augment:                                    # one of the 8 symmetries, applied to everything
            sp = torch.rot90(sp, int(torch.randint(0, 4, (1,))), dims=(-2, -1))
            if int(torch.randint(0, 2, (1,))):
                sp = torch.flip(sp, dims=(-1,))
        out = {k: sp[j:j + 1] for j, k in enumerate(self.KEYS)}
        out["C"] = torch.tensor(float(d["C"][i]))
        return out


def collate(items):
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


# ======================================================================================
# 5. Model
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
    """c_min * Chebyshev(v, goal): admissible and consistent for any Phi-costs on the 8-neighbour grid."""
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
        nn.init.constant_(self.out.bias, -4.0)             # start close to the analytic lower bound

    def forward(self, x):
        feats = [self.enc[0](x)]
        for e in self.enc[1:]:
            feats.append(e(F.max_pool2d(feats[-1], 2)))
        h = feats[-1]
        for up, dec, skip in zip(self.up, self.dec, reversed(feats[:-1])):
            h = dec(torch.cat([skip, up(h)], dim=1))
        return self.out(h)


class HeuristicNet(nn.Module):
    """h(v) >= 0 for one (map, start, goal, Phi) problem; zero on obstacles."""

    def __init__(self, width=32, depth=4, h_scale=32.0):
        super().__init__()
        self.body = HeuristicUNet(6, width, depth)
        self.h_scale = h_scale

    def forward(self, free, start, goal, phi):
        W = free.shape[-1]
        x = torch.cat([free, start, goal, cheb_dist(start) / W, cheb_dist(goal) / W, phi * free], dim=1)
        h = lower_bound(free, goal, phi) + self.h_scale * F.softplus(self.body(x))
        return h * free


# ======================================================================================
# 6. Loss
# ======================================================================================
def huber(x, delta):                                         # x >= 0
    return torch.where(x < delta, 0.5 * x * x / delta, x - 0.5 * delta)


def masked_mean(x, mask):
    """Mean of x over mask inside each sample, then over the samples that have any masked cell."""
    num, den = (x * mask).flatten(1).sum(1), mask.flatten(1).sum(1)
    valid = (den > 0).float()
    return ((num / den.clamp(min=1.0)) * valid).sum() / valid.sum().clamp(min=1.0)


def shifted(x, dr, dc):
    """out[..., r, c] = x[..., r + dr, c + dc], zero outside the map."""
    H, W = x.shape[-2:]
    return F.pad(x, (1, 1, 1, 1))[..., 1 + dr:1 + dr + H, 1 + dc:1 + dc + W]


def consistency_loss(h, phi, free, scale):
    """Hinge on h(v) <= Phi(v') + h(v') for every free 8-neighbour pair (cost of the move v -> v' is Phi(v'))."""
    tot, cnt = 0.0, 0.0
    for dr, dc in NEIGHBORS:
        edge = free * shifted(free, dr, dc)
        viol = F.relu(h - shifted(phi, dr, dc) - shifted(h, dr, dc)) / scale
        tot = tot + (viol * edge).flatten(1).sum(1)
        cnt = cnt + edge.flatten(1).sum(1)
    return (tot / cnt.clamp(min=1.0)).mean()


def heuristic_loss(h, b, w, delta):
    free, goal, phi = b["free"], b["goal"], b["phi"]
    C = b["C"].view(-1, 1, 1, 1)
    scale = C.clamp(min=1.0)
    hs, gs, on, lab, off = b["h_star"], b["g_star"], b["on"], b["lab"], b["off"]
    e = (h - hs) / scale
    over, under = F.relu(e), F.relu(-e)
    parts = {
        "over_on": masked_mean(huber(over, delta), on),
        "over_off": masked_mean(huber(over, delta), off),
        "under": masked_mean(huber(under, delta), lab),
        "cons": consistency_loss(h, phi, free, scale),
        "goal": ((h * goal).flatten(1).sum(1) / scale.flatten()).abs().mean(),
        "foff": masked_mean(F.relu(C - gs - h) / scale, off),
    }
    total = sum(w[k] * v for k, v in parts.items())
    return total, parts


def batch_stats(h, b, tol=0.005):
    """(numerator, denominator) pairs for diagnostics that are accumulated over batches."""
    free, phi, goal = b["free"], b["phi"], b["goal"]
    C = b["C"].view(-1, 1, 1, 1)
    scale = C.clamp(min=1.0)
    hs, gs, on, lab, off = b["h_star"], b["g_star"], b["on"], b["lab"], b["off"]
    e = (h - hs) / scale
    pos = (hs > 1e-3).float()
    s = {
        "over_rate_on": (((e > tol).float() * on).sum(), on.sum()),
        "over_mean_on": ((F.relu(e) * on).sum(), on.sum()),
        "tight_on": (((h / hs.clamp(min=1e-3)) * on * pos).sum(), (on * pos).sum()),
        "tight_lab": (((h / hs.clamp(min=1e-3)) * lab * pos).sum(), (lab * pos).sum()),
        "goal_abs": (((h * goal).flatten(1).sum(1) / scale.flatten()).abs().sum(), torch.tensor(float(h.shape[0]))),
        "foff_rate": (((gs + h < C * (1 - tol)).float() * off).sum(), off.sum()),
    }
    viol, edges = 0.0, 0.0
    for dr, dc in NEIGHBORS:
        edge = free * shifted(free, dr, dc)
        viol = viol + (((h - shifted(phi, dr, dc) - shifted(h, dr, dc)) > tol * scale).float() * edge).sum()
        edges = edges + edge.sum()
    s["cons_viol_rate"] = (viol, edges)
    return {k: (float(n), float(d)) for k, (n, d) in s.items()}


@torch.no_grad()
def eval_on_loader(h_fn, loader, device, w, delta):
    sums, nums, dens, n = {}, {}, {}, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        h = h_fn(b)
        total, parts = heuristic_loss(h, b, w, delta)
        bs = b["free"].shape[0]
        sums["loss"] = sums.get("loss", 0.0) + total.item() * bs
        for k, v in parts.items():
            sums[f"loss_{k}"] = sums.get(f"loss_{k}", 0.0) + v.item() * bs
        for k, (a, d) in batch_stats(h, b).items():
            nums[k], dens[k] = nums.get(k, 0.0) + a, dens.get(k, 0.0) + d
        n += bs
    res = {k: v / max(n, 1) for k, v in sums.items()}
    res.update({k: nums[k] / max(dens[k], 1e-9) for k in nums})
    return res


# ======================================================================================
# 7. Training one member / the ensemble
# ======================================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def member_width(k, args):
    if not args.vary_width:
        return args.width
    return max(8, int(round(args.width * (0.75, 1.0, 1.5)[k % 3] / 8)) * 8)


def train_member(k, train_ds, val_loader, args, w, device):
    set_seed(args.seed + 1000 * k)
    H = train_ds[0]["free"].shape[-1]
    assert H % (2 ** args.depth) == 0, f"map size {H} must be divisible by 2**depth = {2 ** args.depth}"
    cfg = {"width": member_width(k, args), "depth": args.depth, "h_scale": args.h_scale or H / 2.0}
    model = HeuristicNet(**cfg).to(device)
    ds = train_ds
    if args.bootstrap:                                       # bagging: every member sees a different resample
        rng = np.random.default_rng(args.seed + 1000 * k)
        ds = Subset(train_ds, rng.integers(0, len(train_ds), len(train_ds)).tolist())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps, warm = args.epochs * len(loader), len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total_steps - warm, 1))))
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\n=== member {k}: width {cfg['width']}, depth {cfg['depth']}, {n_par / 1e6:.2f}M params, "
          f"{len(ds)} training maps ===")
    log_path = os.path.join(args.out_dir, f"log_member_{k}.csv")
    cols = ["epoch", "train_loss", "val_loss", "val_over_rate_on", "val_over_mean_on", "val_tight_on",
            "val_cons_viol_rate", "val_goal_abs", "val_foff_rate"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(cols)
    best = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run, seen = time.time(), 0.0, 0
        for b in loader:
            b = {kk: v.to(device) for kk, v in b.items()}
            loss, _ = heuristic_loss(model(b["free"], b["start"], b["goal"], b["phi"]), b, w, args.huber_delta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item() * b["free"].shape[0]
            seen += b["free"].shape[0]
        model.eval()
        val = eval_on_loader(lambda b: model(b["free"], b["start"], b["goal"], b["phi"]), val_loader, device, w,
                             args.huber_delta)
        print(f"  epoch {epoch:3d} | train {run / seen:.4f} | val {val['loss']:.4f} | over(path) "
              f"{val['over_rate_on'] * 100:5.1f}% | tight(path) {val['tight_on']:.3f} | cons-viol "
              f"{val['cons_viol_rate'] * 100:5.2f}% | goal {val['goal_abs']:.4f} | f<C* off-path "
              f"{val['foff_rate'] * 100:5.1f}% | {time.time() - t0:.0f}s", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, run / seen, val["loss"], val["over_rate_on"], val["over_mean_on"],
                                    val["tight_on"], val["cons_viol_rate"], val["goal_abs"], val["foff_rate"]])
        if val["loss"] < best:
            best = val["loss"]
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val": val},
                       os.path.join(args.out_dir, f"member_{k}.pt"))
    print(f"  member {k}: best val loss {best:.4f}")


def load_ensemble(out_dir, device):
    nets = []
    for p in sorted(glob.glob(os.path.join(out_dir, "member_*.pt"))):
        ck = torch.load(p, map_location=device, weights_only=False)
        net = HeuristicNet(**ck["config"]).to(device)
        net.load_state_dict(ck["model"])
        net.eval()
        nets.append(net)
    if not nets:
        raise SystemExit(f"no member_*.pt found in {out_dir}")
    return nets


@torch.no_grad()
def ensemble_predict(nets, free, start, goal, phi):
    """(K,B,1,H,W) heuristic maps, one per member. Reduce with .amin(0) (safest), .mean(0) or .amax(0)."""
    return torch.stack([net(free, start, goal, phi) for net in nets])


# ======================================================================================
# 8. Evaluation: constraint diagnostics + real A* searches
# ======================================================================================
@torch.no_grad()
def evaluate_ensemble(nets, ds, device, args, w):
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    cand = {f"member_{k}": (lambda b, net=net: net(b["free"], b["start"], b["goal"], b["phi"]))
            for k, net in enumerate(nets)}
    cand["ens_min"] = lambda b: ensemble_predict(nets, b["free"], b["start"], b["goal"], b["phi"]).amin(0)
    cand["ens_mean"] = lambda b: ensemble_predict(nets, b["free"], b["start"], b["goal"], b["phi"]).mean(0)
    cand["ens_max"] = lambda b: ensemble_predict(nets, b["free"], b["start"], b["goal"], b["phi"]).amax(0)
    cand["cmin*Cheb (analytic)"] = lambda b: lower_bound(b["free"], b["goal"], b["phi"])
    cand["zero (Dijkstra)"] = lambda b: torch.zeros_like(b["free"])
    res = {name: eval_on_loader(fn, loader, device, w, args.huber_delta) for name, fn in cand.items()}
    print(f"\n{'heuristic':24s} {'over%(path)':>11s} {'over mean':>10s} {'tight(path)':>11s} {'cons-viol%':>10s} "
          f"{'|h(goal)|':>10s} {'f<C* off%':>10s}")
    for name, r in res.items():
        print(f"{name:24s} {r['over_rate_on'] * 100:11.2f} {r['over_mean_on']:10.4f} {r['tight_on']:11.3f} "
              f"{r['cons_viol_rate'] * 100:10.2f} {r['goal_abs']:10.4f} {r['foff_rate'] * 100:10.1f}")
    print("(over%: path cells with h > h* by more than 0.5% of C*; tight: mean h/h* on path cells, 1 = exact; "
          "f<C* off%: off-path cells A* would still expand)")
    return res


@torch.no_grad()
def evaluate_search(nets, ds, device, n):
    n = min(n, len(ds))
    if n <= 0:
        return {}
    b = {k: v.to(device) for k, v in collate([ds[i] for i in range(n)]).items()}
    H_all = ensemble_predict(nets, b["free"], b["start"], b["goal"], b["phi"])
    maps = {f"member_{k}": H_all[k] for k in range(len(nets))}
    maps.update({"ens_min": H_all.amin(0), "ens_mean": H_all.mean(0), "ens_max": H_all.amax(0),
                 "cmin*Cheb (analytic)": lower_bound(b["free"], b["goal"], b["phi"]),
                 "zero (Dijkstra)": torch.zeros_like(b["free"])})
    maps = {k: v[:, 0].cpu().numpy() for k, v in maps.items()}
    d, out = ds.d, {}
    print(f"\nA* with the learned heuristic and Phi step costs on {n} maps (same costs the guidance map defines):")
    print(f"{'heuristic':24s} {'expanded':>9s} {'vs zero':>8s} {'optimal%':>9s} {'cost/C*':>8s}")
    base = None
    for name, hm in maps.items():
        exp, opt, ratio = [], [], []
        for i in range(n):
            s, g = tuple(int(x) for x in d["starts"][i]), tuple(int(x) for x in d["goals"][i])
            cost, ne = astar_np(d["free"][i], d["phi"][i], s, g, hm[i])
            exp.append(ne)
            opt.append(cost <= float(d["C_opt"][i]) * (1 + 1e-4) + 1e-6)
            ratio.append(cost / float(d["C_opt"][i]))
        out[name] = {"expanded": float(np.mean(exp)), "optimal_pct": 100 * float(np.mean(opt)),
                     "cost_ratio": float(np.mean(ratio))}
        base = out[name]["expanded"] if name.startswith("zero") else base
    for name, r in out.items():
        rel = f"{100 * r['expanded'] / base:7.0f}%" if base else "      -"
        print(f"{name:24s} {r['expanded']:9.1f} {rel:>8s} {r['optimal_pct']:9.1f} {r['cost_ratio']:8.4f}")
    return out


# ======================================================================================
# 9. Looking at every member separately
# ======================================================================================
def visualize(nets, ds, device, out_dir, indices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = {k: v.to(device) for k, v in collate([ds[i] for i in indices]).items()}
    H_all = ensemble_predict(nets, b["free"], b["start"], b["goal"], b["phi"])[:, :, 0].cpu().numpy()   # (K,n,H,W)
    free, phi = b["free"][:, 0].cpu().numpy(), b["phi"][:, 0].cpu().numpy()
    hs, lab, on = b["h_star"][:, 0].cpu().numpy(), b["lab"][:, 0].cpu().numpy(), b["on"][:, 0].cpu().numpy()
    scale = b["C"].clamp(min=1.0).cpu().numpy()
    K, n, H, W = H_all.shape
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black")
    ecmap = plt.get_cmap("coolwarm").copy()
    ecmap.set_bad("black")
    nan = np.nan

    fig, ax = plt.subplots(n, 4 + K, figsize=(2.5 * (4 + K), 2.5 * n), squeeze=False)
    for r in range(n):
        s = np.unravel_index(b["start"][r, 0].argmax().item(), (H, W))
        g = np.unravel_index(b["goal"][r, 0].argmax().item(), (H, W))
        img = np.where(free[r][..., None] > 0.5, 0.85, 0.0) * np.ones((H, W, 3))
        img[s], img[g] = (0, 0.8, 0), (1, 0, 0)
        vmax = float(max(hs[r][lab[r] > 0.5].max(), H_all[:, r][:, free[r] > 0.5].max()))
        ax[r, 0].imshow(img)
        ax[r, 1].imshow(np.where(free[r] > 0.5, phi[r], nan), cmap=cmap, vmin=0, vmax=1)
        ax[r, 2].imshow(np.where(lab[r] > 0.5, hs[r], nan), cmap=cmap, vmin=0, vmax=vmax)
        for k in range(K):
            ax[r, 3 + k].imshow(np.where(free[r] > 0.5, H_all[k, r], nan), cmap=cmap, vmin=0, vmax=vmax)
        ax[r, 3 + K].imshow(np.where(free[r] > 0.5, H_all[:, r].std(0), nan), cmap=cmap, vmin=0, vmax=vmax / 4)
        titles = ["map (S green, G red)", "Phi (step cost)", "true cost-to-go h*"] + \
                 [f"member {k}" for k in range(K)] + ["spread (std of members)"]
        for j, t in enumerate(titles):
            if r == 0:
                ax[r, j].set_title(t, fontsize=8)
            ax[r, j].axis("off")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "ensemble_h_maps.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(n, K + 2, figsize=(2.5 * (K + 2), 2.5 * n), squeeze=False)
    variants = [H_all[k] for k in range(K)] + [H_all.min(0), H_all.max(0)]
    names = [f"member {k}" for k in range(K)] + ["ensemble min", "ensemble max"]
    for r in range(n):
        for j, (hv, nm) in enumerate(zip(variants, names)):
            e = (hv[r] - hs[r]) / scale[r]
            msk = on[r] > 0.5
            over = float((e[msk] > 0.005).mean() * 100) if msk.any() else float("nan")
            ax[r, j].imshow(np.where(lab[r] > 0.5, e, nan), cmap=ecmap, vmin=-0.3, vmax=0.3)
            ax[r, j].set_title(f"{nm}\nover on path: {over:.0f}%", fontsize=7)
            ax[r, j].axis("off")
    fig.suptitle("(h - h*) / C*   red = overestimate (breaks admissibility), blue = underestimate", fontsize=9)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "ensemble_errors.png")
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    np.savez_compressed(os.path.join(out_dir, "inspect_outputs.npz"), h=H_all, h_star=hs, phi=phi, free=free,
                        on=on, lab=lab, indices=np.array(indices))
    print(f"saved {p1}\nsaved {p2}\nsaved {os.path.join(out_dir, 'inspect_outputs.npz')} "
          f"(h: members x maps x H x W)")


# ======================================================================================
# 10. Main
# ======================================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Ensemble of heuristic CNNs for A* under learned (Phi) step costs")
    p.add_argument("--mode", choices=["train", "inspect"], default="train")
    p.add_argument("--train-data")
    p.add_argument("--val-data", required=True)
    p.add_argument("--test-data", default=None)
    p.add_argument("--invert-map", action="store_true", help="npz maps use 1 = obstacle")
    p.add_argument("--wall-rgb", type=_rgb, default=(0, 0, 0))
    p.add_argument("--start-rgb", type=_rgb, default=(0, 255, 0))
    p.add_argument("--goal-rgb", type=_rgb, default=(255, 0, 0))
    p.add_argument("--color-tol", type=int, default=60)
    p.add_argument("--phi-checkpoint", default=None, help="trained Neural A* checkpoint (best.pt / last.pt)")
    p.add_argument("--phi-const", type=float, default=None, help="use a constant Phi instead of a checkpoint")
    p.add_argument("--target", choices=["dijkstra", "given-path"], default="dijkstra")
    p.add_argument("--n-members", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--h-scale", type=float, default=None, help="scale of the learned residual (default W/2)")
    p.add_argument("--vary-width", action="store_true", help="members get widths 0.75x / 1x / 1.5x in turn")
    p.add_argument("--bootstrap", action="store_true", help="each member trains on a bootstrap resample")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--huber-delta", type=float, default=0.02)
    p.add_argument("--w-over-on", type=float, default=20.0, help="h > h* on path cells (strong)")
    p.add_argument("--w-over-off", type=float, default=2.0, help="h > h* on off-path cells")
    p.add_argument("--w-under", type=float, default=1.0, help="h < h* (weak)")
    p.add_argument("--w-cons", type=float, default=10.0, help="consistency hinge")
    p.add_argument("--w-goal", type=float, default=10.0, help="h(goal) != 0")
    p.add_argument("--w-foff", type=float, default=2.0, help="off-path cells with f = g* + h < C*")
    p.add_argument("--eval-search", type=int, default=100, help="maps used for the real A* comparison (0 = skip)")
    p.add_argument("--workers", type=int, default=4, help="processes for the Dijkstra targets")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    p.add_argument("--n-vis", type=int, default=6)
    p.add_argument("--indices", type=int, nargs="*", default=None, help="maps to plot (inspect mode)")
    p.add_argument("--out-dir", default="runs/heuristics")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.phi_checkpoint is None and args.phi_const is None:
        p.error("give --phi-checkpoint (the trained Neural A*) or --phi-const")
    if args.mode == "train" and not args.train_data:
        p.error("--train-data is required in train mode")
    return args


def run_full_eval(nets, d, name, args, device, w, out):
    ds = HeuristicDataset(d, augment=False)
    out[f"{name}_constraints"] = evaluate_ensemble(nets, ds, device, args, w)
    if args.eval_search > 0:
        out[f"{name}_search"] = evaluate_search(nets, ds, device, args.eval_search)
    return ds


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    w = {"over_on": args.w_over_on, "over_off": args.w_over_off, "under": args.w_under, "cons": args.w_cons,
         "goal": args.w_goal, "foff": args.w_foff}

    val_d = prepare_split("val", load_dataset(args.val_data, args), args, device)
    if args.mode == "inspect":
        nets = load_ensemble(args.out_dir, device)
        ds = HeuristicDataset(val_d, augment=False)
        idx = args.indices if args.indices else list(range(min(args.n_vis, len(ds))))
        visualize(nets, ds, device, args.out_dir, idx)
        return

    train_d = prepare_split("train", load_dataset(args.train_data, args), args, device)
    report_targets("train", train_d, args.target)
    report_targets("val", val_d, args.target)
    train_ds = HeuristicDataset(train_d, augment=not args.no_augment)
    val_loader = DataLoader(HeuristicDataset(val_d, augment=False), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    print(f"\ntraining {args.n_members} members | loss weights {w} | augmentation "
          f"{'off' if args.no_augment else 'on'} | target {args.target}")
    for k in range(args.n_members):
        train_member(k, train_ds, val_loader, args, w, device)

    nets = load_ensemble(args.out_dir, device)
    results = {}
    val_ds = run_full_eval(nets, val_d, "val", args, device, w, results)
    if args.test_data:
        test_d = prepare_split("test", load_dataset(args.test_data, args), args, device)
        run_full_eval(nets, test_d, "test", args, device, w, results)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    visualize(nets, val_ds, device, args.out_dir, list(range(min(args.n_vis, len(val_ds)))))


if __name__ == "__main__":
    main()
