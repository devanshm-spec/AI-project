"""
Neural A* (Yonetani et al., ICML 2021) for binary grid maps.

Contents
--------
make_heuristic        Chebyshev + 0.001 * Euclidean distance to the goal (fixed, not learned)
DifferentiableAStar   A* written with matrix ops so a loss on the search history reaches the guidance map
NeuralAStar           U-Net (VGG-16 backbone) encoder  ->  guidance map  ->  DifferentiableAStar
ConstPhi              Dummy "encoder" that outputs a constant guidance map (for baselines)
PlanningDataset       (map, start, goal, ground-truth path) samples
evaluate              Opt / Exp / Hmean metrics from the paper (+ guidance-map statistics for debugging)
constant_phi_baselines  Opt/Exp/Hmean of A* with a constant Phi: the "no learning" frontier
plan_single           Convenience wrapper for one numpy map

Conventions
-----------
* Maps are float tensors of shape (B, 1, H, W) with 1 = passable and 0 = obstacle.
* start_maps / goal_maps are one-hot (B, 1, H, W).
* Moves: 8-neighbour grid, diagonal corner cutting allowed (as in the paper's convolution-based expansion).

Changes vs. the first version
-----------------------------
* Default encoder is vgg16_bn (BatchNorm); a plain VGG-16 trained from scratch tends to output a
  near-constant guidance map, which then flips between "all 0" (greedy) and "all 1" (plain A*).
* G(v) is now differentiable w.r.t. the costs of the *ancestors* on its path (before, g(v*) was
  detached, so Phi(v) only received gradient through its own entry). `detach_g=True` restores that.
* PlanningDataset accepts an optional `lengths` array (length of a shortest path, in cells). It is
  used by Opt, so the ground-truth map can be something other than a single path.
* evaluate() also returns statistics of the guidance map (phi_mean / phi_sstd / phi_mstd).
* CenteredPhi: the encoder's logits are mean-centred per map over free cells and a fixed bias sets the
  global level of Phi. Reason: the straight-through/softmax surrogate gradient on a *global shift* of Phi
  has the wrong sign (measured: negative, while the true loss increases with Phi), which made the mean of
  Phi drift towards 1 (vanilla A*). Centering projects that component out; the net only learns the
  *relative* costs. Use phi_bias=None for the old behaviour (plain sigmoid output).
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# --------------------------------------------------------------------------------------
# Heuristic
# --------------------------------------------------------------------------------------
def make_heuristic(goal_maps: torch.Tensor, tb_factor: float = 1e-3) -> torch.Tensor:
    """H(v) = Chebyshev(v, goal) + tb_factor * Euclidean(v, goal), shape (B, 1, H, W)."""
    B, _, H, W = goal_maps.shape
    dtype, dev = goal_maps.dtype, goal_maps.device
    goal_idx = goal_maps.flatten(1).argmax(dim=1)
    gr = (goal_idx // W).to(dtype).view(B, 1, 1, 1)
    gc = (goal_idx % W).to(dtype).view(B, 1, 1, 1)
    rows = torch.arange(H, device=dev, dtype=dtype).view(1, 1, H, 1)
    cols = torch.arange(W, device=dev, dtype=dtype).view(1, 1, 1, W)
    dy = (rows - gr).abs()
    dx = (cols - gc).abs()
    chebyshev = torch.maximum(dx, dy)
    euclid = torch.sqrt(dx ** 2 + dy ** 2)
    return chebyshev + tb_factor * euclid


# --------------------------------------------------------------------------------------
# Differentiable A*
# --------------------------------------------------------------------------------------
class AStarOutput(NamedTuple):
    histories: torch.Tensor  # (B,1,H,W) closed list C (differentiable w.r.t. the cost map)
    paths: torch.Tensor      # (B,1,H,W) binary path found by backtracking (no gradient)
    found: torch.Tensor      # (B,) bool, True if the goal was reached


class DifferentiableAStar(nn.Module):
    """Batched, differentiable A* (Algorithm 2 of the paper).

    The module has no trainable parameters. Given a per-node step-cost map Phi it runs A* with
        f(v) = G(v) + H(v),   G(v') = G(v*) + Phi(v')
    where the node selection (Eq. 3) is a straight-through argmax: a hard one-hot in the forward
    pass, the gradient of a softmax over exp(-f/tau) in the backward pass.

    detach_g=False (default): G(v*) keeps its graph, so the cost of every ancestor on the path
    receives gradient. detach_g=True: G(v*) is a constant, only Phi(v) itself gets gradient via G(v).
    """

    def __init__(self, tau: Optional[float] = None, tb_factor: float = 1e-3,
                 max_steps: Optional[int] = None, detach_g: bool = False):
        super().__init__()
        self.tau = tau                # default: sqrt(map width), as in the paper
        self.tb_factor = tb_factor
        self.max_steps = max_steps    # default: number of nodes H*W
        self.detach_g = detach_g
        kernel = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel)

    def forward(self, cost_maps: torch.Tensor, obstacles: torch.Tensor,
                start_maps: torch.Tensor, goal_maps: torch.Tensor) -> AStarOutput:
        B, _, H, W = cost_maps.shape
        N = H * W
        tau = self.tau if self.tau is not None else math.sqrt(W)
        max_steps = self.max_steps if self.max_steps is not None else N
        dev, dtype = cost_maps.device, cost_maps.dtype

        h = make_heuristic(goal_maps, self.tb_factor)
        # start and goal must always be passable
        passable = torch.clamp(obstacles + start_maps + goal_maps, max=1.0).detach()
        start_idx = start_maps.flatten(1).argmax(dim=1)
        goal_idx = goal_maps.flatten(1).argmax(dim=1)

        open_ = start_maps.detach().clone()                  # O
        closed = torch.zeros_like(cost_maps)                 # C (carries the gradient)
        g = torch.zeros_like(cost_maps)                      # G
        parents = start_idx.view(B, 1).expand(B, N).clone()  # Parent(v); Parent(start) = start
        found = torch.zeros(B, dtype=torch.bool, device=dev)

        for _ in range(max_steps):
            # ---- Eq. (3): node selection ------------------------------------------------
            f = g + h
            f_masked = f.detach().masked_fill(open_ < 0.5, float("inf"))
            f_min = f_masked.flatten(1).min(dim=1).values.view(B, 1, 1, 1)
            f_min = torch.where(torch.isinf(f_min), torch.zeros_like(f_min), f_min)  # empty open list
            # subtracting a constant does not change the normalised result; it only avoids underflow
            e = torch.exp(-(f - f_min).clamp(min=0.0) / tau) * open_
            prob = e / e.flatten(1).sum(dim=1).clamp(min=1e-20).view(B, 1, 1, 1)

            idx = prob.detach().flatten(1).argmax(dim=1)
            hard = F.one_hot(idx, N).view(B, 1, H, W).to(dtype)
            sel = hard + prob - prob.detach()                # forward: hard, backward: d prob

            # ---- goal verification flag eta (Eq. 8) -------------------------------------
            has_open = open_.flatten(1).sum(dim=1) > 0
            hit = (idx == goal_idx) & has_open
            found = found | hit
            eta = ((~hit) & has_open).to(dtype).view(B, 1, 1, 1)   # 1 = this sample is still searching
            if not bool(eta.any()):
                break

            open_ = open_ - eta * hard
            closed = closed + eta * sel
            closed_d = closed.detach()

            # ---- Eq. (4): node expansion (all masks are detached) -----------------------
            nbr_all = F.conv2d(hard, self.kernel, padding=1)
            v_nbr = nbr_all * passable * (1 - open_) * (1 - closed_d) * eta       # newly discovered
            v_nbr_bar = nbr_all * passable * open_ * (1 - closed_d) * eta         # already open
            open_ = open_ + v_nbr

            # ---- Eqs. (5)-(6): update G --------------------------------------------------
            # g(v*): `hard` is a constant one-hot, so the only question is whether the graph of g is kept.
            g_for_sel = g.detach() if self.detach_g else g
            g_sel = (g_for_sel * hard).flatten(1).sum(dim=1).view(B, 1, 1, 1)
            g_new = g_sel + cost_maps
            g_old = g
            improved = v_nbr_bar * (g_new.detach() < g_old.detach()).to(dtype)
            g = (g_new * v_nbr
                 + torch.minimum(g_new, g_old) * v_nbr_bar
                 + g_old * (1 - v_nbr - v_nbr_bar))

            # ---- parents (for backtracking) ----------------------------------------------
            upd = (v_nbr + improved).flatten(1) > 0.5
            parents = torch.where(upd, idx.view(B, 1).expand(B, N), parents)

        paths = self._backtrack(parents, start_idx, goal_idx, found, (B, H, W), dtype)
        return AStarOutput(histories=closed, paths=paths, found=found)

    @staticmethod
    @torch.no_grad()
    def _backtrack(parents, start_idx, goal_idx, found, shape, dtype):
        B, H, W = shape
        N = H * W
        path = torch.zeros(B, N, dtype=dtype, device=parents.device)
        cur = goal_idx.clone()
        for _ in range(N):
            path.scatter_(1, cur.view(B, 1), 1.0)
            if bool((cur == start_idx).all()):
                break
            cur = parents.gather(1, cur.view(B, 1)).squeeze(1)
        path = path * found.to(dtype).view(B, 1)   # no path for unsolved problems
        return path.view(B, 1, H, W)


# --------------------------------------------------------------------------------------
# Neural A*
# --------------------------------------------------------------------------------------
class CenteredPhi(nn.Module):
    """Phi = sigmoid(z - mean_free(z) + bias), z = raw U-Net logits, shape (B,1,H,W).

    The per-map mean of z over passable cells (channel 0 of the input) is removed, so the global level of
    Phi is fixed by `bias` (= logit of the level you want, e.g. 1.4 -> ~0.8) and the gradient on a global
    shift is projected out. Only relative costs are learned.
    """

    def __init__(self, unet: nn.Module, bias: float):
        super().__init__()
        self.unet = unet
        self.bias = float(bias)

    def forward(self, x):
        free = x[:, :1]
        z = self.unet(x)
        mu = (z * free).sum(dim=(1, 2, 3), keepdim=True) / free.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
        return torch.sigmoid(z - mu + self.bias)


def build_encoder(name: str = "vgg16_bn", in_channels: int = 2, depth: int = 5,
                  phi_bias: Optional[float] = None) -> nn.Module:
    """U-Net with a VGG-16 backbone (segmentation_models_pytorch), output Phi in [0, 1].

    phi_bias=None: plain sigmoid output. phi_bias=float: mean-centred output (see CenteredPhi)."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError as err:
        raise ImportError("Install the encoder dependency with: pip install segmentation-models-pytorch") from err
    decoder_channels = (256, 128, 64, 32, 16)[:depth]
    unet = smp.Unet(encoder_name=name, encoder_depth=depth, decoder_channels=decoder_channels,
                    encoder_weights=None, in_channels=in_channels, classes=1,
                    activation="sigmoid" if phi_bias is None else None)
    return unet if phi_bias is None else CenteredPhi(unet, phi_bias)


class ConstPhi(nn.Module):
    """Encoder stand-in that returns a constant guidance map. Phi = 1 is vanilla A*, Phi = 0 is greedy."""

    def __init__(self, c: float):
        super().__init__()
        self.c = float(c)

    def forward(self, x):
        return torch.full_like(x[:, :1], self.c)


class NeuralAStar(nn.Module):
    """Encoder (guidance map Phi) + differentiable A*."""

    def __init__(self, encoder_name: str = "vgg16_bn", tau: Optional[float] = None,
                 tb_factor: float = 1e-3, max_steps: Optional[int] = None,
                 encoder: Optional[nn.Module] = None, encoder_depth: int = 5,
                 detach_g: bool = False, phi_bias: Optional[float] = None):
        super().__init__()
        self.encoder = encoder if encoder is not None else build_encoder(encoder_name, 2, encoder_depth, phi_bias)
        self.astar = DifferentiableAStar(tau=tau, tb_factor=tb_factor, max_steps=max_steps, detach_g=detach_g)

    def guidance(self, maps, start_maps, goal_maps):
        """Phi in [0, 1], shape (B,1,H,W). Input = map concatenated with (Vs + Vg)."""
        x = torch.cat([maps, start_maps + goal_maps], dim=1)
        return self.encoder(x)

    def forward(self, maps, start_maps, goal_maps):
        phi = self.guidance(maps, start_maps, goal_maps)
        out = self.astar(phi, maps, start_maps, goal_maps)
        return out, phi


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
class PlanningDataset(Dataset):
    """Planning demonstrations.

    maps    (N, H, W)  binary, 1 = passable / 0 = obstacle (set invert_map=True if yours is the opposite)
    starts  (N, 2)     (row, col)
    goals   (N, 2)     (row, col)
    paths   (N, H, W)  binary ground-truth map (1 along the path, including start and goal).
                       May also be the union of all shortest paths if you pass `lengths`.
    lengths (N,)       optional: number of cells on a shortest path (start and goal included). Used by Opt.
                       If omitted, paths.sum() is used, which assumes `paths` is a single shortest path.
    """

    def __init__(self, maps, starts, goals, paths, invert_map: bool = False, lengths=None):
        maps = np.asarray(maps)
        paths = np.asarray(paths)
        if maps.ndim == 4:
            maps = maps[:, 0]
        if paths.ndim == 4:
            paths = paths[:, 0]
        free = maps > 0.5
        if invert_map:
            free = ~free
        self.maps = torch.from_numpy(free.astype(np.uint8))
        self.paths = torch.from_numpy((paths > 0.5).astype(np.uint8))
        self.starts = torch.from_numpy(np.asarray(starts, dtype=np.int64))
        self.goals = torch.from_numpy(np.asarray(goals, dtype=np.int64))
        if lengths is None:
            self.lengths = self.paths.flatten(1).sum(dim=1).float()
        else:
            self.lengths = torch.from_numpy(np.asarray(lengths, dtype=np.float32))
        self._validate()

    def _validate(self):
        N, H, W = self.maps.shape
        assert self.paths.shape == (N, H, W), "paths must have the same shape as maps"
        assert self.starts.shape == (N, 2) and self.goals.shape == (N, 2), "starts/goals must be (N, 2)"
        assert self.lengths.shape == (N,), "lengths must be (N,)"
        ar = torch.arange(N)
        sr, sc = self.starts[:, 0], self.starts[:, 1]
        gr, gc = self.goals[:, 0], self.goals[:, 1]
        assert ((sr >= 0) & (sr < H) & (sc >= 0) & (sc < W)).all(), "start outside the map"
        assert ((gr >= 0) & (gr < H) & (gc >= 0) & (gc < W)).all(), "goal outside the map"
        assert (self.maps[ar, sr, sc] == 1).all(), "some starts are on obstacles (check invert_map)"
        assert (self.maps[ar, gr, gc] == 1).all(), "some goals are on obstacles (check invert_map)"
        assert ((sr != gr) | (sc != gc)).all(), "start == goal in some samples"
        assert (self.paths[ar, sr, sc] == 1).all() and (self.paths[ar, gr, gc] == 1).all(), \
            "ground-truth paths must contain the start and goal"

    @classmethod
    def from_npz(cls, path: str, invert_map: bool = False):
        d = np.load(path)
        lengths = d["lengths"] if "lengths" in d.files else None
        return cls(d["maps"], d["starts"], d["goals"], d["paths"], invert_map=invert_map, lengths=lengths)

    def __len__(self):
        return self.maps.shape[0]

    def __getitem__(self, i):
        H, W = self.maps.shape[1:]
        start = torch.zeros(1, H, W)
        goal = torch.zeros(1, H, W)
        start[0, self.starts[i, 0], self.starts[i, 1]] = 1.0
        goal[0, self.goals[i, 0], self.goals[i, 1]] = 1.0
        return {"map": self.maps[i].unsqueeze(0).float(), "start": start, "goal": goal,
                "path": self.paths[i].unsqueeze(0).float(), "length": self.lengths[i]}


# --------------------------------------------------------------------------------------
# Metrics (Sec. 4.3 of the paper)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: NeuralAStar, loader, device) -> dict:
    """Returns mean loss, Opt (%), Exp (%), Hmean and guidance-map statistics.

    Opt : % of problems where the predicted path is no longer than the ground-truth shortest path.
    Exp : % reduction of node explorations vs. standard A* (same heuristic, unit costs), floored at 0.
    phi_mean : mean of Phi over passable cells (1 = behaves like A*, 0 = greedy).
    phi_sstd : std of Phi over passable cells inside one map, averaged over maps (~0 = Phi ignores the map).
    phi_mstd : std across maps of the per-map mean of Phi (~0 = Phi ignores the input altogether).
    """
    model.eval()
    sums = {"loss": 0.0, "opt": 0.0, "exp": 0.0, "phi_mean": 0.0, "phi_sstd": 0.0}
    pm_all = []
    n = 0
    for batch in loader:
        m, s, g, p = (batch[k].to(device) for k in ("map", "start", "goal", "path"))
        gt_len = batch["length"].to(device)
        out, phi = model(m, s, g)
        base = model.astar(torch.ones_like(m), m, s, g)       # standard A*: every step costs 1

        loss = F.l1_loss(out.histories, p, reduction="none").flatten(1).mean(dim=1)
        e_star = base.histories.flatten(1).sum(dim=1)
        e_model = out.histories.flatten(1).sum(dim=1)
        exp = (100.0 * (e_star - e_model) / e_star.clamp(min=1.0)).clamp(min=0.0)
        pred_len = out.paths.flatten(1).sum(dim=1)
        opt = ((pred_len <= gt_len + 1e-6) & out.found).float() * 100.0

        cnt = m.flatten(1).sum(dim=1).clamp(min=1.0)
        pm = (phi * m).flatten(1).sum(dim=1) / cnt
        var = (((phi - pm.view(-1, 1, 1, 1)) ** 2) * m).flatten(1).sum(dim=1) / cnt

        sums["loss"] += loss.sum().item()
        sums["opt"] += opt.sum().item()
        sums["exp"] += exp.sum().item()
        sums["phi_mean"] += pm.sum().item()
        sums["phi_sstd"] += var.sqrt().sum().item()
        pm_all.append(pm.cpu())
        n += m.shape[0]
    res = {k: v / n for k, v in sums.items()}
    pm_all = torch.cat(pm_all)
    res["phi_mstd"] = pm_all.std().item() if pm_all.numel() > 1 else 0.0
    res["hmean"] = 2 * res["opt"] * res["exp"] / (res["opt"] + res["exp"] + 1e-8)
    return res


@torch.no_grad()
def constant_phi_baselines(loader, device, values=(0.0, 0.25, 0.5, 0.75, 1.0), tau=None,
                           tb_factor: float = 1e-3, max_steps=None):
    """A* with the same constant Phi everywhere. A learned guidance map has to beat these to count as learning."""
    out = []
    for c in values:
        model = NeuralAStar(encoder=ConstPhi(c), tau=tau, tb_factor=tb_factor, max_steps=max_steps).to(device)
        out.append((c, evaluate(model, loader, device)))
    return out


# --------------------------------------------------------------------------------------
# Inference helper
# --------------------------------------------------------------------------------------
@torch.no_grad()
def plan_single(model: NeuralAStar, map_np: np.ndarray, start_rc, goal_rc, device="cpu",
                invert_map: bool = False):
    """Plan on one HxW map. Returns (guidance_map, search_history, path) as HxW numpy arrays."""
    model.eval()
    free = (map_np > 0.5)
    if invert_map:
        free = ~free
    H, W = free.shape
    m = torch.from_numpy(free.astype(np.float32)).view(1, 1, H, W).to(device)
    s = torch.zeros(1, 1, H, W, device=device)
    g = torch.zeros(1, 1, H, W, device=device)
    s[0, 0, start_rc[0], start_rc[1]] = 1.0
    g[0, 0, goal_rc[0], goal_rc[1]] = 1.0
    out, phi = model(m, s, g)
    return phi[0, 0].cpu().numpy(), out.histories[0, 0].cpu().numpy(), out.paths[0, 0].cpu().numpy()
