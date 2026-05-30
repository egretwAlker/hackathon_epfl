"""Research framework for EPLB. The DeepSeek baseline is one preset.

================================ PIPELINE ================================

   rebalance(hotness, D, R)
        |
        v   [FRAMEWORK]  always-on Adam-style per-iter EMA update:
        |       _update_trace_memory(hotness, key, ema_cfg) -> TraceMemory
        |
        |   [FRAMEWORK]  build EstimatorContext(hotness, trace_mem, place_mem, ema_cfg)
        |
        v   Stage A: estimate_w(ctx) -> (w (L, E), stats)
        |
        v   Stage B: build_deployment(w, D, R, stats) -> proposed (L, D, S)
        |
        v   Stage C: select_layers(proposed, shadow, w, stats) -> (layers_priority, change)
        |
        |   [FRAMEWORK]  if change: update PlacementMemory[key].shadow
        v
   (change, layers_priority, proposed, aux)

Two pieces of state are MAINTAINED BY THE FRAMEWORK on every call:

  - TraceMemory[key]:   Adam EMAs (m1, m2, optional M) tracked at PER-ITER
                        granularity. Bias-corrected stats (mean, std, cov,
                        corr) are derived on demand via TraceMemory.stats().
  - PlacementMemory[key]: shadow of the simulator's cur_deploy_table —
                          updated only when change=True; used by the layer
                          selector to compute per-layer PAR gain.

Both are passed to the weight estimator as part of EstimatorContext, so a
new estimator can be a pure function of available signals.

================================ SYMBOLS ================================

   T = collection_window (1024)        D = n_device (EP group size)
   L = n_layers                        R = n_red_expert (= D in simulator)
   E = n_experts (routed)              P = E + R, S = P / D

================================ API ================================

   The only symbol the evaluator calls is rebalance(...). To swap strategy:

       from submission import Strategy, EmaConfig, set_strategy, w_ema_var, ...
       from functools import partial
       set_strategy(Strategy(
           ema              = EmaConfig(beta1=0.999, beta2=0.9999, track_corr=True),
           estimate_w       = partial(w_ema_var, beta=1.0),
           build_deployment = build_dseplb,
           select_layers    = partial(select_top_k_par_gain, k=20, threshold=0.02),
           name             = "smart",
       ))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch


# =============================================================================
# EMA configuration. Held at the Strategy level; consumed by the framework's
# always-on _update_trace_memory and by TraceMemory.stats().
# =============================================================================

@dataclass
class EmaConfig:
    beta1: float = 0.999              # decay for first moment (mean)
    beta2: float = 0.9999             # decay for second moment (uncentered variance)
    track_corr: bool = False          # if True, also EMA the (L, E, E) cross-moment
    beta_corr: float = 0.999          # decay for cross moment
    bias_correction: bool = True      # divide by 1 - beta**n to undo zero-init


# =============================================================================
# Persistent state. Keyed by (L, E, D, R). The simulator's spawn-based Pool
# isolates each case in its own process so cross-case pollution is impossible;
# the key is defensive insurance only.
# =============================================================================

@dataclass
class TraceMemory:
    """Per-iter Adam-style EMA state. Updated by the framework on every call.

    The framework owns the math (see _update_trace_memory). Estimators read
    the *derived* quantities via stats() — they don't need to write to this.
    """
    m1: np.ndarray                          # (L, E)  Adam first moment, per-iter
    m2: np.ndarray                          # (L, E)  Adam second moment (uncentered)
    n_iters: int = 0                        # total iters observed across calls
    M: Optional[np.ndarray] = None          # (L, E, E)  cross moment; None until track_corr
    n_corr_iters: int = 0                   # iters folded into M

    def stats(self, T_last: int, cfg: EmaConfig) -> Dict[str, Any]:
        """Bias-corrected mean / std / var / cov / corr.

        T_last is the window length of the most recent call; used to scale
        per-iter quantities to window-sum scale (for drop-in compat with
        old w_sum semantics).
        """
        n = self.n_iters
        if cfg.bias_correction:
            m1_hat = self.m1 / max(1.0 - cfg.beta1 ** n, 1e-30)
            m2_hat = self.m2 / max(1.0 - cfg.beta2 ** n, 1e-30)
        else:
            m1_hat = self.m1.copy()
            m2_hat = self.m2.copy()

        var_iter = np.maximum(m2_hat - m1_hat ** 2, 0.0)
        std_iter = np.sqrt(var_iter)

        s: Dict[str, Any] = {
            "reduction":   "ema",
            "mean_iter":   m1_hat,
            "std_iter":    std_iter,
            "var_iter":    var_iter,
            "mean_window": T_last * m1_hat,
            "std_window":  np.sqrt(T_last) * std_iter,
            "std":         np.sqrt(T_last) * std_iter,  # alias
            "n_iters":     n,
        }

        if self.M is not None:
            n_c = self.n_corr_iters
            if cfg.bias_correction:
                M_hat = self.M / max(1.0 - cfg.beta_corr ** n_c, 1e-30)
            else:
                M_hat = self.M.copy()
            cov = M_hat - m1_hat[:, :, None] * m1_hat[:, None, :]
            diag = np.diagonal(cov, axis1=1, axis2=2)
            denom = np.sqrt(np.maximum(diag[:, :, None] * diag[:, None, :], 1e-30))
            corr = np.clip(cov / denom, -1.0, 1.0)
            s["cov"] = cov
            s["corr"] = corr
            s["n_corr_iters"] = n_c
        return s


@dataclass
class PlacementMemory:
    """Shadow of the simulator's cur_deploy_table, updated when change=True."""
    shadow: np.ndarray                      # (L, D, S) int64


_TRACE: Dict[Tuple[int, int, int, int], TraceMemory] = {}
_PLACE: Dict[Tuple[int, int, int, int], PlacementMemory] = {}


def reset_state() -> None:
    _TRACE.clear()
    _PLACE.clear()


def get_state() -> Tuple[Dict, Dict]:
    return _TRACE, _PLACE


# =============================================================================
# Initial shadow layout — matches the simulator's init_deploy_table(default=False).
# Source: quickstart.py:25-40 and dynamic_lb_simulator.py:114-118.
# =============================================================================

def _init_shadow(L: int, D: int, S: int, E: int) -> np.ndarray:
    shadow = np.zeros((L, D, S), dtype=np.int64)
    for d in range(D):
        for s in range(S - 1):
            shadow[:, d, s] = (d * (S - 1) + s) % E
        shadow[:, d, -1] = shadow[:, d, -2]
    return shadow


# =============================================================================
# Framework: always-on per-iter Adam EMA update.
# Vectorized over the T iters in a single call (one closed-form expression).
# =============================================================================

def _update_trace_memory(
    hotness: np.ndarray,
    key: Tuple[int, int, int, int],
    cfg: EmaConfig,
) -> TraceMemory:
    """Apply T iter-level Adam updates to TraceMemory[key] in a single pass.

    Equivalent to the per-iter loop:
        for t in 0..T-1:
            m1 <- beta1 * m1 + (1 - beta1) * x_t
            m2 <- beta2 * m2 + (1 - beta2) * x_t**2
            M  <- beta_c * M + (1 - beta_c) * (x_t outer x_t)        [if track_corr]
    """
    T, L, E = hotness.shape
    x = np.asarray(hotness, dtype=np.float64)

    tm = _TRACE.get(key)
    if tm is None:
        tm = TraceMemory(
            m1=np.zeros((L, E), dtype=np.float64),
            m2=np.zeros((L, E), dtype=np.float64),
        )

    ar = np.arange(T - 1, -1, -1, dtype=np.float64)         # T-1, T-2, ..., 0

    w1 = (1.0 - cfg.beta1) * (cfg.beta1 ** ar)
    tm.m1 = (cfg.beta1 ** T) * tm.m1 + np.einsum("t,tle->le", w1, x)

    w2 = (1.0 - cfg.beta2) * (cfg.beta2 ** ar)
    tm.m2 = (cfg.beta2 ** T) * tm.m2 + np.einsum("t,tle->le", w2, x * x)

    tm.n_iters += T

    if cfg.track_corr:
        wc = (1.0 - cfg.beta_corr) * (cfg.beta_corr ** ar)
        X_layer = x.transpose(1, 0, 2)                       # (L, T, E)
        Xw = X_layer * wc[None, :, None]                     # (L, T, E)
        cross = Xw.transpose(0, 2, 1) @ X_layer              # (L, E, E)
        if tm.M is None:
            tm.M = np.zeros((L, E, E), dtype=np.float64)
            tm.n_corr_iters = 0
        tm.M = (cfg.beta_corr ** T) * tm.M + cross
        tm.n_corr_iters += T

    _TRACE[key] = tm
    return tm


# =============================================================================
# Context object passed to every weight estimator.
# =============================================================================

@dataclass
class EstimatorContext:
    """Inputs available to every Stage A function.

    A new estimator is a pure function of this context: it can read any
    combination of raw hotness, EMA-derived stats, last-committed shadow,
    and the EMA hyperparameters.
    """
    hotness:   np.ndarray                   # (T, L, E)
    trace_mem: TraceMemory                  # EMA state (just updated by framework)
    place_mem: Optional[PlacementMemory]    # last committed deployment, or None on call 1
    ema_cfg:   EmaConfig                    # EMA hyperparameters in use


# =============================================================================
# Stage A: weight estimators.   ctx -> (w (L, E), stats dict)
# =============================================================================

def w_sum(
    ctx: EstimatorContext,
    *,
    lam: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Window-sum estimator with optional variance term: w = window_sum + lam * std.

        lam =  0  -> w = sum_t hotness[t]                  (pure DS-EPLB; bit-identical)
        lam >  0  -> defensive: inflate volatile experts
        lam <  0  -> skeptical: deflate volatile experts (clamped at 0)

    The std signal comes from the framework's always-on Adam EMA state, so
    it costs nothing extra to use. With lam=0 the EMA state is not read.
    """
    w_window = np.asarray(ctx.hotness, dtype=np.float64).sum(axis=0)
    if lam == 0.0:
        return w_window, {"reduction": "sum"}

    stats = ctx.trace_mem.stats(ctx.hotness.shape[0], ctx.ema_cfg)
    w = w_window + lam * stats["std_window"]
    out_stats = {**stats, "reduction": "sum+lam_std"}
    return np.maximum(w, 0.0), out_stats


def w_ema(
    ctx: EstimatorContext,
    *,
    lam: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """EMA-based estimator: w = mean + lam * std.  Both at window-sum scale.

    lam can be any real number:
       lam =  0   -> pure EMA mean (DS-EPLB-like input, just smoother).
       lam >  0   -> INFLATE volatile experts: give them more replicas as a
                     hedge against load spikes. "Defensive" allocation.
       lam <  0   -> DEFLATE volatile experts: give them fewer replicas
                     because their reported load may be noise. "Skeptical"
                     allocation.

    Implementation note: std here is at window-sum scale (sqrt(T) * std_iter),
    so `mean + lam * std` is comparable to a window-sum w. The packer (LPT)
    sees w as the relative load score per (l, e).
    """
    stats = ctx.trace_mem.stats(ctx.hotness.shape[0], ctx.ema_cfg)
    w = stats["mean_window"] + lam * stats["std_window"]
    # Guard: w must be non-negative for the greedy replicator's argmax(w/cnt)
    # to make sense. Clamp at 0 if a large negative lam pushed w below zero
    # for some (l, e).
    return np.maximum(w, 0.0), stats


# =============================================================================
# Stage B: deployment builders.   (w, D, R, stats) -> deployment (L, D, S)
# =============================================================================

def build_dseplb(
    w: np.ndarray,
    n_device: int,
    n_red_expert: int,
    stats: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Global DS-EPLB: greedy replicate by amortized load, LPT pack to devices.
    Equivalent to deepseek-ai/EPLB with num_groups=1, num_nodes=1."""
    L, E = w.shape
    P = E + n_red_expert
    weight_t = torch.from_numpy(np.asarray(w, dtype=np.float64)).float()
    phy2log, _, _ = _rebalance_experts_hierarchical(
        weight_t,
        num_physical_experts=P,
        num_groups=1,
        num_nodes=1,
        num_gpus=n_device,
    )
    return phy2log.numpy().reshape(L, n_device, P // n_device)


# =============================================================================
# Stage C: layer selectors.   (proposed, current, w, stats) -> (priority, change)
# =============================================================================

def select_all(
    proposed: np.ndarray,
    current: Optional[np.ndarray],
    w: np.ndarray,
    stats: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, bool]:
    """DeepSeek default: migrate every layer in natural order, always."""
    return np.arange(proposed.shape[0], dtype=np.int64), True


def select_top_k_par_gain(
    proposed: np.ndarray,
    current: Optional[np.ndarray],
    w: np.ndarray,
    stats: Optional[Dict[str, Any]] = None,
    *,
    k: Optional[int] = None,
    threshold: float = 0.0,
) -> Tuple[np.ndarray, bool]:
    """Migrate only layers whose predicted PAR drop exceeds `threshold`,
    sorted by gain descending, capped at `k`.

    Falls back to range(L) on the first call when no shadow exists yet."""
    if current is None:
        return np.arange(proposed.shape[0], dtype=np.int64), True

    L = proposed.shape[0]
    gains = np.empty(L, dtype=np.float64)
    for l in range(L):
        gains[l] = _par_for_layer(w[l], current[l]) - _par_for_layer(w[l], proposed[l])

    keep = np.where(gains > threshold)[0]
    if keep.size == 0:
        return np.empty(0, dtype=np.int64), False
    keep = keep[np.argsort(-gains[keep])]
    if k is not None:
        keep = keep[: int(k)]
    return keep.astype(np.int64), True


# =============================================================================
# Strategy bundle + active selection
# =============================================================================

@dataclass
class Strategy:
    """Bundle of (ema config, estimator, builder, selector).
    Defaults reproduce DS-EPLB bit-for-bit (w_sum ignores EMA state)."""
    ema:              EmaConfig = field(default_factory=EmaConfig)
    estimate_w:       Callable = w_sum
    build_deployment: Callable = build_dseplb
    select_layers:    Callable = select_all
    name:             str      = "deepseek-default"


DEEPSEEK_STRATEGY: Strategy = Strategy()


# === BUILD_INJECT_BELOW ===
# Default active strategy. build.py replaces the block between these markers
# to bake a chosen strategy (from strategies/*.json) into the upload zip.
# Edit manually only if you also update build.py's regex.
_active_strategy: Strategy = DEEPSEEK_STRATEGY
# === BUILD_INJECT_ABOVE ===


def set_strategy(s: Strategy) -> None:
    """Switch the active strategy. Affects all subsequent rebalance() calls."""
    global _active_strategy
    _active_strategy = s


def get_strategy() -> Strategy:
    return _active_strategy


# =============================================================================
# Entry point — the only symbol the evaluator calls.
# =============================================================================

def rebalance(hotness, n_device, n_red_expert):
    """Required competition API. See module docstring for return semantics."""
    s = _active_strategy
    L, E = hotness.shape[1], hotness.shape[2]
    D, R = int(n_device), int(n_red_expert)
    S = (E + R) // D
    key = (L, E, D, R)

    # Framework: always-on Adam EMA update.
    tm = _update_trace_memory(hotness, key, s.ema)
    pm = _PLACE.get(key)
    ctx = EstimatorContext(hotness=hotness, trace_mem=tm, place_mem=pm, ema_cfg=s.ema)

    # Stage A: derive w (and stats) from raw window + EMA state + last layout.
    w, stats = s.estimate_w(ctx)

    # Stage B: produce a proposed deployment table.
    proposed = s.build_deployment(w, D, R, stats=stats)

    # Stage C: pick which layers to actually migrate.
    shadow = pm.shadow if pm is not None else None
    layers_priority, change = s.select_layers(proposed, shadow, w, stats)

    # Framework: persist shadow update so the next call's selector has truth.
    if change and len(layers_priority) > 0:
        if pm is None:
            pm = PlacementMemory(shadow=_init_shadow(L, D, S, E))
            _PLACE[key] = pm
        pm.shadow[layers_priority] = proposed[layers_priority]

    return change, layers_priority, proposed, {"strategy": s.name, "stats": stats}


# =============================================================================
# ============================== INTERNALS ====================================
# DeepSeek EPLB inlined from
# https://github.com/deepseek-ai/EPLB/blob/main/eplb.py
# Treat these as private to the framework.
# =============================================================================

def _balanced_packing(
    weight: torch.Tensor, num_packs: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LPT bin-packing into num_packs equal-cardinality bins."""
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(
            weight.size(-1), dtype=torch.int64, device=weight.device
        ).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min(
                (j for j in range(num_packs) if pack_items[j] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def _replicate_experts(weight: torch.Tensor, num_phy: int):
    """Greedy replication: argmax amortized load, num_phy - E rounds."""
    n, num_log = weight.shape
    assert num_phy >= num_log
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def _rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """Three-stage EPLB. With num_groups=1, num_nodes=1 this is global EPLB."""
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(perm.shape),
        )
        return inv

    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = _balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (
        ((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1)
        + torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)
    ).flatten(-2)
    mlog2log = inverse(log2mlog)

    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes
    )
    phy2mlog, phyrank, mlogcnt = _replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = _balanced_packing(tokens_per_phy, num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    pphy2mlog = (
        pphy2mlog.view(num_layers, num_nodes, -1)
        + torch.arange(
            0,
            num_logical_experts,
            num_logical_experts // num_nodes,
            device=group_pack_index.device,
        ).view(1, -1, 1)
    ).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def _par_for_layer(hotness_layer: np.ndarray, deployment_layer: np.ndarray) -> float:
    """PAR of one layer given (E,) loads and (D, S) placement."""
    n_experts = hotness_layer.shape[0]
    cut = np.bincount(deployment_layer.reshape(-1), minlength=n_experts)
    if np.any(cut == 0):
        return float("inf")
    weights = hotness_layer / cut
    loads = weights[deployment_layer.reshape(-1)].reshape(deployment_layer.shape).sum(-1)
    return float(loads.max() / loads.mean())
