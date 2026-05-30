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

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch


# Hardware / scoring constants (from the leaderboard score formula). Mirrored
# here so submission.py can reason about score directly inside the builder.
_BALANCED_COMPUTE_SECONDS = 60.0
_EXPERT_BYTES = 88_080_384
_BANDWIDTH_BPS = 900_000_000_000
_TRANSMIT_COST_PER_SLOT = _EXPERT_BYTES / _BANDWIDTH_BPS   # ~9.787e-5 s per slot


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


def _init_shadow_unique(L: int, D: int, S: int, E: int) -> np.ndarray:
    """Unique-per-device cyclic round-robin: slot (d, s) -> expert (d*S + s) % E.

    Each device holds S consecutive expert ids, so no expert appears twice on
    the same device (as long as S <= E, which is always true in this setting).
    Replication is spread across devices instead: with P = D*S = E + D total
    slots, each expert appears in floor(P/E) or ceil(P/E) distinct devices.

    Used by build_incremental when `unique_per_device=True` to bootstrap the
    invariant — the simulator's actual initial table has duplicates (slot S-1
    duplicates slot S-2), so call 1 must migrate away from that.
    """
    assert S <= E, f"unique_per_device requires S <= E, got S={S} E={E}"
    shadow = np.empty((L, D, S), dtype=np.int64)
    for d in range(D):
        for s in range(S):
            shadow[:, d, s] = (d * S + s) % E
    return shadow


def _init_shadow_pin_hottest(
    L: int, D: int, S: int, E: int, w: np.ndarray
) -> np.ndarray:
    """Initial layout that pins each layer's hottest expert on every device.

    Per layer l:
      - slot 0 of every device  -> e_hot[l] = argmax_e w[l, e]
      - slots 1..S-1 on devices -> cyclic round-robin over the other (E-1)
        experts, in any order.

    Properties:
      logcnt[e_hot] = D                        (maximum spreading of hot load)
      logcnt[other] in {1, 2}                  (1 expert gets the spare slot)
      every expert appears >= 1 time           (since D*(S-1) >= E-1)
      every device has S DISTINCT experts      (slot 0 = e_hot, plus S-1
                                                consecutive cycle entries
                                                from the others)

    Used only by build_incremental's `pin_hottest=True` bootstrap on call 1.
    """
    assert D * (S - 1) >= E - 1, (
        f"pin_hottest requires D*(S-1) >= E-1, got D={D} S={S} E={E}"
    )
    shadow = np.empty((L, D, S), dtype=np.int64)
    all_experts = np.arange(E, dtype=np.int64)
    for l in range(L):
        e_hot = int(np.argmax(w[l]))
        shadow[l, :, 0] = e_hot
        other = all_experts[all_experts != e_hot]    # (E-1,)
        # Round-robin (E-1) experts over D*(S-1) slots:
        # slot_idx -> other[slot_idx mod (E-1)]
        # device d, slot s (s in 1..S-1)  <- slot_idx = (s-1)*D + d
        # using device-major order so consecutive `other` entries land on
        # different devices, helping the cyclic spread of duplicates.
        for s in range(1, S):
            for d in range(D):
                slot_idx = (s - 1) * D + d
                shadow[l, d, s] = other[slot_idx % (E - 1)]
    return shadow


def _is_unique_per_device(layer: np.ndarray) -> bool:
    """True iff no expert appears twice on the same device in any layer.
    Accepts (L, D, S) or a single (D, S) layer. O(L*D*S)."""
    if layer.ndim == 3:
        for l_idx in range(layer.shape[0]):
            if not _is_unique_per_device(layer[l_idx]):
                return False
        return True
    D, S = layer.shape
    for d in range(D):
        if len(set(int(x) for x in layer[d])) < S:
            return False
    return True


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
    std_source: str = "ema",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Variance-aware weight estimator: w = mean + lam * std (window-sum scale).

    lam can be any real number:
       lam =  0  -> pure mean (DS-EPLB-like input, possibly smoother).
       lam >  0  -> INFLATE volatile experts (defensive).
       lam <  0  -> DEFLATE volatile experts (skeptical; clamped at w >= 0).

    std_source picks where the mean/std come from:
       "ema"    (default): bias-corrected Adam EMA across all calls. Tracks
                           slow cross-call drift; needs ~few calls to warm up
                           but smooths transient noise.
       "window": within-call statistics computed directly from the T-iter
                 hotness window. No cross-call memory, no warm-up, maximally
                 reactive to the current window only.

    Note: std at window-sum scale = sqrt(T) * std_iter, assuming independent
    iters. The packer (LPT) sees `w` as the relative load score per (l, e).
    """
    T = ctx.hotness.shape[0]
    if std_source == "ema":
        stats = ctx.trace_mem.stats(T, ctx.ema_cfg)
        mean_window = stats["mean_window"]
        std_window = stats["std_window"]
    elif std_source == "window":
        x = np.asarray(ctx.hotness, dtype=np.float64)
        mean_iter = x.mean(axis=0)
        var_iter = x.var(axis=0)
        std_iter = np.sqrt(var_iter)
        mean_window = T * mean_iter           # exact window sum
        std_window = np.sqrt(T) * std_iter    # window-sum scale, indep-iter assumption
        stats = {
            "reduction":   "window",
            "mean_iter":   mean_iter,
            "std_iter":    std_iter,
            "var_iter":    var_iter,
            "mean_window": mean_window,
            "std_window":  std_window,
            "std":         std_window,
            "n_iters":     T,
        }
    else:
        raise ValueError(
            f"std_source must be 'ema' or 'window', got {std_source!r}"
        )

    w = mean_window + lam * std_window
    return np.maximum(w, 0.0), stats


# =============================================================================
# Stage B: deployment builders.   (w, D, R, stats) -> deployment (L, D, S)
# =============================================================================

def build_dseplb(
    w: np.ndarray,
    n_device: int,
    n_red_expert: int,
    stats: Optional[Dict[str, Any]] = None,
    *,
    current: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Global DS-EPLB: greedy replicate by amortized load, LPT pack to devices.
    Equivalent to deepseek-ai/EPLB with num_groups=1, num_nodes=1.

    `current` is ignored (this builder constructs from scratch). It's accepted
    so all builders share a uniform signature."""
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


def build_incremental(
    w: np.ndarray,
    n_device: int,
    n_red_expert: int,
    stats: Optional[Dict[str, Any]] = None,
    *,
    current: Optional[np.ndarray] = None,
    max_moves_per_layer: int = 30,
    min_par_gain_B: float = 1e-3,
    min_par_gain_A: float = 2e-3,
    score_aware: bool = False,
    assumed_cycle_iters: int = 1024,
    assumed_total_iters: int = 10000,
    use_dseplb_init: bool = False,
    max_cold_tries: Optional[int] = None,
    unique_per_device: bool = False,
    cov_quantile: float = 1.0,
    cov_source: str = "ema",
    pin_hottest: bool = False,
    fuse_phases: bool = False,
    strict_b_prune: bool = False,
) -> np.ndarray:
    """Local-search builder. Hill-climbs from `current` toward lower PAR.

    Per-layer inner loop, up to max_moves_per_layer iterations:

        1. d_hot = argmax_d load[d];  d_cold = argmin_d load[d].
           If d_hot == d_cold, layer is balanced -> stop.

        2. PHASE B (1-transmit reassign): enumerate cross-product
              { (s_hot in slots(d_hot),  s_cold in slots(d_cold)) }
           keeping only candidates with N[expert@s_cold] > 1 (so demotion
           preserves the "every expert appears" invariant) and where the
           experts at s_hot and s_cold differ. Sort by descending
           (a[expert@s_hot] - a[expert@s_cold]) and accept the first one
           whose PAR drop exceeds min_par_gain_B.

        3. PHASE A (2-transmit swap, fallback): swap (d_hot, slot with the
           hottest expert) <-> (d_cold, slot with the coldest expert). Accept
           if PAR drop exceeds min_par_gain_A.

        4. Else: layer is at a local optimum -> stop.

    score_aware: when True, replace the raw ΔPAR thresholds (min_par_gain_*)
    with break-even thresholds derived directly from the leaderboard score
    formula. A move is accepted iff its modeled-score-time savings exceed
    its modeled transmit cost:

        compute_value_per_par = 60.0 * cycle / (total_iters * L)
        break_even_B = transmit_cost          / compute_value_per_par
        break_even_A = 2 * transmit_cost      / compute_value_per_par

    where `cycle` ~= collection_interval and `total_iters` is the approximate
    trace length per case. Both are configurable via `assumed_cycle_iters`
    (default 1024) and `assumed_total_iters` (default 10000); since neither
    is known exactly inside rebalance(), they're heuristic constants the
    user can sweep over. The min_par_gain_* params remain as a FLOOR — the
    effective threshold is `max(min_par_gain_X, break_even_X)`.

    fuse_phases: when True, replace the strict-priority "try B then fall back
    to A" inner loop with a single fused enumeration. Each step builds the
    full candidate set from both phases (B reassigns + A swaps) for the
    current d_hot, scores each by net profit
        profit(move) = (cur_par - new_par) - n_transmits * per_transmit_cost
    and applies the global argmax (if profit > 0). per_transmit_cost is:
        score_aware=True:  _TRANSMIT_COST_PER_SLOT / compute_value
        score_aware=False: max(min_par_gain_B, min_par_gain_A / 2)
    This costs more work per step (no first-acceptable short-circuit on phase
    B, and phase A is evaluated alongside even when a phase-B move would also
    pass) but lets the step pick a phase-A swap over a worse phase-B reassign
    when the swap's larger par-drop outweighs its extra transmit. Same filters
    (unique_per_device, cov_quantile) apply.

    strict_b_prune: the d_cold early-break uses
        max_possible_gain = (load[d_hot] - load[d_cold]) / (2 * mean_load)
    as an upper bound on PAR drop. That bound is tight for phase A (pure
    swap, only d_hot/d_cold loads change), but unsound for phase B — a B
    move dilutes amrt[e_rep] and concentrates amrt[e_rem] globally, so other
    devices' loads (including d_hot's) can move by more than gap/2. The
    default (False) keeps the heuristic prune — fast, but can skip valid
    B moves when cnt[e_rep] is small and d_hot holds multiple copies of
    e_rep. With strict_b_prune=True the phase-B prune (and the fused-mode
    prune, which charges only one transmit) is disabled, so every d_cold
    in sorted_cold is enumerated; phase A's prune (non-fused) is left in
    place because it is correct.

    cov_source: when cov_quantile < 1.0, picks where the covariance matrix
    used by the filter comes from:
       "ema"    (default): bias-corrected EMA covariance carried across all
                           calls (requires EmaConfig.track_corr=True). Smooth
                           across windows; needs warm-up.
       "window": sample covariance computed directly from the T-iter hotness
                 window of this call. No cross-call memory, no warm-up; reflects
                 only the most recent window. Requires the framework to stash
                 `stats["hotness"]` (rebalance() does this unconditionally).
                 Does not need track_corr=True.

    The framework eagerly initializes `current` (= PlacementMemory.shadow)
    on call 1 according to Strategy.initial_layout. If `current` arrives None
    anyway (defensive: some non-standard call path), we fall back to the
    round-robin layout that matches the simulator's own initial table — no
    DS-EPLB construction is done here.
    """
    if current is None:
        L_param = w.shape[0]
        E_param = w.shape[1]
        D_param = int(n_device)
        S_param = (E_param + int(n_red_expert)) // D_param
        current = _init_shadow(L_param, D_param, S_param, E_param)

    L, D, S = current.shape

    # Optional one-shot DS-EPLB bootstrap. Triggers only when the seed the
    # framework gave us is the simulator's round-robin initial table — i.e.
    # this is the bootstrap call AND Strategy.initial_layout was left at the
    # round_robin default. Equivalent to setting Strategy.initial_layout to
    # "dseplb" but kept at the builder layer (so it's controllable per
    # build_incremental config without changing the strategy).
    # Bootstrap precedence on call 1 (when `current` matches the simulator's
    # round-robin initial table):
    #   pin_hottest > use_dseplb_init > leave alone
    # `pin_hottest` takes precedence because it's a more specific request
    # (pin the hottest on every device, then fill the rest cyclically).
    if pin_hottest:
        round_robin = _init_shadow(L, D, S, w.shape[1])
        if np.array_equal(current, round_robin):
            current = _init_shadow_pin_hottest(L, D, S, w.shape[1], w)
    elif use_dseplb_init:
        round_robin = _init_shadow(L, D, S, w.shape[1])
        if np.array_equal(current, round_robin):
            current = build_dseplb(w, int(n_device), int(n_red_expert))

    # `unique_per_device` (when True): no bootstrap — keep whatever `current`
    # we got (round-robin / DS-EPLB / etc.). Only the move filters below skip
    # candidates that would *create* a new same-expert-twice-on-one-device.
    # Existing duplicates in `current` persist; we just stop adding more.

    # Effective acceptance thresholds for the inner loop.
    if score_aware:
        compute_value = (
            _BALANCED_COMPUTE_SECONDS
            * float(assumed_cycle_iters)
            / max(1.0, float(assumed_total_iters) * L)
        )
        break_even_B = _TRANSMIT_COST_PER_SLOT / max(compute_value, 1e-30)
        break_even_A = 2.0 * _TRANSMIT_COST_PER_SLOT / max(compute_value, 1e-30)
        thresh_B = max(float(min_par_gain_B), break_even_B)
        thresh_A = max(float(min_par_gain_A), break_even_A)
        # Fused-mode per-transmit cost: charges thresh_B for one transmit,
        # thresh_A == 2 * thresh_B for the swap (matches the legacy thresholds
        # whenever min_par_gain_A <= 2 * break_even_B, which holds for the
        # typical case min_par_gain_A == 0).
        per_tx_cost = max(float(min_par_gain_B), break_even_B)
    else:
        thresh_B = float(min_par_gain_B)
        thresh_A = float(min_par_gain_A)
        # Pick the tighter per-transmit cost so fused phase-A acceptance is
        # at least as strict as the legacy thresh_A.
        per_tx_cost = max(float(min_par_gain_B), float(min_par_gain_A) / 2.0)
    E = w.shape[1]
    out = np.array(current, dtype=np.int64, copy=True)

    # Per-layer covariance-threshold filter setup. When cov_quantile < 1.0
    # we need the layer's covariance matrix from stats (requires the active
    # EmaConfig to have track_corr=True; the framework computes both cov and
    # corr). For each layer we precompute the |cov| threshold at the requested
    # quantile of off-diagonal values, and the inner-loop move filters reject
    # any move that would land an expert adjacent to another DIFFERENT expert
    # whose |cov| exceeds the threshold.
    cov_arr = None
    cov_thresh_per_layer: Optional[np.ndarray] = None
    if cov_quantile < 1.0:
        if cov_source == "ema":
            if stats is not None and "cov" in stats:
                cov_arr = stats["cov"]                      # (L, E, E)
        elif cov_source == "window":
            hotness = stats.get("hotness") if stats is not None else None
            if hotness is None:
                raise ValueError(
                    "cov_source='window' requires stats['hotness']; "
                    "rebalance() should stash it."
                )
            x = np.asarray(hotness, dtype=np.float64)        # (T, L, E)
            T_w = x.shape[0]
            # Sample covariance per layer, per-iter scale to match the EMA
            # path's stats['cov']. Denominator T-1 (unbiased); thresholding
            # is via quantile so the absolute scale doesn't affect filter
            # decisions, but matching scale keeps the two sources comparable.
            mean_iter = x.mean(axis=0)                       # (L, E)
            Xc = (x - mean_iter[None, :, :]).transpose(1, 0, 2)  # (L, T, E)
            denom = max(T_w - 1, 1)
            cov_arr = (Xc.transpose(0, 2, 1) @ Xc) / denom   # (L, E, E)
        else:
            raise ValueError(
                f"cov_source must be 'ema' or 'window', got {cov_source!r}"
            )

        if cov_arr is not None and cov_arr.shape[0] == L:
            cov_thresh_per_layer = np.empty(L, dtype=np.float64)
            off_mask = ~np.eye(E, dtype=bool)
            for l_idx in range(L):
                abs_off = np.abs(cov_arr[l_idx][off_mask])
                cov_thresh_per_layer[l_idx] = float(
                    np.quantile(abs_off, cov_quantile)
                )

    for l in range(L):
        layer = out[l]  # view; in-place edits below mutate `out`
        cnt = np.bincount(layer.reshape(-1), minlength=E).astype(np.int64)
        if np.any(cnt == 0):
            continue

        amrt = w[l] / cnt.astype(np.float64)              # (E,)
        load = amrt[layer].sum(axis=1)                    # (D,)
        # mean(load) = sum(w[l]) / D is INVARIANT under any move (proved earlier),
        # so compute once and reuse for PAR -> max(load) / mean_load.
        mean_load = load.mean()
        w_l = w[l]                                         # local alias

        # Per-layer covariance matrix + threshold (if filter enabled).
        cov_l = None
        cov_thresh_l = None
        if cov_thresh_per_layer is not None and cov_arr is not None:
            cov_l = cov_arr[l]
            cov_thresh_l = float(cov_thresh_per_layer[l])

        for _step in range(int(max_moves_per_layer)):
            cur_max = load.max()
            cur_par = cur_max / mean_load
            d_hot = int(load.argmax())

            # Cold devices ordered coldest-first (excluding d_hot itself).
            sorted_cold = np.argsort(load)
            sorted_cold = sorted_cold[sorted_cold != d_hot]
            if max_cold_tries is not None:
                sorted_cold = sorted_cold[: int(max_cold_tries)]
            if sorted_cold.size == 0:
                break

            slots_hot = layer[d_hot]
            a_hot_slots = amrt[slots_hot]

            if fuse_phases:
                # Fused enumeration: gather every candidate (phase B reassign
                # + phase A swap) for the current d_hot, score by net profit,
                # apply the global argmax. No first-acceptable short-circuit
                # within a step; the outer d_cold loop still early-breaks
                # using max_possible_gain since colder devices can't beat
                # the same bound.
                best_profit = 0.0
                best = None  # ("B", d_cold, sh, sc, e_rep, e_rem,
                             #  new_a_rep, new_a_rem, new_load)
                             # or ("A", d_cold, sh, sc, e_a, e_b, new_load)
                for d_cold_i in sorted_cold:
                    d_cold = int(d_cold_i)
                    if not strict_b_prune:
                        max_possible_gain = (load[d_hot] - load[d_cold]) / (2.0 * mean_load)
                        # Cheapest move (phase B) requires par_drop > per_tx_cost
                        # to beat current best; if not even possible, stop. Bound
                        # is only tight for phase A; strict_b_prune disables it.
                        if max_possible_gain - per_tx_cost <= best_profit:
                            break

                    slots_cold = layer[d_cold]
                    a_cold_slots = amrt[slots_cold]
                    cnt_cold_slots = cnt[slots_cold]

                    # --- Phase B candidates (1 transmit each) ---
                    for sh in range(S):
                        e_rep_cand = int(slots_hot[sh])
                        for sc in range(S):
                            if cnt_cold_slots[sc] <= 1:
                                continue
                            e_rem_cand = int(slots_cold[sc])
                            if e_rep_cand == e_rem_cand:
                                continue
                            if unique_per_device:
                                dup = False
                                for t in range(S):
                                    if t != sc and int(slots_cold[t]) == e_rep_cand:
                                        dup = True
                                        break
                                if dup:
                                    continue
                            if cov_thresh_l is not None and cov_l is not None:
                                too_covariate = False
                                for t in range(S):
                                    if t == sc:
                                        continue
                                    e_other = int(slots_cold[t])
                                    if e_other == e_rep_cand:
                                        continue
                                    if abs(float(cov_l[e_rep_cand, e_other])) > cov_thresh_l:
                                        too_covariate = True
                                        break
                                if too_covariate:
                                    continue

                            new_a_rep = w_l[e_rep_cand] / (cnt[e_rep_cand] + 1)
                            new_a_rem = w_l[e_rem_cand] / (cnt[e_rem_cand] - 1)
                            delta_rep = new_a_rep - amrt[e_rep_cand]
                            delta_rem = new_a_rem - amrt[e_rem_cand]
                            rep_count = (layer == e_rep_cand).sum(axis=1)
                            rem_count = (layer == e_rem_cand).sum(axis=1)
                            new_load_B = load + rep_count * delta_rep + rem_count * delta_rem
                            new_load_B[d_cold] += new_a_rep - new_a_rem
                            par_drop = cur_par - new_load_B.max() / mean_load
                            profit = par_drop - per_tx_cost
                            if profit > best_profit:
                                best_profit = profit
                                best = ("B", d_cold, sh, sc,
                                        e_rep_cand, e_rem_cand,
                                        new_a_rep, new_a_rem, new_load_B)

                    # --- Phase A candidate (1 swap = 2 transmits) ---
                    sh_A = int(a_hot_slots.argmax())
                    sc_A = int(a_cold_slots.argmin())
                    e_a = int(slots_hot[sh_A])
                    e_b = int(slots_cold[sc_A])
                    if e_a != e_b:
                        ok = True
                        if unique_per_device:
                            for t in range(S):
                                if t != sh_A and int(slots_hot[t]) == e_b:
                                    ok = False; break
                                if t != sc_A and int(slots_cold[t]) == e_a:
                                    ok = False; break
                        if ok and cov_thresh_l is not None and cov_l is not None:
                            for t in range(S):
                                if t != sh_A:
                                    e_other = int(slots_hot[t])
                                    if e_other != e_b and abs(float(cov_l[e_b, e_other])) > cov_thresh_l:
                                        ok = False; break
                                if not ok:
                                    break
                                if t != sc_A:
                                    e_other = int(slots_cold[t])
                                    if e_other != e_a and abs(float(cov_l[e_a, e_other])) > cov_thresh_l:
                                        ok = False; break
                        if ok:
                            new_load_A = load.copy()
                            new_load_A[d_hot] = load[d_hot] - amrt[e_a] + amrt[e_b]
                            new_load_A[d_cold] = load[d_cold] - amrt[e_b] + amrt[e_a]
                            par_drop = cur_par - new_load_A.max() / mean_load
                            profit = par_drop - 2.0 * per_tx_cost
                            if profit > best_profit:
                                best_profit = profit
                                best = ("A", d_cold, sh_A, sc_A,
                                        e_a, e_b, new_load_A)

                if best is None:
                    break  # no profitable phase-A or phase-B move
                if best[0] == "B":
                    _, d_cold, sh, sc, e_rep, e_rem, new_a_rep, new_a_rem, new_load_B = best
                    layer[d_cold, sc] = e_rep
                    cnt[e_rep] += 1
                    cnt[e_rem] -= 1
                    amrt[e_rep] = new_a_rep
                    amrt[e_rem] = new_a_rem
                    load = new_load_B
                else:
                    _, d_cold, sh, sc, e_a, e_b, new_load_A = best
                    layer[d_hot, sh] = e_b
                    layer[d_cold, sc] = e_a
                    load = new_load_A
                continue

            applied = False

            # ---- PHASE B (1 transmit): walk d_cold from coldest to warmest ----
            for d_cold_i in sorted_cold:
                d_cold = int(d_cold_i)
                # Early termination: if the load gap is too small even an ideal
                # half-share split can't clear thresh_B. Subsequent d_cold are
                # closer to d_hot, so they can't either -> break the d_cold walk.
                # The (load[d_hot]-load[d_cold])/(2*mean_load) upper bound is
                # tight for phase A but not phase B (amrt dilution can move
                # other devices' loads), so strict_b_prune disables it.
                if not strict_b_prune:
                    max_possible_gain = (load[d_hot] - load[d_cold]) / (2.0 * mean_load)
                    if max_possible_gain <= thresh_B:
                        break

                slots_cold = layer[d_cold]
                a_cold_slots = amrt[slots_cold]
                cnt_cold_slots = cnt[slots_cold]

                pairs = []
                for sh in range(S):
                    e_rep_cand = int(slots_hot[sh])
                    for sc in range(S):
                        if cnt_cold_slots[sc] <= 1:
                            continue
                        e_rem_cand = int(slots_cold[sc])
                        if e_rep_cand == e_rem_cand:
                            continue
                        if unique_per_device:
                            # After overwriting slot sc, would e_rep also live
                            # at some OTHER slot on d_cold? If yes, that move
                            # creates a duplicate on d_cold -> skip.
                            duplicate = False
                            for t in range(S):
                                if t != sc and int(slots_cold[t]) == e_rep_cand:
                                    duplicate = True
                                    break
                            if duplicate:
                                continue
                        if cov_thresh_l is not None and cov_l is not None:
                            # Filter pairs of DIFFERENT experts only — the
                            # same-expert case is handled by unique_per_device
                            # (and the quantile threshold itself was computed
                            # over off-diagonal entries).
                            too_covariate = False
                            for t in range(S):
                                if t == sc:
                                    continue
                                e_other = int(slots_cold[t])
                                if e_other == e_rep_cand:
                                    continue
                                if abs(float(cov_l[e_rep_cand, e_other])) > cov_thresh_l:
                                    too_covariate = True
                                    break
                            if too_covariate:
                                continue
                        pairs.append(
                            (float(a_hot_slots[sh] - a_cold_slots[sc]),
                             sh, sc, e_rep_cand, e_rem_cand)
                        )
                pairs.sort(reverse=True)

                for _exp_gain, sh, sc, e_rep, e_rem in pairs:
                    new_a_rep = w_l[e_rep] / (cnt[e_rep] + 1)
                    new_a_rem = w_l[e_rem] / (cnt[e_rem] - 1)
                    delta_rep = new_a_rep - amrt[e_rep]
                    delta_rem = new_a_rem - amrt[e_rem]

                    rep_count = (layer == e_rep).sum(axis=1)
                    rem_count = (layer == e_rem).sum(axis=1)
                    new_load = load + rep_count * delta_rep + rem_count * delta_rem
                    new_load[d_cold] += new_a_rep - new_a_rem

                    new_par = new_load.max() / mean_load
                    if cur_par - new_par > thresh_B:
                        layer[d_cold, sc] = e_rep
                        cnt[e_rep] += 1
                        cnt[e_rem] -= 1
                        amrt[e_rep] = new_a_rep
                        amrt[e_rem] = new_a_rem
                        load = new_load
                        applied = True
                        break

                if applied:
                    break

            if applied:
                continue

            # ---- PHASE A (2 transmits): also walk d_cold ----
            for d_cold_i in sorted_cold:
                d_cold = int(d_cold_i)
                max_possible_gain = (load[d_hot] - load[d_cold]) / (2.0 * mean_load)
                if max_possible_gain <= thresh_A:
                    break

                slots_cold = layer[d_cold]
                a_cold_slots = amrt[slots_cold]

                sh = int(a_hot_slots.argmax())
                sc = int(a_cold_slots.argmin())
                e_a = int(slots_hot[sh])
                e_b = int(slots_cold[sc])
                if e_a == e_b:
                    continue

                # Uniqueness filter for the swap: e_b would land on d_hot at
                # slot sh; reject if e_b is already on d_hot elsewhere.
                # Similarly e_a -> d_cold at sc.
                if unique_per_device:
                    dup = False
                    for t in range(S):
                        if t != sh and int(slots_hot[t]) == e_b:
                            dup = True; break
                        if t != sc and int(slots_cold[t]) == e_a:
                            dup = True; break
                    if dup:
                        continue
                # Covariance-quantile filter for the swap.
                # DIFFERENT-expert pairs only — same-expert pairs are
                # already rejected above by unique_per_device.
                if cov_thresh_l is not None and cov_l is not None:
                    too_covariate = False
                    for t in range(S):
                        if t != sh:
                            e_other = int(slots_hot[t])
                            if e_other != e_b and abs(float(cov_l[e_b, e_other])) > cov_thresh_l:
                                too_covariate = True; break
                        if t != sc:
                            e_other = int(slots_cold[t])
                            if e_other != e_a and abs(float(cov_l[e_a, e_other])) > cov_thresh_l:
                                too_covariate = True; break
                    if too_covariate:
                        continue

                new_load = load.copy()
                new_load[d_hot] = load[d_hot] - amrt[e_a] + amrt[e_b]
                new_load[d_cold] = load[d_cold] - amrt[e_b] + amrt[e_a]
                new_par = new_load.max() / mean_load
                if cur_par - new_par > thresh_A:
                    layer[d_hot, sh] = e_b
                    layer[d_cold, sc] = e_a
                    load = new_load
                    applied = True
                    break

            if not applied:
                break  # neither B nor A profitable across any d_cold

    return out


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


def select_changed_layers(
    proposed: np.ndarray,
    current: Optional[np.ndarray],
    w: np.ndarray,
    stats: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, bool]:
    """Migrate only layers whose proposed deployment differs from current.

    Natural complement to build_incremental: it only proposes changes for
    layers where local search found a profitable move, so the un-touched
    rows are bit-equal and skipped here -> zero transmit, zero blackout
    iters for those layers.
    """
    if current is None:
        return np.arange(proposed.shape[0], dtype=np.int64), True
    L = proposed.shape[0]
    diff = np.any(
        proposed.reshape(L, -1) != current.reshape(L, -1),
        axis=1,
    )
    changed = np.where(diff)[0].astype(np.int64)
    return changed, changed.size > 0


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
    """Bundle of (ema config, estimator, builder, selector, initial-layout).

    `initial_layout` controls what `PlacementMemory.shadow` is seeded with on
    the very first rebalance call. The simulator's own `cur_deploy_table`
    starts as the round-robin pattern (init_deploy_table default=False), so
    the cheapest call-1 migration cost comes from also starting our shadow
    there and only emitting moves we actually want.

       "round_robin"  : match the simulator's initial layout exactly. With
                        select_changed_layers, call 1's transmit equals the
                        cost of just the moves the builder proposes.
       "dseplb"       : seed with a from-scratch DS-EPLB layout (using a
                        plain window-sum for predictable init). Old behavior;
                        gives a strong initial PAR but costs ~22K transmits
                        on call 1.

    Defaults reproduce DS-EPLB bit-for-bit (w_sum ignores EMA state) on the
    deployment_table output; the only behavior change at the simulator level
    is that `pm.shadow` is now eagerly populated before build_deployment.
    """
    ema:              EmaConfig = field(default_factory=EmaConfig)
    estimate_w:       Callable = w_sum
    build_deployment: Callable = build_dseplb
    select_layers:    Callable = select_all
    initial_layout:   str      = "round_robin"
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

    # Eagerly initialize PlacementMemory on the first call. This means
    # `current` (the shadow) is always non-None by the time we hit the
    # builder, so incremental builders can hill-climb without a special
    # from-scratch path.
    pm = _PLACE.get(key)
    if pm is None:
        if s.initial_layout == "round_robin":
            initial = _init_shadow(L, D, S, E)
        elif s.initial_layout == "dseplb":
            # Use plain window-sum (no variance bias) for predictable init.
            w_init = np.asarray(hotness, dtype=np.float64).sum(axis=0)
            initial = build_dseplb(w_init, D, R)
        else:
            raise ValueError(
                f"Strategy.initial_layout must be 'round_robin' or 'dseplb', "
                f"got {s.initial_layout!r}"
            )
        pm = PlacementMemory(shadow=initial)
        _PLACE[key] = pm

    ctx = EstimatorContext(hotness=hotness, trace_mem=tm, place_mem=pm, ema_cfg=s.ema)

    # Stage A: derive w (and stats) from raw window + EMA state + last layout.
    w, stats = s.estimate_w(ctx)

    # Make the raw hotness window available to Stage B so builders can derive
    # within-window quantities (e.g. build_incremental's cov_source='window')
    # without changing Stage A.
    if stats is None:
        stats = {}
    stats.setdefault("hotness", hotness)

    # Stage B: produce a proposed deployment table.
    shadow = pm.shadow
    proposed = s.build_deployment(w, D, R, stats=stats, current=shadow)

    # Stage C: pick which layers to actually migrate.
    layers_priority, change = s.select_layers(proposed, shadow, w, stats)

    # Framework: persist shadow update so the next call's selector has truth.
    if change and len(layers_priority) > 0:
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
    """LPT bin-packing into num_packs equal-cardinality bins.

    Same algorithm as the upstream deepseek-ai/EPLB (strict LPT: for each
    item in weight-descending order, place into the least-loaded bin with
    remaining capacity). The implementation uses numpy buffers internally
    for the inner loop so element access is microseconds instead of the
    ~10-50 us of iterating a torch tensor in Python; output is converted
    back to torch at the end. Bit-identical to the original.
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(
            weight.size(-1), dtype=torch.int64, device=weight.device
        ).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices_np = weight.float().sort(-1, descending=True).indices.cpu().numpy()
    weight_np = weight.float().cpu().numpy()
    pack_index_np = np.full((num_layers, num_groups), -1, dtype=np.int64)
    rank_in_pack_np = np.full_like(pack_index_np, -1)

    for i in range(num_layers):
        pack_weights = [0.0] * num_packs
        pack_items = [0] * num_packs
        row_idx = indices_np[i]
        row_w = weight_np[i]
        for group in row_idx:
            g = int(group)
            pack = min(
                (j for j in range(num_packs) if pack_items[j] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            pack_index_np[i, g] = pack
            rank_in_pack_np[i, g] = pack_items[pack]
            pack_weights[pack] += float(row_w[g])
            pack_items[pack] += 1
    return torch.from_numpy(pack_index_np), torch.from_numpy(rank_in_pack_np)


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
