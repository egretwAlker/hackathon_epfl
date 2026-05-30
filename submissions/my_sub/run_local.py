#!/usr/bin/env python3
"""Evaluate one strategy on locally-available LmSys traces.

Loads strategies/<NAME>.json, instantiates the Strategy at runtime, runs the
simulator loop, and compares against the DS-EPLB reference (score=100).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from quickstart import (  # noqa: E402
    MODEL_SHAPES,
    init_deploy_table,
    cal_par_per_iter,
    compute_redeploy_cost,
    load_trace,
    modeled_runtime_seconds,
)
from eplb_algorithms import rebalance as _ref_rebalance  # noqa: E402

import submission  # noqa: E402
import strategies  # noqa: E402


def run_loop(
    rebalance_fn,
    hotness: np.ndarray,
    ep: int,
    n_layers: int,
    n_experts: int,
    collection_interval: int,
) -> dict:
    """Generic simulator loop. rebalance_fn(window) -> (change, lp, dep, aux).
    Mirrors quickstart.run_ds_eplb but lets the caller plug in any rebalancer."""
    cur_deploy_table = np.array(
        [init_deploy_table(ep, n_experts, ep, default=False) for _ in range(n_layers)]
    )
    next_deploy_table = np.zeros_like(cur_deploy_table)

    redeploy_finish_iter = 0
    expert_ready = True
    cur_layers_priority: list[int] = []
    transmit_amount = 0
    pars_sum = 0.0
    count = 0
    rebalance_wall = 0.0

    for i in range(1, len(hotness) + 1):
        cur_hotness = hotness[i - 1]
        pars = cal_par_per_iter(cur_hotness, cur_deploy_table)
        pars_sum += float(pars.sum())
        count += pars.size

        if not expert_ready and cur_layers_priority:
            adjust_layer_idx = cur_layers_priority.pop(0)
            transmit_amount += compute_redeploy_cost(
                cur_deploy_table[adjust_layer_idx],
                next_deploy_table[adjust_layer_idx],
            )
            cur_deploy_table[adjust_layer_idx] = next_deploy_table[adjust_layer_idx]

        if len(cur_layers_priority) == 0 and not expert_ready:
            expert_ready = True
            redeploy_finish_iter = i

        if i == redeploy_finish_iter + collection_interval + 1 and expert_ready:
            window = hotness[i - collection_interval:i]
            t0 = time.time()
            change, layers_priority, deployment_table, _ = rebalance_fn(window)
            rebalance_wall += time.time() - t0
            if change:
                selected = np.asarray(layers_priority, dtype=np.int64)
                if selected.size:
                    cur_layers_priority = selected.tolist()
                    expert_ready = False
                    next_deploy_table[selected] = np.asarray(deployment_table)[selected]

    return {
        "mean_par":        pars_sum / count,
        "transmit_amount": transmit_amount,
        "rebalance_wall":  rebalance_wall,
    }


def run_reference(hotness, ep, n_layers, n_experts, collection_interval) -> dict:
    """DS-EPLB baseline (the score=100 reference)."""
    fn = _ref_rebalance(ep, ep, "deepseek")
    return run_loop(fn, hotness, ep, n_layers, n_experts, collection_interval)


def run_strategy(strategy, hotness, ep, n_layers, n_experts, collection_interval) -> dict:
    """Run a Strategy object against the simulator loop. If the strategy's
    EmaConfig has track_corr=True, also captures a per-case summary of the
    final correlation/covariance matrix (off-diagonal stats per layer)."""
    submission.reset_state()
    submission.set_strategy(strategy)
    fn = lambda h: submission.rebalance(h, ep, ep)  # noqa: E731
    result = run_loop(fn, hotness, ep, n_layers, n_experts, collection_interval)

    # Diagnostic: pull final corr/cov matrix from the framework's TraceMemory
    # for inspection. Only present if the strategy enabled track_corr.
    key = (n_layers, n_experts, ep, ep)
    tm = submission._TRACE.get(key)
    if tm is not None and tm.M is not None and tm.n_corr_iters > 0:
        result["corr_summary"] = _summarize_corr_cov(tm, strategy.ema)
    return result


def _summarize_corr_cov(tm, cfg) -> dict:
    """Aggregate off-diagonal correlation and covariance over all layers.

    Returns:
      corr.{mean_abs, p50_abs, p95_abs, p99_abs, max_abs} in [0, 1]
      cov.{mean_abs, p50_abs, p95_abs, max_abs}             raw units (token-count^2)
    Off-diagonal mask excludes the (e, e) self-pairs.
    """
    import numpy as _np
    n_c = tm.n_corr_iters
    if cfg.bias_correction:
        m1_hat = tm.m1 / max(1.0 - cfg.beta1 ** tm.n_iters, 1e-30)
        M_hat  = tm.M  / max(1.0 - cfg.beta_corr ** n_c,    1e-30)
    else:
        m1_hat = tm.m1.copy()
        M_hat  = tm.M.copy()

    cov = M_hat - m1_hat[:, :, None] * m1_hat[:, None, :]
    diag = _np.diagonal(cov, axis1=1, axis2=2)
    denom = _np.sqrt(_np.maximum(diag[:, :, None] * diag[:, None, :], 1e-30))
    corr = _np.clip(cov / denom, -1.0, 1.0)

    L_, E_, _ = corr.shape
    off_mask = ~_np.eye(E_, dtype=bool)              # (E, E)
    off_corr = corr[:, off_mask]                     # (L, E*(E-1))
    off_cov  = cov[:,  off_mask]
    abs_corr = _np.abs(off_corr).ravel()
    abs_cov  = _np.abs(off_cov).ravel()

    return {
        "n_corr_iters":   int(n_c),
        "corr_mean_abs":  float(abs_corr.mean()),
        "corr_p50_abs":   float(_np.quantile(abs_corr, 0.50)),
        "corr_p95_abs":   float(_np.quantile(abs_corr, 0.95)),
        "corr_p99_abs":   float(_np.quantile(abs_corr, 0.99)),
        "corr_max_abs":   float(abs_corr.max()),
        "cov_mean_abs":   float(abs_cov.mean()),
        "cov_p50_abs":    float(_np.quantile(abs_cov, 0.50)),
        "cov_p95_abs":    float(_np.quantile(abs_cov, 0.95)),
        "cov_max_abs":    float(abs_cov.max()),
    }


def score(reference: dict, candidate: dict) -> dict:
    """Leaderboard score given reference and candidate metrics."""
    ref_time = modeled_runtime_seconds(reference["mean_par"], reference["transmit_amount"])
    cand_time = modeled_runtime_seconds(candidate["mean_par"], candidate["transmit_amount"])
    return {
        "ref_time":  ref_time,
        "cand_time": cand_time,
        "score":     100.0 * ref_time / cand_time,
    }


def _row(model, ep, method, par, transmit, time_s, score_s):
    return [model, str(ep), method, f"{par:.4f}", str(transmit), f"{time_s:.2f}s", f"{score_s:.2f}"]


def _print_table(rows):
    header = ["model", "EP", "method", "PAR", "transmit", "modeled_time", "score"]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    fmt = " | ".join(f"{{:>{w}}}" for w in widths)
    print(fmt.format(*header))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default="deepseek-default",
                   help="name (filename stem) under strategies/")
    p.add_argument("--model", choices=sorted(MODEL_SHAPES), default="Qwen3")
    p.add_argument("--ep", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=1200)
    p.add_argument("--collection-interval", type=int, default=1024)
    p.add_argument("--all", action="store_true",
                   help="Evaluate on both DS-R1 and Qwen3 LmSys traces.")
    p.add_argument("--list", action="store_true",
                   help="List available strategy names and exit.")
    args = p.parse_args()

    if args.list:
        for name in strategies.list_strategies():
            print(name)
        return

    available = strategies.list_strategies()
    if args.strategy not in available:
        sys.exit(f"unknown strategy {args.strategy!r}. Available: {available}")

    spec = strategies.load_spec(args.strategy)
    strategy = strategies.build_strategy(spec, submission)

    models = sorted(MODEL_SHAPES) if args.all else [args.model]
    rows = []
    worst_wall = 0.0
    for model in models:
        n_layers, n_experts = MODEL_SHAPES[model]
        hotness = load_trace(REPO_ROOT, model, "LmSys", args.max_iters)
        ref = run_reference(hotness, args.ep, n_layers, n_experts, args.collection_interval)
        cand = run_strategy(strategy, hotness, args.ep, n_layers, n_experts, args.collection_interval)
        s = score(ref, cand)
        worst_wall = max(worst_wall, cand["rebalance_wall"])
        rows.append(_row(model, args.ep, "DS-EPLB",
                         ref["mean_par"], ref["transmit_amount"], s["ref_time"], 100.0))
        rows.append(_row(model, args.ep, args.strategy,
                         cand["mean_par"], cand["transmit_amount"], s["cand_time"], s["score"]))

    print(f"strategy: {args.strategy}\n")
    _print_table(rows)
    print(f"\nrebalance() worst-case cumulative wall-time: {worst_wall:.3f} s  /  5.0 s budget")


if __name__ == "__main__":
    main()
