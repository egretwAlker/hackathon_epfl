#!/usr/bin/env python3
"""Run a grid sweep described by a JSON config under sweeps/.

A sweep config specifies:
  - The strategy topology (which estimate_w / build_deployment / select_layers
    functions to use, fixed across the grid).
  - Sweep axes for: EmaConfig fields, estimator kwargs, selector kwargs.
    Each leaf is either a scalar (fixed) or a list (sweep axis).
  - Which (model, EP) cases to evaluate.
  - Optional named strategies (from strategies/) to also include as rows.

Output: one CSV row per (model, EP, strategy) plus a top-N stdout summary.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SWEEPS_DIR = HERE / "sweeps"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from quickstart import MODEL_SHAPES, load_trace  # noqa: E402
import submission  # noqa: E402
import strategies as strategies_mod  # noqa: E402
from run_local import run_reference, run_strategy, score  # noqa: E402


def _as_list(v: Any) -> List[Any]:
    """Wrap scalars in a single-element list so we can take itertools.product."""
    return v if isinstance(v, list) else [v]


def _strip_comments(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _coerce_k(v):
    """Map 'all' (string) to None for the top-K selector."""
    return None if v == "all" else int(v) if v is not None else None


def _compatible_case(model: str, ep: int) -> bool:
    """Simulator requires E % D == 0 so that S = (E + R) / D is integer with R = D.
    Matches the (md == 'Qwen3' and ep == 256) skip in dynamic_lb_simulator.py:344."""
    n_experts = MODEL_SHAPES[model][1]
    return n_experts % ep == 0


def load_sweep_config(name: str) -> Dict[str, Any]:
    """Resolve `name` either as a sweeps/<name>.json or as a literal path."""
    p = SWEEPS_DIR / f"{name}.json"
    if not p.exists():
        p = Path(name)
    if not p.exists():
        raise FileNotFoundError(f"sweep config not found: {name!r}")
    return _strip_comments(json.loads(p.read_text()))


def list_sweep_configs() -> List[str]:
    return sorted(p.stem for p in SWEEPS_DIR.glob("*.json"))


def _name_strategy(ema_kw: Dict, est_kw: Dict, sel_kw: Dict,
                   bld_kw: Optional[Dict] = None) -> str:
    parts: List[str] = []
    for k, v in ema_kw.items():
        if k == "track_corr" and not v:
            continue
        if k == "bias_correction" and v:
            continue
        if k == "beta_corr" and not ema_kw.get("track_corr"):
            continue
        parts.append(f"{k}={v}")
    for k, v in est_kw.items():
        parts.append(f"{k}={v}")
    for k, v in (bld_kw or {}).items():
        parts.append(f"{k}={v}")
    for k, v in sel_kw.items():
        parts.append(f"{k}={v}")
    return "_".join(parts) or "default"


def build_grid_strategies(cfg: Dict[str, Any]) -> List[tuple]:
    """Expand the JSON sweep spec into a list of (name, Strategy)."""
    est_fn = getattr(submission, cfg["estimate_w_fn"])
    bld_fn = getattr(submission, cfg["build_deployment_fn"])
    sel_fn = getattr(submission, cfg["select_layers_fn"])

    ema_axes = {k: _as_list(v) for k, v in cfg.get("ema", {}).items()}
    est_axes = {k: _as_list(v) for k, v in cfg.get("estimator_params", {}).items()}
    bld_axes = {k: _as_list(v) for k, v in cfg.get("builder_params", {}).items()}
    sel_axes = {k: _as_list(v) for k, v in cfg.get("selector_params", {}).items()}

    if "k" in sel_axes:
        sel_axes["k"] = [_coerce_k(v) for v in sel_axes["k"]]

    ema_keys, ema_vals = list(ema_axes.keys()), list(ema_axes.values())
    est_keys, est_vals = list(est_axes.keys()), list(est_axes.values())
    bld_keys, bld_vals = list(bld_axes.keys()), list(bld_axes.values())
    sel_keys, sel_vals = list(sel_axes.keys()), list(sel_axes.values())

    out: List[tuple] = []
    for ema_combo in itertools.product(*ema_vals):
        for est_combo in itertools.product(*est_vals):
            for bld_combo in itertools.product(*bld_vals):
                for sel_combo in itertools.product(*sel_vals):
                    ema_kw = dict(zip(ema_keys, ema_combo))
                    est_kw = dict(zip(est_keys, est_combo))
                    bld_kw = dict(zip(bld_keys, bld_combo))
                    sel_kw = dict(zip(sel_keys, sel_combo))

                    ema_cfg = submission.EmaConfig(**ema_kw)
                    est = partial(est_fn, **est_kw) if est_kw else est_fn
                    bld = partial(bld_fn, **bld_kw) if bld_kw else bld_fn
                    sel = partial(sel_fn, **sel_kw) if sel_kw else sel_fn

                    name = _name_strategy(ema_kw, est_kw, sel_kw, bld_kw)
                    strat = submission.Strategy(
                        ema=ema_cfg,
                        estimate_w=est,
                        build_deployment=bld,
                        select_layers=sel,
                        name=name,
                    )
                    out.append((name, strat))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="sweep config (filename stem under sweeps/, or a path)")
    p.add_argument("--list", action="store_true",
                   help="List available sweep configs and exit.")
    args = p.parse_args()

    if args.list:
        for name in list_sweep_configs():
            print(name)
        return
    if not args.config:
        sys.exit(f"--config is required. Available: {list_sweep_configs()}")

    cfg = load_sweep_config(args.config)

    # Strategy list: grid + included named strategies
    strat_list: List[tuple] = []
    if cfg.get("include_deepseek"):
        strat_list.append(("deepseek-default", submission.DEEPSEEK_STRATEGY))
    for name in cfg.get("include_strategies", []):
        spec = strategies_mod.load_spec(name)
        strat_list.append((name, strategies_mod.build_strategy(spec, submission)))
    strat_list.extend(build_grid_strategies(cfg))

    # Cases
    models = cfg.get("models", ["Qwen3"])
    eps_list = cfg.get("eps", [32])
    max_iters = cfg.get("max_iters", 1200)
    collection_interval = cfg.get("collection_interval", 1024)
    all_cases = list(itertools.product(models, eps_list))
    cases = [(m, e) for m, e in all_cases if _compatible_case(m, e)]
    skipped = [(m, e) for m, e in all_cases if not _compatible_case(m, e)]
    if skipped:
        print(f"Skipping incompatible (model, EP) cases (E % D != 0): {skipped}")

    # Load each model's trace once.
    print(f"Loading traces for {models} ...", flush=True)
    hot_cache = {m: load_trace(REPO_ROOT, m, "LmSys", max_iters) for m in models}

    # DS-EPLB reference per (model, ep), computed once.
    print("Computing DS-EPLB references ...", flush=True)
    ref_cache: Dict[tuple, dict] = {}
    for model, ep in cases:
        n_layers, n_experts = MODEL_SHAPES[model]
        ref_cache[(model, ep)] = run_reference(
            hot_cache[model], ep, n_layers, n_experts, collection_interval
        )
        r = ref_cache[(model, ep)]
        print(f"  {model} EP{ep}: PAR={r['mean_par']:.4f}  transmit={r['transmit_amount']}")

    # Output path (live-written: each row flushed as soon as it completes).
    out_path = Path(cfg.get("out", "experiments/results/my_sub/sweep.csv"))
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["model", "ep", "strategy",
              "ds_par", "ds_transmit", "ds_time",
              "sub_par", "sub_transmit", "sub_time",
              "rebalance_wall", "score"]
    top = int(cfg.get("top", 5))

    print(f"\nLive results streaming to:\n  {out_path}")
    print(f"  -> tail -f {out_path}   to follow in another shell\n")

    # Sweep with live CSV append. Outer loop is STRATEGIES so each one's
    # full per-case set lands together; we print a one-line aggregate (mean,
    # min, max, std across the cells) the moment a strategy finishes — far
    # better proxy for the leaderboard composite score than any single-cell
    # peak.
    rows: List[Dict[str, Any]] = []
    strategy_summaries: List[Dict[str, Any]] = []
    total = len(strat_list) * len(cases)
    n = 0
    with out_path.open("w", newline="", buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        f.flush()

        for s_idx, (name, strat) in enumerate(strat_list, start=1):
            strat_rows: List[Dict[str, Any]] = []
            for model, ep in cases:
                n += 1
                n_layers, n_experts = MODEL_SHAPES[model]
                hotness = hot_cache[model]
                ref = ref_cache[(model, ep)]
                print(f"[{n:>4}/{total}] strat {s_idx:>3}/{len(strat_list)}  "
                      f"{model} EP{ep:<3}  {name}", end="", flush=True)
                cand = run_strategy(strat, hotness, ep, n_layers, n_experts, collection_interval)
                sc = score(ref, cand)
                row = {
                    "model":          model,
                    "ep":             ep,
                    "strategy":       name,
                    "ds_par":         ref["mean_par"],
                    "ds_transmit":    ref["transmit_amount"],
                    "ds_time":        sc["ref_time"],
                    "sub_par":        cand["mean_par"],
                    "sub_transmit":   cand["transmit_amount"],
                    "sub_time":       sc["cand_time"],
                    "rebalance_wall": cand["rebalance_wall"],
                    "score":          sc["score"],
                }
                rows.append(row)
                strat_rows.append(row)
                writer.writerow(
                    {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()}
                )
                f.flush()
                print(
                    f"   score={sc['score']:7.2f}   "
                    f"PAR={cand['mean_par']:.4f}  "
                    f"transmit={cand['transmit_amount']:>7d}  "
                    f"wall={cand['rebalance_wall']:5.2f}s",
                    flush=True,
                )

            # Per-strategy aggregate over the 7 cells just completed.
            scores = [r["score"] for r in strat_rows]
            mean_s = sum(scores) / len(scores)
            min_s = min(scores)
            max_s = max(scores)
            std_s = (
                (sum((x - mean_s) ** 2 for x in scores) / len(scores)) ** 0.5
                if len(scores) > 1 else 0.0
            )
            strategy_summaries.append({
                "name":  name,
                "mean":  mean_s,
                "min":   min_s,
                "max":   max_s,
                "std":   std_s,
                "cells": len(scores),
            })
            print(
                f"   ---- strategy mean={mean_s:7.2f}  min={min_s:7.2f}  "
                f"max={max_s:7.2f}  std={std_s:5.2f}  ({len(scores)} cells)  {name}\n",
                flush=True,
            )

    # Final summaries — both views.
    rows_by_score = sorted(rows, key=lambda r: -r["score"])
    print(f"\nFinal top {min(top, len(rows))} PER-CELL rows by score:")
    print(f"  {'score':>7}   {'model':<6} EP{'':<3}  strategy")
    for r in rows_by_score[:top]:
        print(f"  {r['score']:7.2f}   {r['model']:<6} EP{r['ep']:<3}  {r['strategy']}")

    strategy_summaries.sort(key=lambda s: -s["mean"])
    print(f"\nTop {min(top, len(strategy_summaries))} strategies by MEAN score "
          f"across {len(cases)} cells:")
    print(f"  {'mean':>7} {'min':>7} {'max':>7} {'std':>6} {'cells':>5}   strategy")
    for s in strategy_summaries[:top]:
        print(f"  {s['mean']:7.2f} {s['min']:7.2f} {s['max']:7.2f} "
              f"{s['std']:6.2f} {s['cells']:>5}   {s['name']}")

    print(f"\nCSV: {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
