#!/usr/bin/env python3
"""Aggregate a sweep CSV by strategy across all (model, EP) cells.

Usage:
  # mean / min / max score per strategy, sorted by mean
  python analyze_sweep.py experiments/results/my_sub/incremental_grid.csv

  # per-cell breakdown for one strategy (find which cell is dragging it down)
  python analyze_sweep.py experiments/results/my_sub/incremental_grid.csv \\
         --strategy "beta1=0.999_beta2=0.9999_lam=16.0_..._k=None_threshold=0.0"

  # sort by worst-case score instead of mean (find robust strategies)
  python analyze_sweep.py <csv> --sort min

The CSV is read live — you can run this while the sweep is still in progress
(rows are flushed per-call).
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path, help="Path to sweep CSV")
    p.add_argument("--top", type=int, default=15, help="Rows to show in aggregate view")
    p.add_argument("--sort", choices=["mean", "min", "max", "std"], default="mean",
                   help="Aggregate stat to sort by (default: mean)")
    p.add_argument("--strategy", help="If set, show per-cell breakdown for this strategy")
    args = p.parse_args()

    if not args.csv.exists():
        sys.exit(f"file not found: {args.csv}")

    with args.csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("CSV has no rows yet")

    # Per-cell breakdown for a specific strategy.
    if args.strategy:
        matches = [r for r in rows if r["strategy"] == args.strategy]
        if not matches:
            sys.exit(f"no rows for strategy {args.strategy!r}")
        matches.sort(key=lambda r: (r["model"], int(r["ep"])))
        print(f"Per-cell scores for: {args.strategy}\n")
        print(f"  {'model':<6} {'EP':>4}  {'PAR':>8} {'transmit':>10} {'score':>8}")
        for r in matches:
            print(f"  {r['model']:<6} {r['ep']:>4}  "
                  f"{float(r['sub_par']):>8.4f} {int(r['sub_transmit']):>10d} "
                  f"{float(r['score']):>8.2f}")
        scores = [float(r["score"]) for r in matches]
        print(f"\n  mean = {statistics.mean(scores):.2f},  "
              f"min = {min(scores):.2f},  max = {max(scores):.2f},  "
              f"std = {statistics.stdev(scores) if len(scores) > 1 else 0:.2f}  "
              f"({len(scores)} cells)")
        return

    # Aggregate by strategy.
    by_strat = collections.defaultdict(list)
    for r in rows:
        by_strat[r["strategy"]].append(float(r["score"]))

    summary = []
    for name, scores in by_strat.items():
        summary.append({
            "name":   name,
            "cells":  len(scores),
            "mean":   statistics.mean(scores),
            "min":    min(scores),
            "max":    max(scores),
            "std":    statistics.stdev(scores) if len(scores) > 1 else 0.0,
        })
    summary.sort(key=lambda x: -x[args.sort])

    cells_per_strat = collections.Counter(r["strategy"] for r in rows)
    max_cells = max(cells_per_strat.values()) if cells_per_strat else 0
    n_partial = sum(1 for s in summary if s["cells"] < max_cells)

    total_strats = len(summary)
    total_rows = sum(s["cells"] for s in summary)
    print(f"Strategies: {total_strats}   total rows: {total_rows}   "
          f"max cells/strategy: {max_cells}", end="")
    if n_partial:
        print(f"   ({n_partial} strategies still in progress)")
    else:
        print()
    print(f"Sorted by: {args.sort}\n")

    print(f"  {'mean':>7} {'min':>7} {'max':>7} {'std':>6} {'cells':>5}   strategy")
    print(f"  {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*5}   {'-'*8}")
    for s in summary[: args.top]:
        partial = " *" if s["cells"] < max_cells else ""
        print(f"  {s['mean']:7.2f} {s['min']:7.2f} {s['max']:7.2f} "
              f"{s['std']:6.2f} {s['cells']:>5}   {s['name']}{partial}")
    if n_partial:
        print("\n* = strategy hasn't been evaluated on all cells yet")


if __name__ == "__main__":
    main()
