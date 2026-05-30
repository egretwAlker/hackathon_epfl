"""Strategy specs as JSON. Shared helper for build.py / run_local.py / sweep.py.

A strategy JSON describes the three pipeline stages by referring to function
names defined in submission.py. Example:

    {
      "estimate_w":       { "fn": "w_ema", "params": { "alpha": 0.3 } },
      "build_deployment": { "fn": "build_dseplb_variance_aware",
                            "params": { "beta": 1.0 } },
      "select_layers":    { "fn": "select_top_k_par_gain",
                            "params": { "k": 20, "threshold": 0.02 } }
    }

Stages without parameters can omit the "params" field. The filename stem
(e.g. "smart" from "smart.json") becomes the strategy name unless overridden
with a top-level "name" field.

`_comment` (or any underscore-prefixed key) is ignored.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
STRATEGIES_DIR = HERE / "strategies"

# Whitelist of function names that strategy JSONs may reference. Lives here so
# we can validate before doing getattr() on submission and so build.py knows
# the universe of options without importing submission.
ALLOWED_FNS = {
    "w_sum", "w_ema",
    "build_dseplb",
    "select_all", "select_top_k_par_gain",
}


def json_path(name: str) -> Path:
    """Resolve a name to a JSON config path. Raises FileNotFoundError if absent."""
    p = STRATEGIES_DIR / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def list_strategies() -> List[str]:
    """All registered strategy names (filename stems in strategies/)."""
    return sorted(p.stem for p in STRATEGIES_DIR.glob("*.json"))


def load_spec(name: str) -> Dict[str, Any]:
    """Read and validate a strategy spec. Returns a dict with stages + name."""
    path = json_path(name)
    raw = json.loads(path.read_text())
    spec = {k: v for k, v in raw.items() if not k.startswith("_")}
    spec.setdefault("name", path.stem)
    _validate(spec)
    return spec


def _validate(spec: Dict[str, Any]) -> None:
    for stage in ("estimate_w", "build_deployment", "select_layers"):
        if stage not in spec:
            raise ValueError(f"strategy spec missing stage {stage!r}")
        if "fn" not in spec[stage]:
            raise ValueError(f"stage {stage!r} missing 'fn'")
        fn = spec[stage]["fn"]
        if fn not in ALLOWED_FNS:
            raise ValueError(
                f"unknown function {fn!r} in stage {stage!r}. "
                f"Allowed: {sorted(ALLOWED_FNS)}"
            )


def build_strategy(spec: Dict[str, Any], submission_module):
    """Construct a runtime Strategy from a spec. Used by run_local / sweep."""
    def make(stage):
        fn = getattr(submission_module, stage["fn"])
        params = stage.get("params") or {}
        return partial(fn, **params) if params else fn

    ema_spec = spec.get("ema") or {}
    ema_cfg = submission_module.EmaConfig(**ema_spec)

    return submission_module.Strategy(
        ema              = ema_cfg,
        estimate_w       = make(spec["estimate_w"]),
        build_deployment = make(spec["build_deployment"]),
        select_layers    = make(spec["select_layers"]),
        name             = spec["name"],
    )


def codegen_strategy(spec: Dict[str, Any]) -> str:
    """Generate the Python expression that constructs this Strategy.

    Used by build.py to inject a baked strategy into submission.py at zip time.
    The output assumes Strategy, EmaConfig, partial, and the stage function
    names are all in scope (they are, inside submission.py).
    """
    def make_code(stage):
        fn = stage["fn"]
        params = stage.get("params") or {}
        if not params:
            return fn
        kw = ", ".join(f"{k}={v!r}" for k, v in sorted(params.items()))
        return f"partial({fn}, {kw})"

    ema_spec = spec.get("ema") or {}
    if ema_spec:
        ema_kw = ", ".join(f"{k}={v!r}" for k, v in sorted(ema_spec.items()))
        ema_code = f"EmaConfig({ema_kw})"
    else:
        ema_code = "EmaConfig()"

    return (
        f"Strategy(\n"
        f"    ema              = {ema_code},\n"
        f"    estimate_w       = {make_code(spec['estimate_w'])},\n"
        f"    build_deployment = {make_code(spec['build_deployment'])},\n"
        f"    select_layers    = {make_code(spec['select_layers'])},\n"
        f"    name             = {spec['name']!r},\n"
        f")"
    )
