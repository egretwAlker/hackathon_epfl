"""Throwaway: compare inc-lam16-mm20 with/without fuse_phases across cases."""
import copy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "submissions" / "my_sub"))

import strategies
import submission
from run_local import run_reference, run_strategy, score
from quickstart import MODEL_SHAPES, load_trace

spec = strategies.load_spec("inc-lam16-mm20")
CI = 1024
MAX_ITERS = 10000
cases = [("DS-R1", 32), ("DS-R1", 64), ("DS-R1", 128), ("Qwen3", 32), ("Qwen3", 64)]

print(f"{'model':<6} {'EP':<4} {'fuse':<5} {'ref_PAR':>8} {'cand_PAR':>9} "
      f"{'ref_TX':>9} {'cand_TX':>9} {'score':>7} {'wall_s':>7}")
for model, EP in cases:
    n_layers, n_experts = MODEL_SHAPES[model]
    if n_experts % EP != 0:
        continue
    hot = load_trace(REPO, model, "LmSys", MAX_ITERS)
    ref = run_reference(hot, EP, n_layers, n_experts, CI)
    for fuse in [False, True]:
        s = copy.deepcopy(spec)
        s["build_deployment"]["params"]["fuse_phases"] = fuse
        s["name"] = f"inc-lam16-mm20{'_fused' if fuse else ''}"
        strat = strategies.build_strategy(s, submission)
        t0 = time.time()
        cand = run_strategy(strat, hot, EP, n_layers, n_experts, CI)
        wall = time.time() - t0
        sc = score(ref, cand)
        print(f"{model:<6} {EP:<4d} {str(fuse):<5} "
              f"{ref['mean_par']:>8.4f} {cand['mean_par']:>9.4f} "
              f"{ref['transmit_amount']:>9d} {cand['transmit_amount']:>9d} "
              f"{sc['score']:>7.2f} {wall:>7.3f}")
