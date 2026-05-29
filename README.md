# MoE Competition Simulator

This repository contains the simulator code for the MoE dynamic load-balancing competition.
It includes one small sample trace for each supported model so local smoke tests can run
without the full private trace set.

## Included Sample Traces

```text
trace/
  DS-R1/
    LmSys.npy
  Qwen3/
    LmSys.npy
```

The full trace set is intentionally not committed. Generated outputs and additional local traces
remain ignored by git.

## Reference Submissions

Participant-style reference implementations are available under `submissions/`:

- `submissions/smoke/submission.py`: minimal no-redeployment API smoke test.
- `submissions/hot_expert_baseline/submission.py`: simple baseline that assigns redundant slots
  to the hottest experts in each layer.

## Install

```bash
python -m pip install -r requirements.txt
```

## Quick Start

After cloning, pull the sample trace files from Git LFS and run the quick-start simulator:

```bash
git lfs pull
python quickstart.py
```

This runs a bounded `Qwen3 / LmSys / EP32` sample using the committed traces and prints PAR,
total transit, estimated transmission time, estimated total time, and score for `Default` and
`DS-EPLB`.

To run both committed sample traces:

```bash
python quickstart.py --all-samples
```

## Run

```bash
python dynamic_lb_simulator.py
```

The default script evaluates the full experiment grid and expects the corresponding full trace set
to be available locally. Use `quickstart.py` for a fresh-clone smoke test.
