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

## Run

```bash
python dynamic_lb_simulator.py
```

The default script evaluates the bundled experiment grid when the corresponding traces are
available.
