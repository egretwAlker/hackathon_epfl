# MoE Competition Simulator

This repository contains the simulator code for the MoE dynamic load-balancing competition.
It includes `LmSys.npy` sample traces for `DS-R1` and `Qwen3`; larger traces and generated
outputs remain excluded.

## Install

```bash
python -m pip install -r requirements.txt
```

## Trace Layout

The committed sample traces use this layout:

```text
trace/
  DS-R1/
    LmSys.npy
  Qwen3/
    LmSys.npy
```

Additional local traces can use the same directory structure, but they are ignored by git.

## Reference Submissions

The `submissions/` directory contains participant-style examples:

```text
submissions/
  smoke/submission.py
  hot_expert_baseline/submission.py
```

## Run

```bash
python dynamic_lb_simulator.py
```

The default script evaluates the bundled experiment grid when traces are available.
