# MoE Competition Simulator

This repository contains only the simulator code for the MoE dynamic load-balancing competition.
Trace files and generated outputs are intentionally excluded.

## Install

```bash
python -m pip install -r requirements.txt
```

## Expected Trace Layout

Place traces outside git using the same layout as the original simulator:

```text
trace/
  DS-R1/
    ShareGPT.npy
    WildChat.npy
    LmSys.npy
    Mix.npy
  Qwen3/
    ShareGPT.npy
    WildChat.npy
    LmSys.npy
```

## Run

```bash
python dynamic_lb_simulator.py
```

The default script evaluates the bundled experiment grid when traces are available.
