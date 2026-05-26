1. This simulator is used to evaluate the performance of dynamic load balancing algorithms for MoE LLMs under the EP paradigm.
    1) Two baselines are provided: a) Default: disables load balancing features; b) DS-EPLB: the load balancing algorithm open-sourced by DeepSeek;
    2) Four hotness datasets are considered, all collected from real-world scenarios: a) ShareGPT; b) WildChat; c) LmSys; d) A mixture of three datasets (Gsm8k + BoolQ + HumanEval);
    3) Two metrics are evaluated: a) PAR: the Peak-Average Ratio of inter-device loads, where per-device load is defined as the summed hotness of experts on the device; b) Expert transmission amounts: indicating the D2D transmission cost of expert weights for redeployment;
    4) EP sizes of 32, 64, 128, and 256 are evaluated;
    5) The focus is on the decoding phase.
2. The proposed algorithm should achieve a better PAR than DS-EPLB, while achieving equal or lower transmission amounts than DS-EPLB.
3. The simulator is executed by running dynamic_lb_simulator.py, and the proposed algorithm should be added at positions marked #TODO.