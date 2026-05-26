from enum import Enum, auto
from typing import Optional
import os
import torch
import numpy as np
from . import deepseek


class EplbAlgorithm(Enum):
    deepseek = auto()
    deepseek_hierarchical = auto()
    proposed = auto()


def rebalance_experts(
        tokens_per_expert: torch.Tensor,
        num_physical_experts: int = 320,
        num_local_physical_experts: int = 64,
        num_groups: Optional[int] = 1,
        num_nodes: int = 1,
        algorithm: EplbAlgorithm = EplbAlgorithm.deepseek,
):
    if algorithm in [EplbAlgorithm.deepseek, EplbAlgorithm.deepseek_hierarchical]:
        physical_to_logical_map, logical_to_all_physical_map, expert_count = \
            deepseek.rebalance_experts(
                weight=tokens_per_expert.sum(dim=0),
                num_replicas=num_physical_experts,
                num_groups=num_groups,
                num_nodes=num_nodes,
                num_gpus=num_physical_experts // num_local_physical_experts,
                enable_hierarchical=algorithm == EplbAlgorithm.deepseek_hierarchical,
            )
        return physical_to_logical_map, logical_to_all_physical_map, expert_count, list(
            range(physical_to_logical_map.shape[0]))
    if algorithm in [EplbAlgorithm.proposed]:
        # TODO: add algorithm details
        pass
    raise NotImplementedError


def rebalance(n_device=64, n_red_expert=64, algorithm='deepseek'):
    def rebalance_(hotness, n_device=n_device, n_red_expert=n_red_expert, algorithm=algorithm):
        n_expert = hotness.shape[-1]
        physical_to_logical_map, logical_to_all_physical_map, expert_count, priority = \
            rebalance_experts(torch.from_numpy(hotness), n_expert + n_red_expert, (n_expert + n_red_expert) // n_device,
                              algorithm=EplbAlgorithm[algorithm])

        return len(priority) > 0, np.array(priority), physical_to_logical_map.numpy().reshape((-1, n_device, (n_expert + n_red_expert) // n_device)), None

    return rebalance_
