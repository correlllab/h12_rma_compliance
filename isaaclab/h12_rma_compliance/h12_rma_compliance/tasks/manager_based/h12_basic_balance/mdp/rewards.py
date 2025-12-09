# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for H12 Basic Balance Task."""

import torch
from isaaclab.managers import SceneEntityCfg


def alive_bonus(env, cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for being alive. Fixed bonus per step."""
    return torch.ones(env.num_envs, device=env.device)


def base_height_l2(env, cfg: SceneEntityCfg, target_height: float = 1.04) -> torch.Tensor:
    """Reward for maintaining target base height (L2 distance from target)."""
    # Get current base height
    base_pos = env.scene["robot"].data.root_pos_w[:, 2]
    
    # Reward based on distance from target height (quadratic penalty)
    height_error = torch.abs(base_pos - target_height)
    reward = torch.exp(-2.0 * height_error)
    
    return reward


def base_height_below_threshold(env, cfg: SceneEntityCfg, threshold: float = 0.4) -> torch.Tensor:
    """Penalty for falling below minimum height threshold."""
    # Get current base height
    base_pos = env.scene["robot"].data.root_pos_w[:, 2]
    
    # Penalty if below threshold
    penalty = torch.where(base_pos < threshold, torch.ones_like(base_pos), torch.zeros_like(base_pos))
    
    return -penalty
