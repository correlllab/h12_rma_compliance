# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def alive_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive. Fixed bonus per step."""
    return torch.ones(env.num_envs, device=env.device)


def base_height_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float = 1.04) -> torch.Tensor:
    """Reward for maintaining target base height (L2 distance from target)."""
    # Get current base height
    asset: Articulation = env.scene[asset_cfg.name]
    base_pos = asset.data.root_pos_w[:, 2]
    
    # Reward based on distance from target height (quadratic penalty)
    height_error = torch.abs(base_pos - target_height)
    reward = torch.exp(-2.0 * height_error)
    
    return reward


def base_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for base velocity. Encourages robot to stay still."""
    # Get base linear velocity
    asset: Articulation = env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_w
    
    # Calculate velocity magnitude
    vel_norm = torch.norm(base_lin_vel, dim=-1)
    
    # Penalty: negative reward proportional to velocity
    penalty = -vel_norm
    
    return penalty

