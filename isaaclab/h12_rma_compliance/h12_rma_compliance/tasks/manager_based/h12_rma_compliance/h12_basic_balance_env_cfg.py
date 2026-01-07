# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Basic H12 Balancing Environment - Learn to stand and balance upright.

This is a minimal environment where the H12 humanoid learns to:
1. Stay alive (alive bonus)
2. Maintain standing posture (base height reward)

No other complex rewards or task-specific objectives.
"""

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.envs import mdp
from h12_rma_compliance.assets.unitree import H12_CFG_HANDLESS

# Import custom reward functions from local mdp module
from . import mdp as local_mdp


##
# Scene definition
##


@configclass
class H12BasicBalanceSceneCfg(InteractiveSceneCfg):
    """Configuration for H12 basic balancing scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = H12_CFG_HANDLESS.replace(prim_path="{ENV_REGEX_NS}/Robot")


##
# Actions definition
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Joint position control - 13 DOF (legs + torso only, no upper body)
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_yaw_joint",
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",

            "right_hip_yaw_joint",
            "right_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",

            "torso_joint",
        ],
        scale=0.25,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Basic proprioceptive observations
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Reset base position and velocity
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # Reset robot joints to default positions
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


def base_height_below_threshold(env, cfg: SceneEntityCfg, threshold: float = 0.4) -> torch.Tensor:
    """Penalty for falling below minimum height threshold."""
    # Get current base height
    base_pos = env.scene[cfg.name].data.root_pos_w[:, 2]
    
    # Return boolean tensor: True if below threshold (triggers termination)
    return base_pos < threshold


@configclass
class RewardsCfg:
    """Reward terms for the MDP - MINIMAL SET."""

    # (1) Alive bonus: reward for staying upright (not falling)
    alive = RewTerm(
        func=local_mdp.alive_bonus,
        weight=5.0,
        params={},
    )

    # (2) Standing reward: encourage maintaining height
    standing = RewTerm(
        func=local_mdp.base_height_l2,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 1.04},
    )

    # (3) Velocity penalty: encourage staying still (zero velocity)
    base_vel_penalty = RewTerm(
        func=local_mdp.base_velocity_penalty,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # (4) Knee symmetry: encourage balanced stance
    knee_symmetry = RewTerm(
        func=local_mdp.knee_symmetry_reward,
        weight=3.0,
        params={},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Base height too low (fell down)
    base_height_low = DoneTerm(
        func=base_height_below_threshold,
        params={"cfg": SceneEntityCfg("robot"), "threshold": 0.4},
    )


##
# Environment configuration
##


@configclass
class H12BasicBalanceEnvCfg(ManagerBasedRLEnvCfg):
    """Basic H12 balancing environment configuration."""

    # Scene settings
    scene: H12BasicBalanceSceneCfg = H12BasicBalanceSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10  # 10 second episodes
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
