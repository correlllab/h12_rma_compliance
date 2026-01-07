"""
Sim2Sim deployment script for H12 Basic Balance on MuJoCo.

Loads an Isaac Lab trained policy and runs it in MuJoCo physics engine.
Matches Isaac Lab observation and action spaces exactly.
"""

import time
import argparse
from pathlib import Path

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml


def get_gravity_orientation(quaternion):
    """Extract gravity orientation (projected gravity) from quaternion.
    
    Args:
        quaternion: [qw, qx, qy, qz]
    
    Returns:
        Gravity vector in base frame
    """
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculate torques using PD control law.
    
    τ = kp * (q_target - q) + kd * (dq_target - dq)
    """
    return (target_q - q) * kp + (target_dq - dq) * kd


def clip_joint_positions(target_pos, lower_limits, upper_limits):
    """Clip joint positions to valid limits."""
    return np.clip(target_pos, lower_limits, upper_limits)


class H12BasicBalanceDeployer:
    """Sim2Sim deployer for H12 basic balance with Isaac Lab policy."""
    
    def __init__(self, config_path, policy_path=None):
        """
        Initialize deployer with config and optional policy.
        
        Args:
            config_path: Path to YAML config file
            policy_path: Path to TorchScript policy (.pt file), optional for PD-only mode
        """
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.xml_path = self.config["xml_path"]
        self.simulation_duration = self.config["simulation_duration"]
        self.simulation_dt = self.config["simulation_dt"]
        self.control_decimation = self.config["control_decimation"]
        
        self.kps = np.array(self.config["kps"], dtype=np.float32)
        self.kds = np.array(self.config["kds"], dtype=np.float32)
        self.default_angles = np.array(self.config["default_angles"], dtype=np.float32)
        
        self.num_actions = self.config["num_actions"]
        self.num_obs = self.config["num_obs"]
        self.num_arm_dofs = 14  # 7 per arm (3 shoulder + 1 elbow_pitch + 1 elbow_roll + 1 wrist_pitch + 1 wrist_yaw)
        self.action_scale = self.config.get("action_scale", 0.25)
        self.ang_vel_scale = self.config.get("ang_vel_scale", 0.2)
        self.dof_pos_scale = self.config.get("dof_pos_scale", 1.0)
        self.dof_vel_scale = self.config.get("dof_vel_scale", 0.1)
        
        # Load motor position limits
        self.legs_lower_limits = np.array(self.config["legs_motor_pos_lower_limit_list"], dtype=np.float32)
        self.legs_upper_limits = np.array(self.config["legs_motor_pos_upper_limit_list"], dtype=np.float32)
        self.torso_lower_limit = self.config["torso_lower_limit"]
        self.torso_upper_limit = self.config["torso_upper_limit"]
        
        # Combine all limits (12 leg joints + 1 torso)
        self.all_lower_limits = np.append(self.legs_lower_limits, self.torso_lower_limit)
        self.all_upper_limits = np.append(self.legs_upper_limits, self.torso_upper_limit)
        
        # Load MuJoCo model
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.simulation_dt
        
        # Load policy if provided
        self.policy = None
        self.use_policy = False
        if policy_path is not None:
            try:
                self.policy = torch.jit.load(policy_path)
                self.use_policy = True
                print(f"Policy loaded: {policy_path}")
            except Exception as e:
                print(f"Failed to load policy: {e}")
                print("Running in PD control only mode")
        
        # State variables
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        # Observation for policy (may be padded to 73 if policy was trained with full model)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        # Full observation if policy expects 73 dims
        self.obs_full = np.zeros(73, dtype=np.float32)  # For padded obs if needed
        self.counter = 0
        
        print(f"\n{'='*60}")
        print(f"Sim2Sim H12 Basic Balance Deployer")
        print(f"{'='*60}")
        print(f"Model: {self.xml_path}")
        print(f"DOFs: {self.m.nq} | Actuators: {self.m.nu}")
        print(f"Control frequency: {1.0 / (self.simulation_dt * self.control_decimation):.0f} Hz")
        print(f"Policy: {'Loaded' if self.use_policy else 'PD Control Only'}")
        print(f"{'='*60}\n")
    
    def build_observation(self):
        """Build observation vector matching Isaac Lab format.
        
        Format: [base_ang_vel (3), projected_gravity (3), 
                 joint_pos_rel (13), joint_vel_rel (13), last_action (13)]
        Total: 45 dims for 13 DOF config
        """
        # Extract only the first 13 controlled joints (legs + torso)
        # d.qpos[0:7] = floating base (x,y,z,qw,qx,qy,qz)
        # d.qpos[7:20] = first 13 controlled joints
        qj = self.d.qpos[7:7+self.num_actions]  # Only first 13 controlled joints
        dqj = self.d.qvel[6:6+self.num_actions]  # Only first 13 controlled joints
        
        # Extract base orientation and angular velocity
        quat = self.d.qpos[3:7]  # Quaternion [w, x, y, z]
        omega = self.d.qvel[3:6]  # Base angular velocity
        
        # Normalize and scale observations
        qj_rel = (qj - self.default_angles) * self.dof_pos_scale
        dqj_scaled = dqj * self.dof_vel_scale
        gravity_orientation = get_gravity_orientation(quat)
        omega_scaled = omega * self.ang_vel_scale
        
        # Assemble observation
        obs_idx = 0
        
        # 1. Base angular velocity (3)
        self.obs[obs_idx:obs_idx+3] = omega_scaled
        obs_idx += 3
        
        # 2. Projected gravity (3)
        self.obs[obs_idx:obs_idx+3] = gravity_orientation
        obs_idx += 3
        
        # 3. Joint positions relative (13)
        self.obs[obs_idx:obs_idx+self.num_actions] = qj_rel
        obs_idx += self.num_actions
        
        # 4. Joint velocities (13)
        self.obs[obs_idx:obs_idx+self.num_actions] = dqj_scaled
        obs_idx += self.num_actions
        
        # 5. Last action (13)
        self.obs[obs_idx:obs_idx+self.num_actions] = self.action
        
        # Pad observation to 73 dims if policy was trained with full model (21 DOF)
        # Format: [base_ang_vel (3), projected_gravity (3), 
        #          joint_pos_rel (21), joint_vel_rel (21), last_action (21)]
        if self.use_policy:
            # Zero out full obs first
            self.obs_full[:] = 0.0
            # Copy over the 45-dim obs to first 45 positions
            self.obs_full[:45] = self.obs
            # Pad remaining 28 dims with zeros (arm joint positions/velocities)
            # self.obs_full[45:] stays zero (padded arm joints)
    
    def get_action(self):
        """Get action from policy or default to standing."""
        if self.use_policy and self.policy is not None:
            # Use padded observation (73 dims) if policy expects it
            obs_tensor = torch.from_numpy(self.obs_full).unsqueeze(0)
            with torch.no_grad():
                action = self.policy(obs_tensor).detach().numpy().squeeze()
            return action
        else:
            # Default: stay at default standing angles
            return np.zeros(self.num_actions, dtype=np.float32)
    
    def run(self):
        """Run the simulation with policy or PD control."""
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            start_time = time.time()
            
            while viewer.is_running() and time.time() - start_time < self.simulation_duration:
                step_start = time.time()
                
                # Apply PD control (only for first 13 actuated joints)
                tau = pd_control(
                    self.target_dof_pos, 
                    self.d.qpos[7:7+self.num_actions],  # Only first 13 joints
                    self.kps, 
                    np.zeros_like(self.kds), 
                    self.d.qvel[6:6+self.num_actions],  # Only first 13 joints
                    self.kds
                )
                
                # Set upper body (arms) to zero angles with PD control
                # Arm DOF indices: qpos[7+13 : 7+13+14] = qpos[20:34]
                #                 qvel[6+13 : 6+13+14] = qvel[19:33]
                arm_start_qpos = 7 + self.num_actions
                arm_start_qvel = 6 + self.num_actions
                
                arm_target_angles = np.zeros(self.num_arm_dofs, dtype=np.float32)
                arm_current_angles = self.d.qpos[arm_start_qpos:arm_start_qpos+self.num_arm_dofs]
                arm_current_vels = self.d.qvel[arm_start_qvel:arm_start_qvel+self.num_arm_dofs]
                
                # PD gains for arms (low stiffness to keep them relaxed)
                arm_kps = np.ones(self.num_arm_dofs) * 50.0   # Low stiffness
                arm_kds = np.ones(self.num_arm_dofs) * 1.5    # Low damping
                
                arm_tau = pd_control(
                    arm_target_angles,
                    arm_current_angles,
                    arm_kps,
                    np.zeros(self.num_arm_dofs),  # No target velocity
                    arm_current_vels,
                    arm_kds
                )
                
                # Combine control torques: [leg+torso (13) + arm (14)] = 27 total
                self.d.ctrl[:self.num_actions] = tau
                self.d.ctrl[self.num_actions:self.num_actions+self.num_arm_dofs] = arm_tau
                
                mujoco.mj_step(self.m, self.d)
                
                self.counter += 1
                
                # Policy inference at control frequency
                if self.counter % self.control_decimation == 0:
                    self.build_observation()
                    action = self.get_action()
                    self.action = action.copy()
                    
                    # Transform action to target positions
                    target_pos = action * self.action_scale + self.default_angles
                    
                    # Clip target positions to valid joint limits
                    self.target_dof_pos = np.clip(target_pos, self.all_lower_limits, self.all_upper_limits)
                
                # Sync viewer
                viewer.sync()
                
                # Time keeping
                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def main():
    parser = argparse.ArgumentParser(
        description="Sim2Sim deployment of H12 Basic Balance policy on MuJoCo"
    )
    parser.add_argument(
        "config_file",
        type=str,
        default="h1_2.yaml",
        nargs="?",
        help="Path to YAML config file (default: h1_2.yaml)"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to TorchScript policy (.pt file). If not provided, runs PD control only."
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    if not Path(args.config_file).exists():
        raise FileNotFoundError(f"Config file not found: {args.config_file}")
    
    if args.policy and not Path(args.policy).exists():
        print(f"⚠️  Policy file not found: {args.policy}")
        print("Running in PD control only mode")
        args.policy = None
    
    # Run deployer
    deployer = H12BasicBalanceDeployer(args.config_file, args.policy)
    deployer.run()


if __name__ == "__main__":
    main()
