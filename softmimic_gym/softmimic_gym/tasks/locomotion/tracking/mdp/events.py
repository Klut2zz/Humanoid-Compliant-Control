
from __future__ import annotations

import torch
import torch.nn.functional as F
import warnings
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

# new change
from softmimic_gym.assets.robots.g1_compat import G1_SOFTMIMIC_29_JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def push_by_adding_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by adding a random velocity within the given ranges to the current root velocity.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and adds them to the current velocity before setting
    them into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]
    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    # add the random velocities to the current velocities
    vel_w += rand_samples
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

def apply_forcefield_force_torque(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Applies reactive force-field forces and torques to the robot.

    This function reads from a command term and applies the computed force-field
    forces and torques to the specified bodies of the robot.

    It is backward-compatible with command generators that expose single-link
    force targets compatible with `ComplianceAugmentedReferenceCommand`.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return

    # -- Setup: Get assets and command term --
    robot: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    num_active_envs = len(env_ids)

    all_body_ids = command_term.sorted_keypoint_ids.to(robot.device)
    num_bodies = len(all_body_ids)

    forces_to_apply = torch.zeros(num_active_envs, num_bodies, 3, device=env.device)
    torques_to_apply = torch.zeros(num_active_envs, num_bodies, 3, device=env.device)
    env_row_indices = torch.arange(num_active_envs, device=env.device)

    forcefield_forces_b = command_term.forcefield_forces_b[env_ids]
    forcefield_torques_b = command_term.forcefield_torques_b[env_ids]
    body_ids = command_term.body_ids[env_ids]

    active_mask = body_ids >= 0
    if torch.any(active_mask):
        active_rows = env_row_indices[active_mask]
        active_body_ids = body_ids[active_mask]
        col_indices = torch.searchsorted(all_body_ids, active_body_ids)
        forces_to_apply[active_rows, col_indices] = forcefield_forces_b[active_mask]
        torques_to_apply[active_rows, col_indices] = forcefield_torques_b[active_mask]

    # -- Apply forces and torques to the simulation --
    robot.set_external_force_and_torque(
        forces=forces_to_apply,
        torques=torques_to_apply,
        body_ids=all_body_ids.tolist(),
        env_ids=env_ids,
    )

def reset_robot_state_by_command(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    jpos_range: dict[str, tuple[float, float]],
    jvel_range: dict[str, tuple[float, float]],
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly offset from the reference command."""
    
    env.command_manager.get_term(command_name).reset_motions(env_ids)

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # get command
    joint_pos_cmd = env.command_manager.get_term(command_name).dof_pos[env_ids]
    joint_vel_cmd = env.command_manager.get_term(command_name).dof_vel[env_ids]
    lin_vel_xy_cmd = env.command_manager.get_term(command_name).root_state["root_vel"][env_ids, 0:2]
    ang_vel_yaw_cmd = env.command_manager.get_term(command_name).root_state["root_ang_vel"][env_ids, 2:3]
    root_pos_cmd = env.command_manager.get_term(command_name).root_state["root_pos"][env_ids, 0:3]
    # root_yaw_cmd = env.command_manager.get_term(command_name).root_state["root_yaw"][env_ids, 0]
    # root_pitch_cmd = env.command_manager.get_term(command_name).root_state["root_pitch"][env_ids, 0]
    root_quat_cmd = env.command_manager.get_term(command_name).root_state["root_rot"][env_ids, 0:4]
    root_vel_global_cmd = env.command_manager.get_term(command_name).root_state["root_vel_global"][env_ids, 0:3]
    root_ang_vel_global_cmd = env.command_manager.get_term(command_name).root_state["root_ang_vel_global"][env_ids, 0:3]

    # joints
    # bias these values randomly
    joint_pos = joint_pos_cmd[:, 0] + math_utils.sample_uniform(*jpos_range, joint_pos_cmd[:,0].shape, joint_pos_cmd.device)
    joint_vel = joint_vel_cmd[:,0] + math_utils.sample_uniform(*jvel_range, joint_pos_cmd[:,0].shape, joint_pos_cmd.device)

    # new change
    # new change
    strict_g1_mapping = joint_pos.shape[-1] == len(G1_SOFTMIMIC_29_JOINT_NAMES)
    if strict_g1_mapping:
        joint_ids, resolved_names = asset.find_joints(
            list(G1_SOFTMIMIC_29_JOINT_NAMES), preserve_order=True
        )
        if list(resolved_names) != list(G1_SOFTMIMIC_29_JOINT_NAMES):
            raise RuntimeError(
                "Reset could not resolve the canonical 29DOF interface: "
                f"resolved={list(resolved_names)}."
            )
        joint_ids = torch.as_tensor(joint_ids, device=asset.device, dtype=torch.long)
    else:
        joint_ids = asset_cfg.joint_ids
    
    # clamp joint pos to limits
    # new change
    joint_pos_limits = (
        asset.data.soft_joint_pos_limits[env_ids[:, None], joint_ids[None, :]]
        if strict_g1_mapping
        else asset.data.soft_joint_pos_limits[env_ids, asset_cfg.joint_ids]
    )
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    # new change
    joint_vel_limits = (
        asset.data.soft_joint_vel_limits[env_ids[:, None], joint_ids[None, :]]
        if strict_g1_mapping
        else asset.data.soft_joint_vel_limits[env_ids]
    )
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    
    
    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    
    # positions = root_pos_cmd + rand_samples[:, 0:3] + env.scene.env_origins[env_ids]
    # positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    # positions[:, 2] = root_pos_cmd[:, 2]+0.075 ## Adjust for G1?
    # positions[:, 2] = root_pos_cmd[:, 2] + rand_samples[:, 2]
    positions = root_pos_cmd[:,0, :3] + rand_samples[:, :3]
    # input(positions[:, 2])
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    # orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], root_pitch_cmd + rand_samples[:, 4], root_yaw_cmd + rand_samples[:, 5])
    # orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], root_pitch_cmd + rand_samples[:, 4], root_yaw_cmd + rand_samples[:, 5])
    # orientations = math_utils.quat_mul(math_utils.quat_from_euler_xyz(torch.zeros_like(rand_samples[:, 3]), root_pitch_cmd, root_yaw_cmd), orientations_delta)
    
    orientations_ref = root_quat_cmd[:,0, [3, 0, 1, 2]] # convert from (x,y,z,w) to (w,x,y,z)
    # orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    orientations = math_utils.quat_mul(orientations_ref, orientations_delta)
    
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    
    velocities = rand_samples
    velocities[:, 0:3] += root_vel_global_cmd[:, 0]
    velocities[:, 3:6] += root_ang_vel_global_cmd[:, 0]
    # velocities[:, 0:2] += lin_vel_xy_cmd # TODO apply after verifying global frame

    # print(positions, orientations, velocities, joint_pos, joint_vel)
    # input()
    # print("reset")
    # input(joint_pos)

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    # new change
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)
    asset.set_joint_position_target(joint_pos, joint_ids=joint_ids, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, joint_ids=joint_ids, env_ids=env_ids)

def randomize_rigid_body_com(
    env: ManagerBasedEnv, 
    env_ids: torch.Tensor | None,
    max_displacement: float,
    asset_cfg: SceneEntityCfg,
):
    """Randomize the CoM of the bodies by adding a random value sampled from the given range.

    .. tip::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()[:, body_ids, :3]

    # Randomize the com in range -max displacement to max displacement
    coms += torch.rand_like(coms) * 2 * max_displacement - max_displacement

    # Set the new coms
    new_coms = asset.root_physx_view.get_coms().clone()
    new_coms[:, asset_cfg.body_ids, 0:3] = coms
    asset.root_physx_view.set_coms(new_coms, env_ids)


# new change
def write_online_reference_kinematic_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Visual-gate only: write the accepted q_ref directly like Mink viewer."""

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)
    joint_ids = torch.as_tensor(
        command.control_joint_ids, device=asset.device, dtype=torch.long
    )
    joint_pos = command.dof_pos[env_ids, 1]
    joint_vel = command.dof_vel[env_ids, 1]
    asset.write_joint_state_to_sim(
        joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids
    )
    asset.set_joint_position_target(
        joint_pos, joint_ids=joint_ids, env_ids=env_ids
    )
    asset.set_joint_velocity_target(
        joint_vel, joint_ids=joint_ids, env_ids=env_ids
    )
