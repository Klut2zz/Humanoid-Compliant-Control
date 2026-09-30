# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm


def is_full_slice(s):
    return (isinstance(s, slice) and 
            s.start is None and 
            s.stop is None and 
            s.step is None)


# new change
def bad_reference_orientation(
    env,
    command_name: str,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate on orientation error relative to the motion, not world upright.

    Motions such as box-pick deliberately lean the pelvis by more than 45
    degrees.  An absolute upright termination would label a correct reference
    pose as failure and prevent a general controller from learning it.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    current_gravity = asset.data.projected_gravity_b
    reference_gravity = env.command_manager.get_term(command_name).root_state[
        "projected_gravity_b"
    ][:, 1]
    current_gravity = torch.nn.functional.normalize(current_gravity, dim=-1)
    reference_gravity = torch.nn.functional.normalize(reference_gravity, dim=-1)
    cosine = torch.sum(current_gravity * reference_gravity, dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cosine).abs() > limit_angle


# new change
def stationary_support_containment(
    env,
    command_name: str,
    reference_keypoint_ids: list[int],
    root_limit_m: float,
    foot_limit_m: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    support_foot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate stationary rollouts after material root or stance translation.

    COM remaining between two translating feet is not successful stationary
    support.  Compare simulator-world root/feet with the immutable nominal
    reference so horizontal skating cannot continue collecting alive and
    orientation reward.
    """

    if root_limit_m <= 0.0 or foot_limit_m <= 0.0:
        raise ValueError("Stationary containment limits must be positive.")
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    root_reference_xy = command.original_root_state["root_pos"][:, 1, :2]
    root_error = torch.linalg.vector_norm(
        asset.data.root_pos_w[:, :2] - root_reference_xy, dim=-1
    )
    reference_ids = torch.as_tensor(
        reference_keypoint_ids, device=asset.device, dtype=torch.long
    )
    current_foot_xy = asset.data.body_pos_w[:, support_foot_cfg.body_ids, :2]
    reference_foot_xy = command.original_keypoints[:, 1].index_select(
        1, reference_ids
    )[..., :2]
    if current_foot_xy.shape != reference_foot_xy.shape:
        raise ValueError(
            "Stationary containment foot mapping mismatch: "
            f"current={tuple(current_foot_xy.shape)}, "
            f"reference={tuple(reference_foot_xy.shape)}."
        )
    foot_error = torch.linalg.vector_norm(
        current_foot_xy - reference_foot_xy, dim=-1
    ).amax(dim=1)
    return (root_error > root_limit_m) | (foot_error > foot_limit_m)

"""
Root terminations.
"""

def bad_keypoint_deviation(env, threshold: float, command_name: str, keypoint_body_ids: list[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the commanded joint position."""
    asset: Articulation = env.scene[asset_cfg.name]
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    commanded_joint_pos = env.command_manager.get_command(command_name)[:, :asset.num_joints][:, asset_cfg.joint_ids]
    
    current_root_pos = asset.data.root_pos_w[:, :3].clone()
    current_root_pos[:, :2] = 0.0 # ignore x and y position
    current_root_rot = asset.data.root_quat_w[:, :4].clone()
    current_root_rot = current_root_rot[:, [1, 2, 3, 0]]  # convert to xyzw
    
    commanded_root_pos = env.command_manager.get_term(command_name).root_state["root_pos"][:, :3].clone()
    commanded_root_pos[:, :2] = 0.0 # ignore x and y position
    commanded_root_rot = env.command_manager.get_term(command_name).root_state["root_rot"][:, :4].clone()
    commanded_keypoints = env.command_manager.get_term(command_name).keypoints
    # current_keypoints = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]  # Get the current keypoints from the asset's body positions
    current_keypoints = asset.data.body_pos_w[:, keypoint_body_ids, :3]  # Get the current keypoints from the asset's body positions

    # terminate if any keypoint deviates more than 0.5m
    deviation = torch.norm(current_keypoints - commanded_keypoints, dim=-1)
    return torch.any(deviation > threshold, dim=-1).to(asset.data.joint_pos.device)



def bad_keypoint_deviation_local(env, threshold: float, adaptive: bool,  command_name: str, keypoint_body_ids: list[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the commanded joint position."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    current_root_pos = asset.data.root_pos_w[:, :3].clone()
    current_root_pos[:, 2] = 0.0 # ignore z position
    current_root_rot = asset.data.root_quat_w[:, :4].clone()
    # current_root_rot = current_root_rot[:, x[1, 2, 3, 0]]  # convert to xyzw
    _, _, current_yaw = math_utils.euler_xyz_from_quat(current_root_rot)
    
    commanded_root_pos = env.command_manager.get_term(command_name).root_state["root_pos"][:,0, :3].clone()
    commanded_root_pos[:, 2] = 0.0 # ignore z position
    commanded_root_rot = env.command_manager.get_term(command_name).root_state["root_rot"][:,0, :4].clone()
    commanded_root_rot = commanded_root_rot[:, [3, 0, 1, 2]]  # convert to wxyz
    _, _, commanded_yaw = math_utils.euler_xyz_from_quat(commanded_root_rot)

    yaw_diff = math_utils.wrap_to_pi(current_yaw - commanded_yaw)
    yaw_rotation = math_utils.quat_from_euler_xyz(torch.zeros_like(current_yaw), torch.zeros_like(current_yaw), yaw_diff)
    
    # TODO(gmargo): Why are the ankles swapped??
    # hardcoded_keypoint_body_ids = [
    #     0, # pelvis
    #     9, # left_hip_yaw_link
    #     12, # left_knee_link
    #     26, # right_ankle_roll_link
    #     10, # right_hip_yaw_link
    #     13, # right_knee_link
    #     23, # left_ankle_roll_link
    #     11, # torso_link
    #     28, # left_shoulder_yaw_link
    #     30, # left_elbow_link
    #     36, # left_wrist_yaw_link
    #     29, # right_shoulder_yaw_link
    #     31, # right_elbow_link
    #     37, # right_wrist_yaw_link
    # ]

    commanded_keypoints = env.command_manager.get_term(command_name).keypoints
    relative_timestamp = env.command_manager.get_term(command_name).relative_timestamp
    # current_keypoints = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]  # Get the current keypoints from the asset's body positions
    current_keypoints = asset.data.body_pos_w[:, keypoint_body_ids, :3]  # Get the current keypoints from the asset's body positions
    
    commanded_keypoints_local = commanded_keypoints[:, 0] - commanded_root_pos.unsqueeze(1)
    commanded_keypoints_local = math_utils.quat_rotate(
        yaw_rotation.unsqueeze(1).repeat(1, commanded_keypoints.shape[2], 1),
        commanded_keypoints_local,
    )
    current_keypoints_local = current_keypoints - current_root_pos.unsqueeze(1)

    # print(f"Current keypoints: {current_keypoints_local[0]}")
    # print(f"Commanded keypoints: {commanded_keypoints_local[0]}")
    # input()

    # terminate if any keypoint deviates more than 0.5m
    deviation = torch.norm(current_keypoints_local - commanded_keypoints_local, dim=-1)
    if adaptive:
        threshold_arr = torch.full(relative_timestamp.shape, threshold, device = asset.data.joint_pos.device)
        mod_threshold = torch.max(threshold_arr, -relative_timestamp/4+1).unsqueeze(1).expand(-1, deviation.shape[1])
        termination = torch.any(deviation > mod_threshold, dim=-1).to(asset.data.joint_pos.device)
    else:
        termination = torch.any(deviation > threshold, dim=-1).to(asset.data.joint_pos.device)
    # print(termination)
    # print(threshold)
    # print(adaptive)
    # input()
    return termination

def bad_force_link_deviation_local(
    env: ManagerBasedRLEnv, threshold: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)

    current_root_pos = asset.data.root_pos_w[:, :3].clone()
    current_root_pos[:, 2] = 0.0 # ignore z position
    current_root_rot = asset.data.root_quat_w[:, :4].clone()
    # current_root_rot = current_root_rot[:, [1, 2, 3, 0]]  # convert to xyzw
    _, _, current_yaw = math_utils.euler_xyz_from_quat(current_root_rot)
    
    commanded_root_pos = env.command_manager.get_term(command_name).adapted_root_state["root_pos"][:,1,:3].clone()
    commanded_root_pos[:, 2] = 0.0 # ignore z position
    commanded_root_rot = env.command_manager.get_term(command_name).adapted_root_state["root_rot"][:,1,:4].clone()
    commanded_root_rot = commanded_root_rot[:, [3, 0, 1, 2]]  # convert to wxyz
    _, _, commanded_yaw = math_utils.euler_xyz_from_quat(commanded_root_rot)
    
    yaw_diff = math_utils.wrap_to_pi(current_yaw - commanded_yaw)
    yaw_rotation = math_utils.quat_from_euler_xyz(
        torch.zeros_like(current_yaw), 
        torch.zeros_like(current_yaw), 
        yaw_diff
    )
    
    keypoint_body_ids_tensor = command_term.keypoint_body_ids_tensor
    if keypoint_body_ids_tensor is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=command_term.device)
    force_link_id = command_term.force_link_id[:, 1, 0].clone()
    force_link_id[force_link_id < 0] = 0  # No force link applied, set to 0
    force_link_idxs = (keypoint_body_ids_tensor.unsqueeze(0) == force_link_id.unsqueeze(1)).nonzero()[:, 1]
    env_ids = torch.arange(env.num_envs, device=command_term.device)

    adapted_keypoints = command_term.adapted_keypoints[env_ids, 1]
    current_keypoints = env.scene["robot"].data.body_pos_w[:, keypoint_body_ids_tensor, :3]
    
    adapted_keypoints_local = adapted_keypoints - commanded_root_pos.unsqueeze(1)
    current_keypoints_local = current_keypoints - current_root_pos.unsqueeze(1)
    
    adapted_keypoints_local = math_utils.quat_rotate(
        yaw_rotation.unsqueeze(1).repeat(1, adapted_keypoints.shape[1], 1),
        adapted_keypoints_local,
    )

    adapted_keypoint_error = torch.norm(adapted_keypoints_local - current_keypoints_local, dim=-1)

    # print(adapted_keypoints_local - current_keypoints_local)

    force_link_error = adapted_keypoint_error[env_ids, force_link_idxs]
    # force_link_error[command_term.force_link_id[:, 1, 0] < 0] = 0.0  # No error if no force link is applied
    force_link_error[~command_term.active_force_mask] = 0.0  # No error if no force link is applied
    
    termination = force_link_error > threshold

    return termination.to(asset.data.joint_pos.device)


def bad_force_link_and_keypoint_deviation_local(env, threshold: float, adaptive: bool,  command_name: str, keypoint_body_ids: list[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the commanded joint position."""
    
    force_link_termination = bad_force_link_deviation_local(env, threshold, command_name, asset_cfg)
    keypoint_termination = bad_keypoint_deviation_local(env, threshold, adaptive, command_name, keypoint_body_ids, asset_cfg)
    
    return torch.logical_and(force_link_termination, keypoint_termination)
    
