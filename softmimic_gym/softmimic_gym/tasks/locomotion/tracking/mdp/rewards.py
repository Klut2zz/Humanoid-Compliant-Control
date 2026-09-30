from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import torch.nn.functional as F
from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject

# new change
from .keypoints import resolve_keypoint_body_ids

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_GRAVITY_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _gravity_vector(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    cached = _GRAVITY_CACHE.get(key)
    if cached is None:
        cached = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype)
        _GRAVITY_CACHE[key] = cached
    return cached

def is_full_slice(s):
    return (isinstance(s, slice) and 
            s.start is None and 
            s.stop is None and 
            s.step is None)

def target_orientation_exp(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using exponential kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # gravity_vec_base = asset.data.projected_gravity_b
    if asset_cfg.body_ids is None or is_full_slice(asset_cfg.body_ids):
        # If no body_ids are specified, use the root body
        projected_gravity_b = asset.data.projected_gravity_b
    else:
        body_id = asset_cfg.body_ids[0] # just use the first body id for now
        asset = env.scene[asset_cfg.name]
        # body_states_w = asset.data.body_state_w[:, body_id]
        body_quat_w = asset.data.body_quat_w[:, body_id]
        gravity_vec_w = _gravity_vector(body_quat_w.device, body_quat_w.dtype).expand(body_quat_w.shape[0], -1)
        projected_gravity_b = math_utils.quat_rotate_inverse(body_quat_w, gravity_vec_w)

    # target_pitch = env.command_manager.get_term(command_name).root_state["root_pitch"][:, 0]
    # target_roll = env.command_manager.get_term(command_name).root_state["root_roll"][:, 0]
    # eulers = torch.stack((target_roll, target_pitch, torch.zeros_like(target_pitch)), dim=1)
    # target_rotation_matrix = math_utils.matrix_from_euler(eulers, convention="XYZ")
    # gravity_vec_rollpitched = torch.bmm(target_rotation_matrix, projected_gravity_b.unsqueeze(-1)).squeeze(-1)
    # error = torch.norm(gravity_vec_rollpitched[:, :2], dim=1)

    target_gravity_b = env.command_manager.get_term(command_name).root_state["projected_gravity_b"]
    # print(target_gravity_b, projected_gravity_b)
    dot_product = torch.sum(projected_gravity_b * target_gravity_b[:,1], dim=-1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Clamp to avoid NaN from acos
    angle = torch.acos(dot_product).abs()
    error = torch.sin(angle)  # Use sine of the angle for a smoother penalty

    rew_gravity = torch.exp(-1 * error)
    return rew_gravity

def joint_deviation_from_command_exp(env, command_name: str, sigma: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the commanded joint position."""
    asset: Articulation = env.scene[asset_cfg.name]
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # commanded_joint_pos = env.command_manager.get_command(command_name)[:, :asset.num_joints][:, asset_cfg.joint_ids]
    # new change
    reference_joint_pos = env.command_manager.get_term(command_name).dof_pos[:, 1]
    commanded_joint_pos = (
        reference_joint_pos
        if reference_joint_pos.shape[-1] == current_joint_pos.shape[-1]
        else reference_joint_pos[:, asset_cfg.joint_ids]
    )
    angle = current_joint_pos - commanded_joint_pos
    
    # print("rwd")
    # print(commanded_joint_pos)
    # print(current_joint_pos)
    # input()

    rew_angle = torch.exp(-torch.norm(angle, dim=1) / sigma**2)
    return rew_angle


# new change
def joint_subset_deviation_from_command_mse_exp(
    env,
    command_name: str,
    reference_joint_ids: list[int],
    sigma: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint-count-normalized position tracking for an explicit mapping."""

    asset: Articulation = env.scene[asset_cfg.name]
    current = asset.data.joint_pos[:, asset_cfg.joint_ids]
    reference = env.command_manager.get_term(command_name).dof_pos[:, 1]
    reference_ids = torch.as_tensor(reference_joint_ids, device=reference.device, dtype=torch.long)
    desired = reference.index_select(1, reference_ids)
    if current.shape != desired.shape:
        raise ValueError(f"Joint position mapping mismatch: q={current.shape}, ref={desired.shape}.")
    mse = torch.mean(torch.square(current - desired), dim=1)
    return torch.exp(-mse / sigma**2)


# new change
def joint_subset_velocity_deviation_from_command_mse_exp(
    env,
    command_name: str,
    reference_joint_ids: list[int],
    sigma: float = 0.50,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Velocity tracking for an explicit articulation/reference mapping."""

    asset: Articulation = env.scene[asset_cfg.name]
    current = asset.data.joint_vel[:, asset_cfg.joint_ids]
    reference = env.command_manager.get_term(command_name).dof_vel[:, 1]
    reference_ids = torch.as_tensor(reference_joint_ids, device=reference.device, dtype=torch.long)
    desired = reference.index_select(1, reference_ids)
    if current.shape != desired.shape:
        raise ValueError(f"Joint velocity mapping mismatch: dq={current.shape}, ref={desired.shape}.")
    mse = torch.mean(torch.square(current - desired), dim=1)
    return torch.exp(-mse / sigma**2)


# new change
def whole_body_com_support_center_exp(
    env,
    sigma: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    support_foot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
    ),
) -> torch.Tensor:
    """Reward horizontal whole-body COM proximity to the double-support center.

    StandCoordination references keep both feet planted.  The geometric
    midpoint is intentionally used instead of the legacy contact attribution,
    which the official G1 USD can report on the knee links.
    """

    if sigma <= 0.0:
        raise ValueError("COM support sigma must be positive.")
    asset: Articulation = env.scene[asset_cfg.name]
    body_com = asset.data.body_com_pos_w
    # new change
    # IsaacLab keeps default articulation properties on CPU while live body
    # state is on the simulation device.
    masses = asset.data.default_mass.to(
        device=body_com.device, dtype=body_com.dtype
    )
    whole_body_com_xy = (
        body_com[..., :2] * masses.unsqueeze(-1)
    ).sum(dim=1) / masses.sum(dim=1, keepdim=True)
    support_xy = body_com[:, support_foot_cfg.body_ids, :2].mean(dim=1)
    distance_sq = torch.sum(torch.square(whole_body_com_xy - support_xy), dim=1)
    return torch.exp(-distance_sq / sigma**2)

def joint_deviation_from_command_velocities_exp(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities that deviate from the commanded joint velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    current_joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    commanded_joint_vel = env.command_manager.get_term(command_name).dof_vel[:,1, asset_cfg.joint_ids]
    angle = current_joint_vel - commanded_joint_vel

    rew_angle = torch.exp(-torch.norm(angle, dim=1) / 10.0)
    return rew_angle

def joint_deviation_from_command_contacts(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint positions that deviate from the commanded joint position."""
    # extract the used quantities (to enable type-hinting)

    desired_contacts = env.command_manager.get_term(command_name).foot_contacts[:,1]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    
    contacts_match = torch.sum(contacts == desired_contacts, dim=1).float()
    rew_contact = 2 - contacts_match

    # desire_swing_but_make_contact = torch.sum(contacts * (1 - desired_contacts), dim=1).float()
    # rew_contact = desire_swing_but_make_contact

    return rew_contact

def joint_deviation_from_command_contacts_prob(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    
    desired_contacts_prob = env.command_manager.get_term(command_name).foot_contacts[:,1]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts_bool = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    contacts = contacts_bool.float()

    squared_error = torch.square(contacts - desired_contacts_prob)

    rew_contact = torch.sum(squared_error, dim=1)

    return rew_contact

def keypoint_deviation_from_command(env, command_name: str, keypoint_body_ids: list[int | str],  sigma: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the commanded joint position."""
    asset: Articulation = env.scene[asset_cfg.name]
    # new change
    keypoint_body_ids = resolve_keypoint_body_ids(asset, keypoint_body_ids)
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # commanded_joint_pos = env.command_manager.get_command(command_name)[:, :asset.num_joints][:, asset_cfg.joint_ids]
    commanded_joint_pos = env.command_manager.get_term(command_name).dof_pos[:,1, asset_cfg.joint_ids]
    
    current_root_pos = asset.data.root_pos_w[:, :3].clone()
    current_root_pos[:, :2] = 0.0 # ignore x and y position
    current_root_rot = asset.data.root_quat_w[:, :4].clone()
    current_root_rot = current_root_rot[:, [1, 2, 3, 0]]  # convert to xyzw
    
    commanded_root_pos = env.command_manager.get_term(command_name).root_state["root_pos"][:,1,:3].clone()
    commanded_root_pos[:, :2] = 0.0 # ignore x and y position
    commanded_root_rot = env.command_manager.get_term(command_name).root_state["root_rot"][:,1,:4]
    
    commanded_keypoints = env.command_manager.get_term(command_name).keypoints[:,1]
    # current_keypoints = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]  # Get the current keypoints from the asset's body positions
    current_keypoints = asset.data.body_pos_w[:, keypoint_body_ids, :3]  # Get the current keypoints from the asset's body positions
    
    l2_error = torch.norm(current_keypoints - commanded_keypoints, dim=(1, 2)).to(asset.device)
    return torch.exp(-l2_error / sigma**2)
    # norm_keypoint_diff = torch.norm(torch.tensor(current_keypoints - commanded_keypoints, device=asset.device), dim=2)
    # mean_error = torch.mean(norm_keypoint_diff, dim=1)
    # return torch.exp(-mean_error)
    

def keypoint_deviation_from_command_local(env, command_name: str, keypoint_body_ids: list[int | str], sigma: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the commanded joint position."""
    asset: Articulation = env.scene[asset_cfg.name]
    # new change
    keypoint_body_ids = resolve_keypoint_body_ids(asset, keypoint_body_ids)
    
    current_root_pos = asset.data.root_pos_w[:, :3].clone()
    current_root_pos[:, 2] = 0.0 # ignore z position
    current_root_rot = asset.data.root_quat_w[:, :4].clone()
    # current_root_rot = current_root_rot[:, [1, 2, 3, 0]]  # convert to xyzw
    _, _, current_yaw = math_utils.euler_xyz_from_quat(current_root_rot)
    
    commanded_root_pos = env.command_manager.get_term(command_name).root_state["root_pos"][:,1,:3].clone()
    commanded_root_pos[:, 2] = 0.0 # ignore z position
    commanded_root_rot = env.command_manager.get_term(command_name).root_state["root_rot"][:,1,:4].clone()
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

    commanded_keypoints = env.command_manager.get_term(command_name).keypoints[:,1]
    # current_keypoints = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]  # Get the current keypoints from the asset's body positions
    current_keypoints = asset.data.body_pos_w[:, keypoint_body_ids, :3]  # Get the current keypoints from the asset's body positions
    
    commanded_keypoints_local = commanded_keypoints - commanded_root_pos.unsqueeze(1)
    current_keypoints_local = current_keypoints - current_root_pos.unsqueeze(1)
    # Rotate the commanded keypoints to the local frame
    commanded_keypoints_local = math_utils.quat_rotate(
        yaw_rotation.unsqueeze(1).repeat(1, commanded_keypoints.shape[1], 1),
        commanded_keypoints_local,
    )
    
    # input(keypoint_body_ids)
    # print(commanded_keypoints_local - current_keypoints_local)
    # print(asset.body_names[keypoint_body_ids])
    # for i, id in enumerate(keypoint_body_ids):
    #     print(f"Keypoint {asset.body_names[id]}: {commanded_keypoints_local[:, i]} vs {current_keypoints_local[:, i]}")
    # input()
    
    
    # l2_error = torch.norm(current_keypoints_local - commanded_keypoints_local, dim=(1, 2)).to(asset.device)
    # rwd = torch.exp(-l2_error / sigma**2)
    error = torch.sum(torch.square(current_keypoints_local - commanded_keypoints_local), dim=-1)  # L2 error
    rwd = torch.exp(-error.mean(-1) / sigma**2)  # Apply the exponential kernel to the error
    
    # print(torch.exp(-l2_error / sigma**2))

    return rwd


# new change
def keypoint_deviation_from_command_local_mse_exp(
    env,
    command_name: str,
    keypoint_body_ids: list[int | str],
    sigma: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Root/yaw-aligned keypoint reward normalized by keypoint count."""

    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = resolve_keypoint_body_ids(asset, keypoint_body_ids)
    command = env.command_manager.get_term(command_name)
    current_keypoints = asset.data.body_pos_w[:, body_ids, :3]
    desired_keypoints = command.keypoints[:, 1]
    if current_keypoints.shape != desired_keypoints.shape:
        raise ValueError(
            "Keypoint mapping mismatch: "
            f"current={current_keypoints.shape}, reference={desired_keypoints.shape}."
        )

    current_root = asset.data.root_pos_w[:, :3].clone()
    current_root[:, 2] = 0.0
    _, _, current_yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w[:, :4])

    desired_root = command.root_state["root_pos"][:, 1, :3].clone()
    desired_root[:, 2] = 0.0
    desired_quat_wxyz = command.root_state["root_rot"][:, 1, [3, 0, 1, 2]]
    _, _, desired_yaw = math_utils.euler_xyz_from_quat(desired_quat_wxyz)
    yaw_delta = math_utils.wrap_to_pi(current_yaw - desired_yaw)
    yaw_rotation = math_utils.quat_from_euler_xyz(
        torch.zeros_like(yaw_delta), torch.zeros_like(yaw_delta), yaw_delta
    )

    desired_local = desired_keypoints - desired_root.unsqueeze(1)
    desired_local = math_utils.quat_rotate(
        yaw_rotation.unsqueeze(1).expand(-1, desired_local.shape[1], -1),
        desired_local,
    )
    current_local = current_keypoints - current_root.unsqueeze(1)
    mse = torch.mean(torch.sum(torch.square(current_local - desired_local), dim=-1), dim=1)
    return torch.exp(-mse / sigma**2)


# new change
def keypoint_subset_deviation_from_command_local_mse_exp(
    env,
    command_name: str,
    keypoint_body_ids: list[int | str],
    reference_keypoint_ids: list[int],
    sigma: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Root/yaw-aligned tracking reward for an explicit keypoint subset.

    Articulation body ids and reference-motion keypoint ids are deliberately
    independent.  This prevents torso/arm coordination credit from being
    diluted by the six lower-body keypoints in stationary manipulation tasks.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = resolve_keypoint_body_ids(asset, keypoint_body_ids)
    command = env.command_manager.get_term(command_name)
    reference_ids = torch.as_tensor(
        reference_keypoint_ids, device=asset.device, dtype=torch.long
    )
    desired_keypoints = command.keypoints[:, 1].index_select(1, reference_ids)
    current_keypoints = asset.data.body_pos_w[:, body_ids, :3]
    if current_keypoints.shape != desired_keypoints.shape:
        raise ValueError(
            "Keypoint subset mapping mismatch: "
            f"current={tuple(current_keypoints.shape)}, "
            f"reference={tuple(desired_keypoints.shape)}."
        )

    current_root = asset.data.root_pos_w[:, :3].clone()
    current_root[:, 2] = 0.0
    _, _, current_yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w[:, :4])

    desired_root = command.root_state["root_pos"][:, 1, :3].clone()
    desired_root[:, 2] = 0.0
    desired_quat_wxyz = command.root_state["root_rot"][:, 1, [3, 0, 1, 2]]
    _, _, desired_yaw = math_utils.euler_xyz_from_quat(desired_quat_wxyz)
    yaw_delta = math_utils.wrap_to_pi(current_yaw - desired_yaw)
    yaw_rotation = math_utils.quat_from_euler_xyz(
        torch.zeros_like(yaw_delta), torch.zeros_like(yaw_delta), yaw_delta
    )

    desired_local = desired_keypoints - desired_root.unsqueeze(1)
    desired_local = math_utils.quat_rotate(
        yaw_rotation.unsqueeze(1).expand(-1, desired_local.shape[1], -1),
        desired_local,
    )
    current_local = current_keypoints - current_root.unsqueeze(1)
    mse = torch.mean(torch.sum(torch.square(current_local - desired_local), dim=-1), dim=1)
    return torch.exp(-mse / sigma**2)

def keypoint_orientation_deviation_from_command_local(env, command_name: str, keypoint_body_ids: list[int | str], sigma: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the commanded joint position."""
    asset: Articulation = env.scene[asset_cfg.name]
    # new change
    keypoint_body_ids = resolve_keypoint_body_ids(asset, keypoint_body_ids)
    
    # current_root_pos = asset.data.root_pos_w[:, :3].clone()
    # current_root_pos[:, 2] = 0.0 # ignore z position
    current_root_rot = asset.data.root_quat_w[:, :4].clone()
    _, _, current_yaw = math_utils.euler_xyz_from_quat(current_root_rot)
    
    # commanded_root_pos = env.command_manager.get_term(command_name).root_state["root_pos"][:,1,:3].clone()
    # commanded_root_pos[:, 2] = 0.0 # ignore z position
    commanded_root_rot = env.command_manager.get_term(command_name).root_state["root_rot"][:,1,:4].clone()
    commanded_root_rot = commanded_root_rot[:, [3, 0, 1, 2]]  # convert to wxyz
    _, _, commanded_yaw = math_utils.euler_xyz_from_quat(commanded_root_rot)
    
    yaw_diff = math_utils.wrap_to_pi(current_yaw - commanded_yaw)
    yaw_rotation = math_utils.quat_from_euler_xyz(torch.zeros_like(current_yaw), torch.zeros_like(current_yaw), yaw_diff)

    commanded_keypoint_quats = env.command_manager.get_term(command_name).keypoint_rotations[:,1] # (N, 14, 4)
    commanded_keypoint_quats = commanded_keypoint_quats#[:, :, [3, 0, 1, 2]]  # convert to wxyz
    current_keypoint_quats = asset.data.body_quat_w[:, keypoint_body_ids, :4]  # 

    # Transform the commanded keypoints into the local frame
    # print(yaw_rotation.shape, yaw_rotation.unsqueeze(1).repeat(1, commanded_keypoint_quats.shape[1], 1).shape, commanded_keypoint_quats.shape)
    # input()
    commanded_keypoint_quats_local = math_utils.quat_mul(
        yaw_rotation.unsqueeze(1).repeat(1, commanded_keypoint_quats.shape[1], 1),
        commanded_keypoint_quats,
    )
    
    # Compute the rotation error between the commanded and current keypoints

    error = math_utils.quat_error_magnitude(
        commanded_keypoint_quats_local,
        current_keypoint_quats,
     ) ** 2
    
    # print(error)

    return torch.exp(-error.mean(-1) / sigma**2)  # Apply the exponential kernel to the error

def feet_slide_proportional(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]# > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contact_forces, dim=1)
    return reward


# new change
def stationary_root_xy_velocity_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared horizontal root speed for stationary controllers."""

    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)


# new change
def stationary_root_xy_position_error_l2(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared root XY drift from the current motion reference."""

    asset = env.scene[asset_cfg.name]
    reference_xy = env.command_manager.get_term(command_name).root_state[
        "root_pos"
    ][:, 1, :2]
    return torch.sum(
        torch.square(asset.data.root_pos_w[:, :2] - reference_xy), dim=1
    )


# new change
def stationary_root_xy_position_error_bounded_huber(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    delta_m: float = 0.10,
) -> torch.Tensor:
    """Bounded root-anchor loss that cannot dominate long-rollout critic targets."""

    if delta_m <= 0.0:
        raise ValueError(f"delta_m must be positive, got {delta_m}.")
    asset = env.scene[asset_cfg.name]
    reference_xy = env.command_manager.get_term(command_name).root_state[
        "root_pos"
    ][:, 1, :2]
    normalized_error = torch.abs(
        asset.data.root_pos_w[:, :2] - reference_xy
    ) / delta_m
    huber = torch.where(
        normalized_error <= 1.0,
        0.5 * torch.square(normalized_error),
        normalized_error - 0.5,
    )
    return torch.tanh(torch.sum(huber, dim=1))


# new change
def stationary_root_xy_velocity_bounded_huber(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    delta_mps: float = 0.15,
) -> torch.Tensor:
    """Bounded horizontal root-speed loss for stable long rollouts."""

    if delta_mps <= 0.0:
        raise ValueError(f"delta_mps must be positive, got {delta_mps}.")
    asset = env.scene[asset_cfg.name]
    normalized_speed = torch.abs(asset.data.root_lin_vel_b[:, :2]) / delta_mps
    huber = torch.where(
        normalized_speed <= 1.0,
        0.5 * torch.square(normalized_speed),
        normalized_speed - 0.5,
    )
    return torch.tanh(torch.sum(huber, dim=1))


# new change
def fixed_foot_xy_position_error_l2(
    env,
    command_name: str,
    reference_keypoint_ids: list[int],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Squared world-XY error of both support feet to their fixed references."""

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    reference_ids = torch.as_tensor(
        reference_keypoint_ids,
        device=asset.device,
        dtype=torch.long,
    )
    current_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    reference_xy = command.keypoints[:, 1].index_select(1, reference_ids)[..., :2]
    if current_xy.shape != reference_xy.shape:
        raise ValueError(
            "Fixed-foot reward mapping mismatch: "
            f"current={current_xy.shape}, reference={reference_xy.shape}."
        )
    return torch.sum(torch.square(current_xy - reference_xy), dim=(1, 2))


# new change
def fixed_foot_xy_position_error_bounded_huber(
    env,
    command_name: str,
    reference_keypoint_ids: list[int],
    asset_cfg: SceneEntityCfg,
    delta_m: float = 0.10,
) -> torch.Tensor:
    """Bounded Huber loss for fixed-world bilateral foot XY anchors.

    The old squared loss grows without limit once a long rollout drifts away
    from its anchors and can dominate the critic target.  This term preserves
    the quadratic small-error basin and linear Huber transition, then bounds
    the per-environment result to [0, 1) with tanh.
    """

    if delta_m <= 0.0:
        raise ValueError(f"delta_m must be positive, got {delta_m}.")
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    reference_ids = torch.as_tensor(
        reference_keypoint_ids,
        device=asset.device,
        dtype=torch.long,
    )
    current_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    reference_xy = command.keypoints[:, 1].index_select(1, reference_ids)[..., :2]
    if current_xy.shape != reference_xy.shape:
        raise ValueError(
            "Fixed-foot bounded-Huber mapping mismatch: "
            f"current={current_xy.shape}, reference={reference_xy.shape}."
        )
    normalized_error = torch.abs(current_xy - reference_xy) / delta_m
    huber = torch.where(
        normalized_error <= 1.0,
        0.5 * torch.square(normalized_error),
        normalized_error - 0.5,
    )
    return torch.tanh(torch.sum(huber, dim=(1, 2)))


# new change
def resolved_double_support(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold_n: float = 20.0,
) -> torch.Tensor:
    """One when both empirically resolved G1 support channels are loaded."""

    contact_sensor = env.scene.sensors[sensor_cfg.name]
    vertical_force = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2]
        .abs()
        .max(dim=1)
        .values
    )
    return (vertical_force >= threshold_n).all(dim=1).to(torch.float32)


# new change
def resolved_support_force_margin(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold_n: float = 20.0,
    transition_n: float = 20.0,
) -> torch.Tensor:
    """Smooth bounded reward for retaining non-zero load on both support feet."""

    if transition_n <= 0.0:
        raise ValueError(
            f"transition_n must be positive, got {transition_n}."
        )
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    vertical_force = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2]
        .abs()
        .max(dim=1)
        .values
    )
    minimum_force = vertical_force.min(dim=1).values
    return torch.sigmoid((minimum_force - threshold_n) / transition_n)


def track_lin_vel_global(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    cmd_vel = env.command_manager.get_term(command_name).root_state["root_vel_global"]

    if asset_cfg.body_ids is None or is_full_slice(asset_cfg.body_ids):
        # If no body_ids are specified, use the root body
        body_lin_vel_w = asset.data.root_lin_vel_w[:, :2]
    else:
        body_id = asset_cfg.body_ids[0] # just use the first body id for now
        asset = env.scene[asset_cfg.name]
        # body_states_w = asset.data.body_state_w[:, body_id]
        body_quat_w = asset.data.body_quat_w[:, body_id]
        body_lin_vel_w = asset.data.body_lin_vel_w[:, body_id, :3]

    lin_vel_error = torch.sum(
        torch.square(cmd_vel - body_lin_vel_w),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)

def track_lin_vel_local(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    cmd_vel = env.command_manager.get_term(command_name).root_state["root_vel"]

    if asset_cfg.body_ids is None or is_full_slice(asset_cfg.body_ids):
        # If no body_ids are specified, use the root body
        body_lin_vel_b = asset.data.root_lin_vel_b[:, :2]
    else:
        body_id = asset_cfg.body_ids[0] # just use the first body id for now
        asset = env.scene[asset_cfg.name]
        # body_states_w = asset.data.body_state_w[:, body_id]
        # body_lin_vel_b = asset.data.body_lin_vel_b[:, body_id, :3]
        body_quat_w = asset.data.body_quat_w[:, body_id]
        body_lin_vel_w = asset.data.body_lin_vel_w[:, body_id, :3]
        body_lin_vel_b = math_utils.quat_rotate_inverse(body_quat_w, body_lin_vel_w)
    
    lin_vel_error = torch.sum(
        torch.square(cmd_vel[:,1] - body_lin_vel_b),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_global(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    cmd_ang_vel = env.command_manager.get_term(command_name).root_state["root_ang_vel_global"]
    
    if asset_cfg.body_ids is None or is_full_slice(asset_cfg.body_ids):
        # If no body_ids are specified, use the root body
        body_ang_vel_w = asset.data.root_ang_vel_w
    else:
        body_id = asset_cfg.body_ids[0] # just use the first body id for now
        asset = env.scene[asset_cfg.name]
        # body_states_w = asset.data.body_state_w[:, body_id]
        body_ang_vel_w = asset.data.body_ang_vel_w[:, body_id, :3]
        
    ang_vel_error = torch.sum(
        torch.square(cmd_ang_vel - body_ang_vel_w),
        dim=1,
    )

    return torch.exp(-ang_vel_error / std**2)

def track_ang_vel_local(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    cmd_ang_vel = env.command_manager.get_term(command_name).root_state["root_ang_vel"]

    if asset_cfg.body_ids is None or is_full_slice(asset_cfg.body_ids):
        # If no body_ids are specified, use the root body
        body_ang_vel_b = asset.data.root_ang_vel_b
    else:
        body_id = asset_cfg.body_ids[0]
        asset = env.scene[asset_cfg.name]
        # body_states_w = asset.data.body_state_w[:, body_id]
        body_quat_w = asset.data.body_quat_w[:, body_id]
        body_ang_vel_w = asset.data.body_ang_vel_w[:, body_id, :3]
        body_ang_vel_b = math_utils.quat_rotate_inverse(body_quat_w, body_ang_vel_w)
    
    ang_vel_error = torch.sum(
        torch.square(cmd_ang_vel[:,1] - body_ang_vel_b),
        dim=1,
    )
    
    return torch.exp(-ang_vel_error / std**2)


def force_command_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float = 10.0,
) -> torch.Tensor:
    """
    Reward tracking of force commands using an exponential kernel.
    """
    command_term = env.command_manager.get_term(command_name)
    target_forces_w = command_term.target_forces_w
    forcefield_forces_w = command_term.forcefield_forces_w
    force_error = torch.norm(target_forces_w - forcefield_forces_w, dim=1)
    return torch.exp(-force_error**2 / sigma**2)

def torque_command_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float = 2.0,
) -> torch.Tensor:
    """
    Reward tracking of torque commands using an exponential kernel.
    """
    command_term = env.command_manager.get_term(command_name)
    target_torques_w = command_term.target_torques_w
    forcefield_torques_w = command_term.forcefield_torques_w
    torque_error = torch.norm(target_torques_w - forcefield_torques_w, dim=1)
    return torch.exp(-torque_error**2 / sigma**2)

def force_command_tracking_normalized(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float = 10.0,
) -> torch.Tensor:
    """Normalized reward for tracking force commands."""
    command_term = env.command_manager.get_term(command_name)
    forcefield_stiffness = command_term.forcefield_stiffness
    desired_stiffness = command_term.desired_stiffness
    sum_stiffness = forcefield_stiffness + desired_stiffness
    sum_stiffness[sum_stiffness == 0] = 1.0

    target_forces_w = command_term.target_forces_w
    forcefield_forces_w = command_term.forcefield_forces_w
    force_error = torch.norm(target_forces_w - forcefield_forces_w, dim=1)
    normalized_error = force_error / sum_stiffness
    return torch.exp(-normalized_error**2 / sigma**2)

def torque_command_tracking_normalized(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Normalized reward for tracking torque commands."""
    command_term = env.command_manager.get_term(command_name)
    forcefield_stiffness = command_term.forcefield_rotational_stiffness
    desired_stiffness = command_term.desired_rotational_stiffness
    sum_stiffness = forcefield_stiffness + desired_stiffness
    sum_stiffness[sum_stiffness == 0] = 1.0

    target_torques_w = command_term.target_torques_w
    forcefield_torques_w = command_term.forcefield_torques_w
    torque_error = torch.norm(target_torques_w - forcefield_torques_w, dim=1)
    normalized_error = torque_error / sum_stiffness
    return torch.exp(-normalized_error**2 / sigma**2)

def force_link_keypoint_tracking_local(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of force link keypoints in the anchored frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    keypoint_body_ids_tensor = command_term.keypoint_body_ids_tensor
    if keypoint_body_ids_tensor is None:
        return torch.ones(env.num_envs, device=command_term.device)

    env_ids = torch.arange(env.num_envs, device=command_term.device)
    anchor_rot = command_term.ff_anchor_rot
    anchor_pos_robot = command_term.ff_anchor_pos
    anchor_pos_ref = command_term.ff_anchor_ref_pos
    commanded_keypoints = command_term.adapted_keypoints[:, 1]
    keypoints_relative = commanded_keypoints - anchor_pos_ref.unsqueeze(1)
    keypoints_transformed = math_utils.quat_rotate(anchor_rot.unsqueeze(1), keypoints_relative) + anchor_pos_robot.unsqueeze(1)
    current_keypoints = asset.data.body_pos_w[:, keypoint_body_ids_tensor, :3]
    keypoint_error = torch.norm(keypoints_transformed - current_keypoints, dim=-1)

    force_link_id = command_term.force_link_id[:, 1, 0].clone()
    force_link_id[force_link_id < 0] = 0
    force_link_idxs = (keypoint_body_ids_tensor.unsqueeze(0) == force_link_id.unsqueeze(1)).nonzero()[:, 1]
    force_link_error = keypoint_error[env_ids, force_link_idxs]

    reward = torch.exp(-force_link_error**2 / sigma**2)
    reward[~command_term.active_force_mask] = 1.0
    return reward


# new change
def online_compliant_keypoint_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track the per-environment online ``x_ref + F/K`` link target."""

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    env_ids = torch.arange(env.num_envs, device=asset.device)
    current = asset.data.body_pos_w[env_ids, command.selected_body_ids]
    desired = command.keypoints[
        env_ids, 1, command.selected_keypoint_ids
    ]
    error = torch.norm(current - desired, dim=-1)
    return torch.exp(-torch.square(error / sigma))

def force_link_keypoint_orientation_tracking_local(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of force link keypoint orientations in the anchored frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    keypoint_body_ids_tensor = command_term.keypoint_body_ids_tensor
    if keypoint_body_ids_tensor is None:
        return torch.ones(env.num_envs, device=command_term.device)

    env_ids = torch.arange(env.num_envs, device=command_term.device)
    anchor_rot = command_term.ff_anchor_rot
    commanded_keypoint_quats = command_term.adapted_keypoint_rotations[:, 1]
    num_keypoints = commanded_keypoint_quats.shape[1]
    anchor_rot_broadcastable = anchor_rot.unsqueeze(1).repeat(1, num_keypoints, 1)
    transformed_commanded_quats = math_utils.quat_mul(anchor_rot_broadcastable, commanded_keypoint_quats)
    current_keypoint_quats = asset.data.body_quat_w[:, keypoint_body_ids_tensor, :4]
    keypoint_orientation_error = math_utils.quat_error_magnitude(transformed_commanded_quats, current_keypoint_quats)

    force_link_id = command_term.force_link_id[:, 1, 0].clone()
    force_link_id[force_link_id < 0] = 0
    force_link_idxs = (keypoint_body_ids_tensor.unsqueeze(0) == force_link_id.unsqueeze(1)).nonzero()[:, 1]
    force_link_error = keypoint_orientation_error[env_ids, force_link_idxs]

    reward = torch.exp(-force_link_error**2 / sigma**2)
    reward[~command_term.active_force_mask] = 1.0
    return reward
