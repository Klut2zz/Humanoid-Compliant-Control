# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Sub-module containing a command generator for reading augmented reference motions.
This command generator extends ReferenceCommand to handle CSV files that include
the original reference motion, an "adapted" motion, external forces/torques,
and forcefield metadata for reactive force computation.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

# Import the base class
from .reference_command import ReferenceCommand
import isaaclab.utils.math as math_utils
from isaaclab.markers.visualization_markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import ComplianceAugmentedReferenceCommandCfg


def quat_from_axis_to_vector(
    vectors: torch.Tensor,
    axis: str = "z"
) -> torch.Tensor:
    """
    Calculates quaternions that rotate a specified canonical axis to align with the given vectors.
    Args:
        vectors (torch.Tensor): A tensor of shape (N, 3) representing the target vectors.
        axis (str): The canonical axis to rotate from. Can be 'x', 'y', or 'z'.
    Returns:
        torch.Tensor: A tensor of shape (N, 4) representing the calculated quaternions [w, x, y, z].
    """
    if axis == "x":
        from_axis = torch.tensor([1.0, 0.0, 0.0], device=vectors.device)
    elif axis == "y":
        from_axis = torch.tensor([0.0, 1.0, 0.0], device=vectors.device)
    else: # 'z'
        from_axis = torch.tensor([0.0, 0.0, 1.0], device=vectors.device)
    
    from_axis = from_axis.repeat(vectors.shape[0], 1)
    
    # Normalize input vectors safely
    vectors_norm = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    vectors_safe = torch.where(vectors_norm > 1e-6, vectors / vectors_norm, from_axis)
    
    # Calculate rotation axis (cross product) and angle (dot product)
    rot_axis = torch.cross(from_axis, vectors_safe, dim=1)
    dot = torch.sum(from_axis * vectors_safe, dim=-1)
    angle = torch.acos(torch.clamp(dot, -1.0, 1.0))

    # Handle the case where vectors are parallel or anti-parallel
    parallel_mask = dot > 0.99999
    antiparallel_mask = dot < -0.99999
    
    if axis == 'x' or axis == 'z':
        rot_axis[antiparallel_mask] = torch.tensor([0.0, 1.0, 0.0], device=vectors.device)
    else: # axis == 'y'
        rot_axis[antiparallel_mask] = torch.tensor([1.0, 0.0, 0.0], device=vectors.device)

    # Normalize the rotation axis safely
    rot_axis_norm = torch.linalg.norm(rot_axis, dim=-1, keepdim=True)
    rot_axis_safe = torch.where(rot_axis_norm > 1e-6, rot_axis / rot_axis_norm, from_axis)
    
    quats = math_utils.quat_from_angle_axis(angle, rot_axis_safe)
    quats[parallel_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=vectors.device)
    
    return quats

def quat_to_angle_axis(quaternions: torch.Tensor):
    """
    Convert quaternions to angle-axis representation.
    The angle will be in the range [0, pi].

    Args:
        quaternions (torch.Tensor): Input quaternions of shape (..., 4) with format [w, x, y, z].

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - angle (torch.Tensor): The angle of rotation in radians, in [0, pi].
            - axis (torch.Tensor): The axis of rotation (unit vector).
    """
    # Ensure quaternions are normalized
    q_norm = torch.nn.functional.normalize(quaternions, p=2, dim=-1)
    
    # --- MODIFICATION FOR WXYZ FORMAT ---
    # Extract scalar and vector parts
    # w is the first element, xyz are the last three
    w = q_norm[..., 0]
    xyz = q_norm[..., 1:]
    
    # To ensure the angle is in [0, pi], we take the shortest path.
    # A quaternion q and -q represent the same rotation. We choose the one
    # with a non-negative scalar part 'w'.
    neg_w_mask = w < 0
    w[neg_w_mask] *= -1
    xyz[neg_w_mask] *= -1

    # Calculate angle
    # The magnitude of the vector part is sin(angle/2)
    # The scalar part is cos(angle/2)
    angle = 2 * torch.acos(w)

    # Calculate axis
    # The axis is the normalized vector part.
    # We use a safe normalization that handles the case of near-zero vectors.
    vec_mag = torch.linalg.norm(xyz, dim=-1)
    # The axis is undefined for an identity rotation, so we can use a default.
    # A small epsilon is used to avoid division by zero.
    axis = xyz / (vec_mag.unsqueeze(-1) + 1e-8)
    
    # For true identity quaternions (angle is zero), the axis is arbitrary.
    # The safe normalization above handles this by producing a zero vector.
    # To be more explicit for users, we can set a default axis like [1, 0, 0].
    identity_mask = vec_mag < 1e-8
    default_axis = torch.tensor([1.0, 0.0, 0.0], device=quaternions.device, dtype=quaternions.dtype)
    # Use broadcasting to fill in the default axis where needed
    axis[identity_mask] = default_axis

    return angle, axis


class ComplianceAugmentedReferenceCommand(ReferenceCommand):
    """
    Command generator that samples motions from augmented AMASS-style datasets with forcefield info.

    This class extends `ReferenceCommand` to read from CSV files that contain four blocks of data:
    1. The original reference motion (e.g., from AMASS).
    2. An 'adapted' motion, which has the same format but is prefixed with 'adap_'.
    3. Force and Torque information, including the link ID and force/torque vectors.
    4. Forcefield metadata (stiffness, setpoint/origin, plane normal) for reactive force computation.

    The original reference motion is accessible, but by default, the main properties (`root_state`,
    `dof_pos`, etc.) are overridden to return the adapted motion and its derivatives. The new
    force/torque data are exposed through new properties. A config flag determines whether to
    use pre-computed feedforward forces or calculate them reactively from the forcefield data.
    """

    cfg: ComplianceAugmentedReferenceCommandCfg

    def __init__(self, cfg: ComplianceAugmentedReferenceCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.""" 
        
        # -- NEW: Buffers for anchoring the forcefield to the robot's initial state during an event --
        # This allows the forcefield's motion to be relative to the robot's pose at the moment of contact.
        self._ff_anchor_pos = torch.zeros(env.num_envs, 3, device=env.device)
        self._ff_anchor_rot = torch.zeros(env.num_envs, 4, device=env.device)
        self._ff_anchor_rot[:, 3] = 1.0  # Initialize to identity quaternion
        self._ff_anchor_ref_pos = torch.zeros(env.num_envs, 3, device=env.device)
        self._last_ff_stiffness = torch.zeros(env.num_envs, device=env.device)
        self._last_ff_rotational_stiffness = torch.zeros(env.num_envs, device=env.device)

        super().__init__(cfg, env)

        self._sorted_keypoint_ids = torch.linspace(0, 38, 39, dtype=torch.long, device=self.device)
        self.num_bodies = len(self._sorted_keypoint_ids)

        # Initialize buffers for the new augmented data.
        self._adapted_root_pos = torch.zeros(self.num_envs, self.n_future_steps + 1, 3, device=self.device)
        self._adapted_root_rot = torch.zeros(self.num_envs, self.n_future_steps + 1, 4, device=self.device)
        self._adapted_root_vel = torch.zeros(self.num_envs, self.n_future_steps + 1, 3, device=self.device)
        self._adapted_root_ang_vel = torch.zeros(self.num_envs, self.n_future_steps + 1, 3, device=self.device)
        self._adapted_root_vel_global = torch.zeros(self.num_envs, self.n_future_steps + 1, 3, device=self.device)
        self._adapted_root_ang_vel_global = torch.zeros(self.num_envs, self.n_future_steps + 1, 3, device=self.device)
        self._adapted_dof_pos = torch.zeros(self.num_envs, self.n_future_steps + 1, self.num_joints, device=self.device)
        self._adapted_dof_vel = torch.zeros(self.num_envs, self.n_future_steps + 1, self.num_joints, device=self.device)
        self._adapted_foot_contacts = torch.zeros(self.num_envs, self.n_future_steps + 1, 2, device=self.device)

        # Buffers for force and torque information
        self._force_link_id = torch.zeros(self.num_envs, self.n_future_steps + 1, 1, device=self.device, dtype=torch.long)
        self._force_vector = torch.zeros(self.num_envs, self.n_future_steps + 1, 3, device=self.device)
        self._torque_vector = torch.zeros(self.num_envs, self.n_future_steps + 1, 3, device=self.device)
        self._stiffness = torch.zeros(self.num_envs, self.n_future_steps + 1, 1, device=self.device)
        self._rotational_stiffness = torch.zeros(self.num_envs, self.n_future_steps + 1, 1, device=self.device)

        # --- Buffers for forcefield metadata ---
        self._ff_stiffness = torch.zeros(self.num_envs, self.n_future_steps + 1, 1, device=self.device)
        self._ff_rotational_stiffness = torch.zeros(self.num_envs, self.n_future_steps + 1, 1, device=self.device) # NEW
        self._ff_origin = torch.zeros(self.num_envs, self.n_future_steps + 1, 3, device=self.device)
        self._ff_setpoint_rot = torch.zeros(self.num_envs, self.n_future_steps + 1, 4, device=self.device) # NEW
        self._ff_normal = torch.zeros(self.num_envs, self.n_future_steps + 1, 3, device=self.device)
        steps = self.n_future_steps + 1
        self._force_vector_world = torch.zeros(self.num_envs, steps, 3, device=self.device)
        self._torque_vector_world = torch.zeros_like(self._force_vector_world)
        self._active_force_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Initialize metrics for logging
        self.metrics["force_link_keypoint_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_link_keypoint_orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_error_magnitude"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["torque_error_magnitude"] = torch.zeros(self.num_envs, device=self.device)

        # Perform an initial update to populate all buffers
        self._update_command(step=False)



    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)

        # refresh the event start state to clean any bad state
        self._last_ff_stiffness[env_ids] = 0.0
        self._last_ff_rotational_stiffness[env_ids] = 0.0
        self._active_force_mask[env_ids] = False
        self._force_vector_world[env_ids] = 0.0
        self._torque_vector_world[env_ids] = 0.0

        return super().reset(env_ids)

    def reset_motions(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)

        # refresh the event start state to clean any bad state
        self._last_ff_stiffness[env_ids] = 0.0
        self._last_ff_rotational_stiffness[env_ids] = 0.0
        self._active_force_mask[env_ids] = False
        self._force_vector_world[env_ids] = 0.0
        self._torque_vector_world[env_ids] = 0.0

        return super().reset_motions(env_ids)

    def _update_metrics(self):
        """Update metrics for the command generator."""
        if not getattr(self.cfg, "enable_metrics", False):
            return
        super()._update_metrics()

        # Compute the keypoint error for the force link of the adapted motion
        if self._adapted_keypoints is not None:
            keypoint_body_ids_tensor = self.keypoint_body_ids_tensor
            if keypoint_body_ids_tensor is None:
                return
            force_link_id = self._force_link_id[:, 0, 0].clone()
            force_link_id[force_link_id < 0] = 0  # No force link applied, set to 0
            force_link_idxs = (keypoint_body_ids_tensor.unsqueeze(0) == force_link_id.unsqueeze(1)).nonzero()[:, 1]
            env_ids = torch.arange(self.num_envs, device=self.device)

            current_root_pos = self.robot.data.root_pos_w[:, :3].clone(); current_root_pos[:, 2] = 0.0
            current_root_rot = self.robot.data.root_quat_w[:, :4].clone()
            _, _, current_yaw = math_utils.euler_xyz_from_quat(current_root_rot)
            
            commanded_root_pos = self.adapted_root_state["root_pos"][:,1,:3].clone(); commanded_root_pos[:, 2] = 0.0
            commanded_root_rot = self.adapted_root_state["root_rot"][:,1,:4].clone()
            commanded_root_rot = commanded_root_rot[:, [3, 0, 1, 2]]  # convert to wxyz
            _, _, commanded_yaw = math_utils.euler_xyz_from_quat(commanded_root_rot)
            
            yaw_diff = math_utils.wrap_to_pi(current_yaw - commanded_yaw)
            yaw_rotation = math_utils.quat_from_euler_xyz(torch.zeros_like(current_yaw), torch.zeros_like(current_yaw), yaw_diff)

            current_keypoints = self._env.scene["robot"].data.body_pos_w[:, keypoint_body_ids_tensor, :3]
            adapted_keypoints = self._adapted_keypoints[env_ids, 1]
            
            adapted_keypoints_local = adapted_keypoints - commanded_root_pos.unsqueeze(1)
            current_keypoints_local = current_keypoints - current_root_pos.unsqueeze(1)
            
            adapted_keypoints_local = math_utils.quat_rotate(
                yaw_rotation.unsqueeze(1).repeat(1, adapted_keypoints.shape[1], 1),
                adapted_keypoints_local,
            )
            
            adapted_keypoint_error = torch.norm(adapted_keypoints_local - current_keypoints_local, dim=-1)

            force_link_error = adapted_keypoint_error[env_ids, force_link_idxs]
            # force_link_error[self._force_link_id[:, 0, 0] < 0] = 0.0  # No error if no force link is applied
            force_link_error[~self.active_force_mask] = 0.0  # No error if no force link is applied

            self.metrics["force_link_keypoint_error"] = force_link_error
            
        # # compute the orientation error for the force link of the adapted motion
        # if self._adapted_keypoint_rotations is not None:
        #     current_keypoint_rotations = self.robot.data.body_quat_w[:, keypoint_body_ids_tensor, :4]
        #     adapted_keypoint_rotations = self._adapted_keypoint_rotations[env_ids, 1]

        #     adapted_keypoint_rotations_wxyz = adapted_keypoint_rotations[:, [3, 0, 1, 2]]
        #     current_keypoint_rotations_wxyz = current_keypoint_rotations[:, [3, 0, 1, 2]]

        #     # Apply yaw rotation to the adapted keypoint rotations
        #     adapted_keypoint_rotations_wxyz = math_utils.quat_mul(
        #         yaw_rotation.unsqueeze(1).repeat(1, adapted_keypoint_rotations.shape[1], 1),
        #         adapted_keypoint_rotations_wxyz,
        #     )

        #     # Compute relative rotation from adapted to current
        #     relative_rotations = math_utils.quat_mul(
        #         math_utils.quat_conjugate(adapted_keypoint_rotations_wxyz),
        #         current_keypoint_rotations_wxyz,
        #     )

        #     # Convert to angle-axis and extract the angle
        #     angles, _ = quat_to_angle_axis(relative_rotations)
            
        #     force_link_orientation_error = angles[env_ids, force_link_idxs]
        #     force_link_orientation_error[~self.active_force_mask] = 0.0
            
        #     self.metrics["force_link_keypoint_orientation_error"] = force_link_orientation_error

        # Compute the force and torque error magnitudes
        force_error = self.forcefield_forces_w - self.target_forces_w
        torque_error = self.forcefield_torques_w - self.target_torques_w
        
        self.metrics["force_error_magnitude"] = torch.norm(force_error, dim=-1)
        self.metrics["torque_error_magnitude"] = torch.norm(torque_error, dim=-1)

    def _update_command(self, step=True):
        """Update commands based on current motion time."""
        motion_times = self.motion_count * self.motion_dt
        motion_res = self.motion_lib.get_motion_state(
            self.motion_ids,
            motion_times,
            offset=self.offset,
            future_frame_dt=self.future_dt,
        )

        env_origins = self._env.scene.env_origins[self.motion_ids, :3]
        env_offsets = env_origins.unsqueeze(1)
        self._populate_reference_buffers(motion_res, env_origins)

        # --- Populate NEW Augmented Data Buffers ---
        if self.cfg.demo_zero_xy:
            adapted_root_pos = motion_res["adapted_root_pos"].clone() + env_offsets
            adapted_root_pos[:, :, 0:2] = env_offsets[:, :, 0:2]
        else:
            adapted_root_pos = motion_res["adapted_root_pos"].clone() + env_offsets
        
        self._adapted_root_pos, self._adapted_root_rot = adapted_root_pos, motion_res["adapted_root_rot"]
        self._adapted_root_vel, self._adapted_root_ang_vel = motion_res["adapted_root_vel"], motion_res["adapted_root_ang_vel"]
        self._adapted_root_vel_global, self._adapted_root_ang_vel_global = motion_res["adapted_root_vel_global"], motion_res["adapted_root_ang_vel_global"]
        self._adapted_dof_pos, self._adapted_dof_vel = motion_res["adapted_dof_pos"], motion_res["adapted_dof_vel"]
        self._adapted_foot_contacts, self._adapted_keypoints = motion_res["adapted_foot_contacts"], motion_res["adapted_keypoints"]
        self._adapted_keypoint_rotations = motion_res.get("adapted_keypoint_rotations", None)
        self._adapted_root_gravity_vec = motion_res["adapted_gravity_vec"]
        self._force_vector, self._torque_vector = motion_res["force_vector"], motion_res["torque_vector"]
        self._stiffness, self._rotational_stiffness = motion_res["stiffness"], motion_res["rotational_stiffness"]

        if self._adapted_keypoints is not None:
            if self.cfg.demo_zero_xy:
                self._adapted_keypoints[:, :, :, 0:2] -= motion_res["root_pos"][:, :, 0:2].unsqueeze(2)
                self._adapted_keypoints[:, :, :, 0:2] += self._root_pos[:, :, 0:2].unsqueeze(2)
            else:
                self._adapted_keypoints += env_offsets.unsqueeze(1)
            
        # --- Populate and transform forcefield metadata with robot-relative anchoring ---
        self._ff_stiffness = motion_res["ff_stiffness"]
        self._ff_rotational_stiffness = motion_res["ff_rotational_stiffness"]
        
        # 1. Apply standard env_origin transformation to the raw forcefield data
        if self.cfg.demo_zero_xy:
            ff_origin_world = motion_res["ff_origin"].clone() + env_offsets
            ff_origin_world[:, :, 0:2] = env_offsets[:, :, 0:2]
        else:
            ff_origin_world = motion_res["ff_origin"].clone() + env_offsets
        
        # Rotations do not need env_origin translation
        ff_setpoint_rot_world = motion_res["ff_setpoint_rot"]
        ff_normal_world = motion_res["ff_normal"]  # Normal is a direction vector, no translation needed

        # 2. Logic to anchor forcefield to the robot's state at the start of an interaction
        current_ff_stiffness = self._ff_stiffness[:, 0, 0]
        current_ff_rotational_stiffness = self._ff_rotational_stiffness[:, 0, 0]
        event_start_mask = (self._last_ff_stiffness < 0.1) & (current_ff_stiffness >= 0.1)
        event_active_mask = current_ff_stiffness >= 0.1

        # 2a. On the rising edge of a force event, record the relative transform
        if torch.any(event_start_mask):
            start_env_ids = torch.where(event_start_mask)[0]
            
            # Get robot's current state (position and yaw) for these envs
            current_root_pos = self.robot.data.root_pos_w[start_env_ids, :3]
            current_root_rot = self.robot.data.root_quat_w[start_env_ids, :4]
            _, _, current_yaw = math_utils.euler_xyz_from_quat(current_root_rot)
            
            # Get the reference motion's state at the event start time
            commanded_root_pos = self._adapted_root_pos[start_env_ids, 0, :3]
            commanded_root_rot = self._adapted_root_rot[start_env_ids, 0, :4]  # xyzw
            commanded_root_rot_wxyz = commanded_root_rot[:, [3, 0, 1, 2]]
            _, _, commanded_yaw = math_utils.euler_xyz_from_quat(commanded_root_rot_wxyz)
            
            # Calculate and store the delta transform for anchoring the forcefield trajectory
            delta_yaw = math_utils.wrap_to_pi(current_yaw - commanded_yaw)
            delta_rot_quat = math_utils.quat_from_euler_xyz(torch.zeros_like(delta_yaw), torch.zeros_like(delta_yaw), delta_yaw)
            
            self._ff_anchor_pos[start_env_ids] = current_root_pos
            self._ff_anchor_rot[start_env_ids] = delta_rot_quat
            self._ff_anchor_ref_pos[start_env_ids] = commanded_root_pos

        # 2b. For all environments with an active force, apply the stored anchor transform
        if torch.any(event_active_mask):
            active_env_ids = torch.where(event_active_mask)[0]
            
            # Retrieve anchor data for active environments
            anchor_rot = self._ff_anchor_rot[active_env_ids]
            anchor_ref_pos = self._ff_anchor_ref_pos[active_env_ids]
            anchor_pos = self._ff_anchor_pos[active_env_ids]
            
            # Get current and future forcefield data for transformation
            current_ff_origin = ff_origin_world[active_env_ids]
            current_ff_normal = ff_normal_world[active_env_ids]
            current_ff_setpoint_rot = ff_setpoint_rot_world[active_env_ids]
            
            # Apply the transform: O_new = R_delta * (O_data - P_ref_start) + P_robot_start
            origin_relative_to_ref_start = current_ff_origin - anchor_ref_pos.unsqueeze(1)
            origin_relative_rotated = math_utils.quat_rotate(anchor_rot.unsqueeze(1), origin_relative_to_ref_start)
            final_origin = origin_relative_rotated + anchor_pos.unsqueeze(1)
            final_origin[:, :, 2] = current_ff_origin[:, :, 2]

            # Apply rotation to the normal vector: N_new = R_delta * N_data
            final_normal = math_utils.quat_rotate(anchor_rot.unsqueeze(1), current_ff_normal)
            
            # Apply rotation to the setpoint orientation: R_new = R_delta * R_data
            num_future_steps = current_ff_setpoint_rot.shape[1]
            anchor_rot_broadcasted = anchor_rot.unsqueeze(1).repeat(1, num_future_steps, 1)
            final_setpoint_rot = math_utils.quat_mul(anchor_rot_broadcasted, current_ff_setpoint_rot)

            # Update the main buffers for these active environments
            ff_origin_world[active_env_ids] = final_origin
            ff_normal_world[active_env_ids] = final_normal
            ff_setpoint_rot_world[active_env_ids] = final_setpoint_rot

        # 3. Final assignment to class properties
        self._ff_origin = ff_origin_world
        self._ff_setpoint_rot = ff_setpoint_rot_world
        self._ff_normal = ff_normal_world
        
        # 4. Update state for the next step's rising-edge detection
        self._last_ff_stiffness = current_ff_stiffness.clone()
        self._last_ff_rotational_stiffness = current_ff_rotational_stiffness.clone()

        # Map MuJoCo link IDs from the data file to Isaac Lab's internal body indices
        force_link_id_mujoco = motion_res["force_link_id"].clone()
        mujoco_to_isaaclab_link_id_map = { -1: -1, 23: 36, 30: 37, 16: 11, 17: 17, 19: 28, 24: 20, 26: 29 }
        self._force_link_id = torch.full_like(force_link_id_mujoco, -1)
        for mujoco_id, isaaclab_id in mujoco_to_isaaclab_link_id_map.items():
            self._force_link_id[force_link_id_mujoco == mujoco_id] = isaaclab_id

        anchor_quat = self._ff_anchor_rot.unsqueeze(1)
        self._force_vector_world = math_utils.quat_rotate(anchor_quat, self._force_vector)
        self._torque_vector_world = math_utils.quat_rotate(anchor_quat, self._torque_vector)
        self._active_force_mask = self._force_vector_world[:, 1].norm(dim=-1) > 1e-6
        
        self._finalize_update(step)

    # -- Properties for accessing adapted data --

    @property
    def adapted_root_state(self) -> dict[str, torch.Tensor]:
        """Return the current **adapted** root state of the motion."""
        return {
            "root_pos": self._adapted_root_pos, "root_rot": self._adapted_root_rot,
            "root_vel": self._adapted_root_vel, "root_ang_vel": self._adapted_root_ang_vel,
            "root_vel_global": self._adapted_root_vel_global, "root_ang_vel_global": self._adapted_root_ang_vel_global,
            "projected_gravity_b": self._adapted_root_gravity_vec
        }
    
    @property
    def adapted_dof_pos(self) -> torch.Tensor:
        """Return the **adapted** target joint positions."""
        return self._adapted_dof_pos

    @property
    def adapted_dof_vel(self) -> torch.Tensor:
        """Return the **adapted** target joint velocities."""
        return self._adapted_dof_vel

    @property
    def adapted_foot_contacts(self) -> torch.Tensor:
        """Return the **adapted** foot contact states."""
        return self._adapted_foot_contacts
    
    @property
    def adapted_keypoints(self):
        """Return the **adapted** keypoints."""
        return self._adapted_keypoints
    
    @property
    def adapted_keypoint_rotations(self):
        """Return the **adapted** keypoint rotations."""
        return self._adapted_keypoint_rotations


    # -- Override base class properties to return adapted data by default --

    @property
    def root_state(self) -> dict[str, torch.Tensor]:
        """Return the current **adapted** root state of the motion."""
        return self.adapted_root_state if self.cfg.override_reference else super().root_state

    @property
    def dof_pos(self) -> torch.Tensor:
        """Return the **adapted** target joint positions."""
        return self._adapted_dof_pos if self.cfg.override_reference else super().dof_pos

    @property
    def dof_vel(self) -> torch.Tensor:
        """Return the **adapted** target joint velocities."""
        return self._adapted_dof_vel if self.cfg.override_reference else super().dof_vel
    
    @property
    def foot_contacts(self) -> torch.Tensor:
        """Return the **adapted** foot contact states."""
        return self._adapted_foot_contacts if self.cfg.override_reference else super().foot_contacts

    @property
    def keypoints(self):
        """Return the **adapted** keypoints."""
        return self._adapted_keypoints if self.cfg.override_reference else super().keypoints
    
    @property
    def keypoint_rotations(self):
        """Return the **adapted** keypoint rotations."""
        return self._adapted_keypoint_rotations if self.cfg.override_reference else super().keypoint_rotations

    # -- Properties for accessing force, torque, and forcefield data --
        
    @property
    def force_link_id(self) -> torch.Tensor:
        """Return the ID of the link to which an external force was applied."""
        return self._force_link_id

    @property
    def force_vector(self) -> torch.Tensor:
        """Return the 3D external force vector from the CSV, rotated based on the yaw offset at the start of the force event."""
        return self._force_vector_world

    @property
    def torque_vector(self) -> torch.Tensor:
        """Return the 3D external torque vector from the CSV, rotated based on the yaw offset at the start of the force event."""
        return self._torque_vector_world
    
    def _compute_feedforward_force_w(self) -> torch.Tensor:
        """Computes forces in the world frame using the pre-computed 'feedforward' vectors from the dataset."""
        return self._force_vector_world[:, 1, :]

    def _compute_reactive_forcefield_force_w(self) -> torch.Tensor:
        """Computes forces in the world frame reactively based on the robot's state and forcefield metadata."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        target_body_ids = self.body_indices[:]
        # feedforward_forces = self.
        # active_force_mask = feedforw_compute_feedforward_force_w()ard_forces.norm(dim=-1) > 0
        active_force_mask = self.active_force_mask

        # Get current body positions for environments with an active force
        body_positions_w = torch.zeros(self.num_envs, 3, device=self.device)
        if torch.any(active_force_mask):
            active_env_ids = env_ids[active_force_mask]
            active_body_ids = target_body_ids[active_force_mask]
            body_positions_w[active_force_mask] = self.robot.data.body_pos_w[active_env_ids, active_body_ids]

        # Get forcefield parameters for the current timestep
        stiffness = self._stiffness[:, 0]
        ff_stiffness = self._ff_stiffness[:, 0, 0]
        ff_origin = self._ff_origin[:, 0]
        ff_normal = self._ff_normal[:, 0]

        # Calculate effective stiffness
        k_eff = ff_stiffness

        global_force_vector = torch.zeros_like(body_positions_w)

        # Handle plane-based forcefields
        is_plane_mask = torch.linalg.norm(ff_normal, dim=-1) > 0.1
        plane_envs = active_force_mask & is_plane_mask
        if torch.any(plane_envs):
            p_current = body_positions_w[plane_envs]
            penetration = -torch.sum((p_current - ff_origin[plane_envs]) * ff_normal[plane_envs], dim=-1)
            force_mag = torch.clamp(penetration, min=0.0) * k_eff[plane_envs]
            global_force_vector[plane_envs] = force_mag.unsqueeze(-1) * ff_normal[plane_envs]

        # Handle setpoint-based forcefields
        setpoint_envs = active_force_mask & ~is_plane_mask
        if torch.any(setpoint_envs):
            p_current = body_positions_w[setpoint_envs]
            force_dir = ff_origin[setpoint_envs] - p_current
            global_force_vector[setpoint_envs] = k_eff[setpoint_envs].unsqueeze(-1) * force_dir

        return global_force_vector
    
    def _compute_feedforward_torque_w(self) -> torch.Tensor:
        """Computes torques in the world frame using the pre-computed 'feedforward' vectors from the dataset."""
        return self._torque_vector_world[:, 1, :]

    def _compute_reactive_forcefield_torque_w(self) -> torch.Tensor:
        """Computes torques in the world frame reactively based on the robot's orientation and forcefield metadata."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        target_body_ids = self.body_indices[:]
        active_force_mask = self.active_force_mask
        
        # Initialize output tensor
        global_torque_vector = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Only compute for environments with an active force event
        if torch.any(active_force_mask):
            active_env_ids = env_ids[active_force_mask]
            active_body_ids = target_body_ids[active_force_mask]

            # Get current body orientations
            body_quat_w = self.robot.data.body_quat_w[active_env_ids, active_body_ids]
            
            # Get forcefield parameters for the current timestep
            rot_stiffness = self._ff_rotational_stiffness[active_force_mask, 0, 0]
            ff_setpoint_rot = self._ff_setpoint_rot[active_force_mask, 0][..., [3, 0, 1, 2]]

            # Calculate rotational error quaternion: q_error = q_target * inv(q_current)
            # This quaternion represents the rotation needed to get from current to target
            body_quat_w_inv = math_utils.quat_inv(body_quat_w)
            delta_rot_quat = math_utils.quat_mul(ff_setpoint_rot, body_quat_w_inv)
            
            # Convert the error quaternion to an axis-angle representation (rotation vector)
            # The vector's direction is the axis of torque, and magnitude is the angle.
            angle, axis = quat_to_angle_axis(delta_rot_quat)
            rot_vec_error = axis * angle.unsqueeze(-1)
            
            # Torque is stiffness multiplied by the angular error vector
            # τ = k * (angle * axis)
            global_torque_vector[active_force_mask] = rot_stiffness.unsqueeze(-1) * rot_vec_error

            # delta_body = math_utils.quat_mul(math_utils.quat_inv(body_quat_w), ff_setpoint_rot)
            # angle, axis_body = quat_to_angle_axis(delta_body)
            # tau_body = rot_stiffness.unsqueeze(-1) * (axis_body * angle.unsqueeze(-1))
            
            # # tau_world = math_utils.quat_rotate(body_quat_w, tau_body)
            # global_torque_vector[active_force_mask] = tau_world

        return global_torque_vector

    @property
    def forcefield_forces_w(self) -> torch.Tensor:
        """
        Return force field forces in the world frame.
        The computation method is determined by `cfg.force_computation_mode`.
        """
        if self.cfg.force_computation_mode == 'feedforward':
            return self._compute_feedforward_force_w()
        elif self.cfg.force_computation_mode == 'forcefield':
            return self._compute_reactive_forcefield_force_w()
        else:
            raise ValueError(f"Unknown force_computation_mode: {self.cfg.force_computation_mode}")

    @property
    def forcefield_torques_w(self) -> torch.Tensor:
        """
        Return force field torques in the world frame.
        The computation method is determined by `cfg.force_computation_mode`.
        """
        if self.cfg.force_computation_mode == 'feedforward':
            return self._compute_feedforward_torque_w()
        elif self.cfg.force_computation_mode == 'forcefield':
            return self._compute_reactive_forcefield_torque_w()
        else:
            raise ValueError(f"Unknown force_computation_mode: {self.cfg.force_computation_mode}")
        
    @property
    def target_forces_w(self) -> torch.Tensor:
        """Alias for `forcefield_forces_w` for compatibility."""
        return self._compute_feedforward_force_w()
    
    @property
    def target_torques_w(self) -> torch.Tensor:
        """Alias for the pre-computed feedforward torques for compatibility."""
        return self._compute_feedforward_torque_w()
    
    @property
    def target_forces_b(self) -> torch.Tensor:
        """Return the target forces transformed into the target body's local frame."""
        global_force_vector = self._compute_feedforward_force_w()
        body_ids = self.body_indices
        env_ids = torch.arange(self.num_envs, device=self.device)
        body_quat_w = self.robot.data.body_quat_w[env_ids, body_ids, :].squeeze(1)
        return math_utils.quat_apply(math_utils.quat_inv(body_quat_w), global_force_vector)
    
    @property
    def target_torques_b(self) -> torch.Tensor:
        """Return the target torques transformed into the target body's local frame."""
        global_torque_vector = self._compute_feedforward_torque_w()
        body_ids = self.body_indices
        env_ids = torch.arange(self.num_envs, device=self.device)
        body_quat_w = self.robot.data.body_quat_w[env_ids, body_ids, :].squeeze(1)
        return math_utils.quat_apply(math_utils.quat_inv(body_quat_w), global_torque_vector)

    @property
    def forcefield_forces_b(self) -> torch.Tensor:
        """
        Return force field forces in the target body's local frame.
        Computation depends on `cfg.force_computation_mode`.
        """
        global_force_vector = self.forcefield_forces_w
        body_ids = self.body_indices
        env_ids = torch.arange(self.num_envs, device=self.device)
        body_quat_w = self.robot.data.body_quat_w[env_ids, body_ids, :].squeeze(1)
        return math_utils.quat_apply(math_utils.quat_inv(body_quat_w), global_force_vector)

    @property
    def forcefield_torques_b(self) -> torch.Tensor:
        """Return the force field torques transformed into the target body's local frame."""
        global_torque_vector = self.forcefield_torques_w
        body_ids = self.body_indices
        env_ids = torch.arange(self.num_envs, device=self.device)
        body_quat_w = self.robot.data.body_quat_w[env_ids, body_ids, :].squeeze(1)
        return math_utils.quat_apply(math_utils.quat_inv(body_quat_w), global_torque_vector)
    
    @property
    def desired_stiffness(self) -> torch.Tensor:
        """Return the stiffness of the force field."""
        return self._stiffness[:, 0].clone()

    @property
    def desired_rotational_stiffness(self) -> torch.Tensor:
        """Return the rotational stiffness of the force field."""
        return self._rotational_stiffness[:, 0].clone()
    
    @property
    def forcefield_stiffness(self) -> torch.Tensor:
        """Alias for `ff_stiffness` for compatibility."""
        return self._ff_stiffness[:, 0, 0].clone()
    
    @property
    def forcefield_rotational_stiffness(self) -> torch.Tensor:
        """Alias for `ff_rotational_stiffness` for compatibility."""
        return self._ff_rotational_stiffness[:, 0, 0].clone()

    @property
    def ff_anchor_pos(self) -> torch.Tensor:
        """The robot's root position in the world frame at the start of the force event."""
        return self._ff_anchor_pos

    @property
    def ff_anchor_rot(self) -> torch.Tensor:
        """The rotational difference (yaw-only quaternion) between the robot and the reference motion
        at the start of the force event. This is the key transform to align trajectories."""
        return self._ff_anchor_rot

    @property
    def ff_anchor_ref_pos(self) -> torch.Tensor:
        """The reference motion's root position in the world frame at the start of the force event."""
        return self._ff_anchor_ref_pos

    # -- Properties for compatibility --
    @property
    def body_indices(self) -> torch.Tensor:
        return self._force_link_id[:, 0, 0]
    
    @property
    def sorted_keypoint_ids(self) -> torch.Tensor:
        """A sorted tensor of ALL possible body IDs that can be targeted by a force."""
        return self._sorted_keypoint_ids
    
    @property
    def body_ids(self) -> torch.Tensor:
        """An alias for `body_indices` for compatibility."""
        return self.body_indices
    
    @property
    def active_force_mask(self) -> torch.Tensor:
        """A boolean mask indicating which environments have an active force applied."""
        return self._active_force_mask

    # -- Debug Visualization --

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set up or tear down visualization markers."""
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if not hasattr(self, "force_arrow_visualizer"):
                self.force_arrow_visualizer = VisualizationMarkers(self.cfg.force_vector_visualizer_cfg)
            if not hasattr(self, "recomputed_force_visualizer"):
                self.recomputed_force_visualizer = VisualizationMarkers(self.cfg.recomputed_force_vector_visualizer_cfg)
            if self.cfg.debug_vis_keypoints:
                if not hasattr(self, "adapted_keypoints_visualizer"):
                    self.adapted_keypoints_visualizer = VisualizationMarkers(self.cfg.adapted_keypoints_visualizer_cfg)
                if not hasattr(self, "ff_setpoint_visualizer"):
                    self.ff_setpoint_visualizer = VisualizationMarkers(self.cfg.ff_setpoint_visualizer_cfg)
            # if not hasattr(self, "ff_plane_visualizer"):
            #     self.ff_plane_visualizer = VisualizationMarkers(self.cfg.ff_plane_visualizer_cfg)
            # if not hasattr(self, "target_torque_visualizer"):
            #     self.target_torque_visualizer = VisualizationMarkers(self.cfg.target_torque_visualizer_cfg)
            # if not hasattr(self, "computed_torque_visualizer"):
            #     self.computed_torque_visualizer = VisualizationMarkers(self.cfg.computed_torque_visualizer_cfg)
            
            self.force_arrow_visualizer.set_visibility(True)
            self.recomputed_force_visualizer.set_visibility(True)
            if self.cfg.debug_vis_keypoints:
                self.adapted_keypoints_visualizer.set_visibility(True)
                self.ff_setpoint_visualizer.set_visibility(True)
            # self.ff_plane_visualizer.set_visibility(True)
            # self.target_torque_visualizer.set_visibility(True)
            # self.computed_torque_visualizer.set_visibility(True)
        else:
            if hasattr(self, "force_arrow_visualizer"): self.force_arrow_visualizer.set_visibility(False)
            if hasattr(self, "recomputed_force_visualizer"): self.recomputed_force_visualizer.set_visibility(False)
            if self.cfg.debug_vis_keypoints:
                if hasattr(self, "adapted_keypoints_visualizer"): self.adapted_keypoints_visualizer.set_visibility(False)
                if hasattr(self, "ff_setpoint_visualizer"): self.ff_setpoint_visualizer.set_visibility(False)
            # if hasattr(self, "ff_plane_visualizer"): self.ff_plane_visualizer.set_visibility(False)
            # if hasattr(self, "target_torque_visualizer"): self.target_torque_visualizer.set_visibility(False)
            # if hasattr(self, "computed_torque_visualizer"): self.computed_torque_visualizer

    def _debug_vis_callback(self, event):
        """Update visualization markers in the simulation."""
        super()._debug_vis_callback(event)

        
        # Common variables for force visualizations
        env_ids = torch.arange(self.num_envs, device=self.device)
        target_body_ids = self.body_indices[:]
        
        # Get body positions safely, handling envs where no force is applied (ID=-1)
        body_positions_w = torch.zeros(self.num_envs, 3, device=self.device)
        active_body_mask = target_body_ids >= 0
        if torch.any(active_body_mask):
            active_env_ids = env_ids[active_body_mask]
            active_body_ids = target_body_ids[active_body_mask]
            body_positions_w[active_body_mask] = self.robot.data.body_pos_w[active_env_ids, active_body_ids]

        # --- Visualize Feedforward Force Arrow (from CSV) using the helper function ---
        force_global = self._compute_feedforward_force_w()
        # active_force_mask = torch.linalg.norm(force_global, dim=-1) > 0.1
        active_force_mask = self.active_force_mask
        if torch.any(active_force_mask):
            self.force_arrow_visualizer.set_visibility(True)
            active_forces = force_global[active_force_mask]
            active_pos = body_positions_w[active_force_mask]
            
            arrow_orientations = quat_from_axis_to_vector(active_forces, axis='x')
            force_magnitude = torch.linalg.norm(active_forces, dim=-1)
            arrow_length = torch.clamp(force_magnitude / 5.0, 0.0, 10.0)
            
            default_scale = self.force_arrow_visualizer.cfg.markers["arrow"].scale
            arrow_scales = torch.tensor(default_scale, device=self.device).repeat(len(active_forces), 1)
            arrow_scales[:, 0] *= arrow_length # Set length (X-axis)
            
            arrow_positions = active_pos + (math_utils.normalize(active_forces) * arrow_scales[:, 0:1] * 0.1)
            self.force_arrow_visualizer.visualize(translations=arrow_positions, orientations=arrow_orientations, scales=arrow_scales)
        else:
            self.force_arrow_visualizer.set_visibility(False)

        # --- Visualize Recomputed Force Arrow (from forcefield) using the helper function ---
        recomputed_force_global = self._compute_reactive_forcefield_force_w()
        active_recomputed_force_mask = torch.linalg.norm(recomputed_force_global, dim=-1) > 0.1
        if torch.any(active_recomputed_force_mask):
            self.recomputed_force_visualizer.set_visibility(True)
            active_forces = recomputed_force_global[active_recomputed_force_mask]
            active_pos = body_positions_w[active_recomputed_force_mask]
            
            arrow_orientations = quat_from_axis_to_vector(active_forces, axis='x')
            force_magnitude = torch.linalg.norm(active_forces, dim=-1)
            arrow_length = torch.clamp(force_magnitude / 5.0, 0.0, 10.0)
            
            default_scale = self.recomputed_force_visualizer.cfg.markers["arrow"].scale
            arrow_scales = torch.tensor(default_scale, device=self.device).repeat(len(active_forces), 1)
            arrow_scales[:, 0] *= arrow_length
            
            arrow_positions = active_pos + (math_utils.normalize(active_forces) * arrow_scales[:, 0:1] * 0.1)
            self.recomputed_force_visualizer.visualize(translations=arrow_positions, orientations=arrow_orientations, scales=arrow_scales)
        else:
            self.recomputed_force_visualizer.set_visibility(False)

        # # --- Visualize Forcefield metadata ---
        
        if self.cfg.debug_vis_keypoints:
            ff_origin = self._ff_origin[:, 0]
            ff_normal = self._ff_normal[:, 0]
            is_ff_active_mask = self._ff_stiffness[:, 0, 0] > 0.1
            
            # Visualize Setpoint (Sphere)
            is_setpoint_mask = torch.linalg.norm(ff_normal, dim=-1) < 0.1
            active_setpoint_mask = is_ff_active_mask & is_setpoint_mask
            if torch.any(active_setpoint_mask):
                setpoint_pos = ff_origin[active_setpoint_mask]
                self.ff_setpoint_visualizer.visualize(translations=setpoint_pos)#, marker_env_ids=torch.where(active_setpoint_mask)[0])

            # Visualize Adapted Keypoints
            if self._adapted_keypoints is not None:
                num_keypoints = self._adapted_keypoints.shape[2]
                current_root_pos = self.robot.data.root_pos_w[:, :3].clone(); current_root_pos[:, 2] = 0.0
                current_root_rot = self.robot.data.root_quat_w[:, :4].clone()
                _, _, current_yaw = math_utils.euler_xyz_from_quat(current_root_rot)
                
                commanded_root_pos = self._adapted_root_pos[:,0, :3].clone(); commanded_root_pos[:, 2] = 0.0
                commanded_root_rot = self._adapted_root_rot[:,0, :4].clone()
                commanded_root_rot = commanded_root_rot[:, [3, 0, 1, 2]] 
                _, _, commanded_yaw = math_utils.euler_xyz_from_quat(commanded_root_rot)
                
                yaw_diff = math_utils.wrap_to_pi(current_yaw - commanded_yaw)
                yaw_rotation = math_utils.quat_from_euler_xyz(torch.zeros_like(current_yaw), torch.zeros_like(current_yaw), yaw_diff)
                
                adapted_commanded_keypoints_local = self._adapted_keypoints[:,0, :, :3].clone() - commanded_root_pos.unsqueeze(1)
                adapted_commanded_keypoints_local = math_utils.quat_rotate(
                    yaw_rotation.unsqueeze(1).repeat(1, adapted_commanded_keypoints_local.shape[1], 1),
                    adapted_commanded_keypoints_local
                )

                marker_pos = adapted_commanded_keypoints_local + current_root_pos.unsqueeze(1)
                marker_pos = marker_pos.reshape(-1, 3)
                marker_quat = torch.zeros(marker_pos.shape[0], 4, device=self.device); marker_quat[:, 3] = 1.0
                
                self.adapted_keypoints_visualizer.visualize(translations=marker_pos, orientations=marker_quat)


        # # Visualize Plane (thin Box)
        # is_plane_mask = ~is_setpoint_mask
        # active_plane_mask = is_ff_active_mask & is_plane_mask
        # if torch.any(active_plane_mask):
        #     plane_pos = ff_origin[active_plane_mask]
        #     plane_normals = ff_normal[active_plane_mask]
        #     plane_orientations = quat_from_axis_to_vector(plane_normals, axis='z')
        #     self.ff_plane_visualizer.visualize(translations=plane_pos, orientations=plane_orientations)#, marker_env_ids=torch.where(active_plane_mask)[0])
            
        # # Visualize Target Torque (Desired Torque from dataset)
        # target_torque_w = self.target_torques_w
        # active_target_torque_mask = torch.linalg.norm(target_torque_w, dim=-1) > 0.01
        # if torch.any(active_target_torque_mask):
        #     active_torques = target_torque_w[active_target_torque_mask]
        #     active_pos = body_positions_w[active_target_torque_mask]

        #     arrow_orientations = quat_from_axis_to_vector(active_torques, axis='x')
        #     torque_magnitude = torch.linalg.norm(active_torques, dim=-1)
        #     arrow_length = torch.clamp(torque_magnitude / 0.3, 0.0, 30.0)

        #     default_scale = self.target_torque_visualizer.cfg.markers["arrow"].scale
        #     arrow_scales = torch.tensor(default_scale, device=self.device).repeat(len(active_torques), 1)
        #     arrow_scales[:, 0] *= arrow_length

        #     arrow_positions = active_pos + (math_utils.normalize(active_torques) * arrow_scales[:, 0:1] * 0.1)
        #     self.target_torque_visualizer.visualize(
        #         translations=arrow_positions, orientations=arrow_orientations, scales=arrow_scales
        #     )

        # # Visualize Computed Torque (from forcefield)
        # computed_torque_w = self.forcefield_torques_w
        # active_computed_torque_mask = torch.linalg.norm(computed_torque_w, dim=-1) > 0.01
        # if torch.any(active_computed_torque_mask):
        #     active_torques = computed_torque_w[active_computed_torque_mask]
        #     active_pos = body_positions_w[active_computed_torque_mask]

        #     arrow_orientations = quat_from_axis_to_vector(active_torques, axis='x')
        #     torque_magnitude = torch.linalg.norm(active_torques, dim=-1)
        #     arrow_length = torch.clamp(torque_magnitude / 0.3, 0.0, 30.0)

        #     default_scale = self.computed_torque_visualizer.cfg.markers["arrow"].scale
        #     arrow_scales = torch.tensor(default_scale, device=self.device).repeat(len(active_torques), 1)
        #     arrow_scales[:, 0] *= arrow_length

        #     arrow_positions = active_pos + (math_utils.normalize(active_torques) * arrow_scales[:, 0:1] * 0.1)
        #     self.computed_torque_visualizer.visualize(
        #         translations=arrow_positions, orientations=arrow_orientations, scales=arrow_scales
        #     )
