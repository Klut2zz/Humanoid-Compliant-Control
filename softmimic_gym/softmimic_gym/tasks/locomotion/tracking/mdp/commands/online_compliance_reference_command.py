# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Online force-conditioned reference generation for compliant standing."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.markers.visualization_markers import VisualizationMarkers

from softmimic_gym.assets.robots.g1_compat import (
    G1_SOFTMIMIC_LEFT_ARM_ACTION_IDS,
    G1_SOFTMIMIC_RIGHT_ARM_ACTION_IDS,
)

from .reference_command import ReferenceCommand
# new change
from .compliance_augmented_reference_command import quat_from_axis_to_vector

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import OnlineComplianceReferenceCommandCfg


# new change
class OnlineComplianceReferenceCommand(ReferenceCommand):
    """Generate intermittent force events and feasible whole-body references.

    The original SoftMimic augmentation solves constrained IK offline and then
    trains PPO to imitate the augmented CSV.  This command keeps the same
    reference-construction idea inside every IsaacLab rollout.  A batched,
    damped least-squares solve maps ``F / K`` into the strict 29DOF interface
    while retaining feet, COM, pelvis, posture, and keypoint constraints.
    No generated transition is written to disk before PPO consumes it.
    """

    def __init__(
        self, cfg: OnlineComplianceReferenceCommandCfg, env: ManagerBasedEnv
    ):
        # ReferenceCommand.__init__ invokes _update_command once.  The guard
        # prevents virtual dispatch from using online buffers before they exist.
        self._online_ready = False
        super().__init__(cfg, env)

        self.cfg = cfg
        self._step_dt = float(env.step_dt)
        self._horizon_steps = self.n_future_steps + 1
        self._env_ids = torch.arange(self.num_envs, device=self.device)
        self._canonical_joint_ids = torch.as_tensor(
            self.control_joint_ids, device=self.device, dtype=torch.long
        )
        jacobian_offset = 0 if not self.robot.is_fixed_base else 0
        if not self.robot.is_fixed_base:
            jacobian_offset = 6
        self._jacobian_joint_columns = self._canonical_joint_ids + jacobian_offset

        candidate_ids, candidate_names = self.robot.find_bodies(
            list(cfg.force_body_names), preserve_order=True
        )
        if list(candidate_names) != list(cfg.force_body_names):
            raise RuntimeError(
                "Online compliance force bodies did not resolve in order: "
                f"requested={list(cfg.force_body_names)}, resolved={list(candidate_names)}"
            )
        self._candidate_body_ids = torch.as_tensor(
            candidate_ids, device=self.device, dtype=torch.long
        )
        # new change
        # Preserve the original five body-context slots when a derived task
        # samples only a subset (ArmLocalIKV2 uses left/right wrist slots 2/4).
        if cfg.force_body_context_slots:
            if len(cfg.force_body_context_slots) != len(candidate_names):
                raise RuntimeError(
                    "force_body_context_slots must match force_body_names."
                )
            context_slots = list(cfg.force_body_context_slots)
        else:
            context_slots = list(range(len(candidate_names)))
        if any(
            slot < 0 or slot >= int(cfg.force_body_context_size)
            for slot in context_slots
        ):
            raise RuntimeError("Online force body context slot is out of range.")
        self._candidate_context_slots = torch.as_tensor(
            context_slots, device=self.device, dtype=torch.long
        )
        self.sorted_keypoint_ids = torch.sort(self._candidate_body_ids).values

        if self.keypoint_body_ids_tensor is None:
            raise RuntimeError("Online compliant IK requires configured keypoints.")
        keypoint_lookup = {
            int(body_id): index
            for index, body_id in enumerate(self.keypoint_body_ids_tensor.tolist())
        }
        try:
            candidate_keypoint_ids = [
                keypoint_lookup[int(body_id)] for body_id in candidate_ids
            ]
        except KeyError as exc:
            raise RuntimeError(
                "Every online force body must also be a reference keypoint."
            ) from exc
        self._candidate_keypoint_ids = torch.as_tensor(
            candidate_keypoint_ids, device=self.device, dtype=torch.long
        )
        # new change
        # Offline SoftMimic only lists the two wrists as forceable links.  V2
        # solves a wrist request in the corresponding seven arm joints instead
        # of allowing the waist/legs/opposite arm to absorb the displacement.
        self._candidate_local_joint_masks = torch.zeros(
            len(candidate_names), self.num_joints, device=self.device, dtype=torch.bool
        )
        for slot, body_name in enumerate(candidate_names):
            if body_name.startswith("left_"):
                self._candidate_local_joint_masks[
                    slot, list(G1_SOFTMIMIC_LEFT_ARM_ACTION_IDS)
                ] = True
            elif body_name.startswith("right_"):
                self._candidate_local_joint_masks[
                    slot, list(G1_SOFTMIMIC_RIGHT_ARM_ACTION_IDS)
                ] = True

        # new change
        # The first online prototype used PhysX Jacobians from the *measured*
        # robot pose and then added the solution to q_ref.  That mixes two
        # configurations and can turn an arm request into whole-body motion.
        # V2 instead builds the same G1 MJCF used by the released augmentation
        # and evaluates batched FK/Jacobians directly at q_ref.
        self._reference_kinematics_enabled = (
            cfg.reference_kinematics_backend == "pytorch_mjcf"
        )
        self._reference_serial_chains: dict[str, object] = {}
        self._reference_chain_action_ids: dict[str, tuple[int, ...]] = {}
        self._reference_chain_keypoint_ids: dict[str, tuple[int, ...]] = {}
        if self._reference_kinematics_enabled:
            self._setup_reference_kinematics()

        foot_ids, _ = self.robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        pelvis_ids, _ = self.robot.find_bodies(["pelvis"], preserve_order=True)
        self._foot_body_ids = torch.as_tensor(
            foot_ids, device=self.device, dtype=torch.long
        )
        self._pelvis_body_id = int(pelvis_ids[0])

        self._event_time = torch.zeros(self.num_envs, device=self.device)
        self._ramp_time = torch.ones(self.num_envs, device=self.device)
        self._hold_time = torch.ones(self.num_envs, device=self.device)
        self._sampled_body_slot = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.selected_body_ids = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.selected_keypoint_ids = torch.zeros_like(self.selected_body_ids)
        self._sampled_peak_force_world = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._accepted_peak_force_world = torch.zeros_like(
            self._sampled_peak_force_world
        )
        self._accepted_peak_displacement_world = torch.zeros_like(
            self._sampled_peak_force_world
        )
        self._peak_joint_delta = torch.zeros(
            self.num_envs, self.num_joints, device=self.device
        )
        self._peak_keypoint_delta = torch.zeros(
            self.num_envs,
            len(self.keypoint_body_ids),
            3,
            device=self.device,
        )
        self._event_stiffness = torch.full(
            (self.num_envs, 1),
            float(cfg.stiffness_range_n_per_m[1]),
            device=self.device,
        )
        self._solution_pending = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.ik_feasible_scale = torch.zeros(self.num_envs, device=self.device)
        self.ik_active_residual_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self.ik_max_foot_motion_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self.ik_com_residual_m = torch.zeros(self.num_envs, device=self.device)
        self.ik_direction_cosine = torch.zeros(self.num_envs, device=self.device)
        self.ik_active_relative_residual = torch.zeros(
            self.num_envs, device=self.device
        )
        self.ik_peak_joint_speed_rad_s = torch.zeros(
            self.num_envs, device=self.device
        )
        self.ik_solve_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.ik_accept_count = torch.zeros_like(self.ik_solve_count)
        # new change
        # Aggregate each forceable link independently.  A combined acceptance
        # rate can hide an asymmetric left/right IK or joint-limit problem.
        num_force_slots = len(cfg.force_body_names)
        self._solve_count_by_slot = torch.zeros(
            num_force_slots, device=self.device, dtype=torch.long
        )
        self._accept_count_by_slot = torch.zeros_like(self._solve_count_by_slot)
        self._max_force_by_slot = torch.zeros(num_force_slots, device=self.device)
        self._max_joint_delta_by_slot = torch.zeros_like(self._max_force_by_slot)
        self._max_keypoint_delta_by_slot = torch.zeros_like(self._max_force_by_slot)
        self._max_residual_by_slot = torch.zeros_like(self._max_force_by_slot)
        self._accepted_direction_sum = torch.zeros((), device=self.device)
        self._accepted_direction_count = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self._accepted_direction_min = torch.ones((), device=self.device)
        self._max_active_residual_seen = torch.zeros((), device=self.device)
        self._max_foot_motion_seen = torch.zeros((), device=self.device)
        self._max_com_residual_seen = torch.zeros((), device=self.device)
        self._max_joint_speed_seen = torch.zeros((), device=self.device)
        # new change
        self._max_joint_delta_seen = torch.zeros((), device=self.device)
        self._max_keypoint_delta_seen = torch.zeros((), device=self.device)
        # new change
        self._max_accepted_force_seen = torch.zeros((), device=self.device)
        # new change
        # Keep generated-reference feasibility separate from closed-loop
        # execution quality.  The latter compares measured keypoints with the
        # accepted online reference while force is active.
        self.actual_active_link_error_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self.actual_keypoint_mean_error_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self._actual_active_error_sum = torch.zeros((), device=self.device)
        self._actual_active_error_count = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self._actual_active_error_max = torch.zeros((), device=self.device)
        # new change
        self._actual_active_error_sum_by_slot = torch.zeros(
            num_force_slots, device=self.device
        )
        self._actual_active_error_count_by_slot = torch.zeros(
            num_force_slots, device=self.device, dtype=torch.long
        )
        self._actual_active_error_max_by_slot = torch.zeros(
            num_force_slots, device=self.device
        )
        # new change
        self.actual_root_xy_drift_m = torch.zeros(self.num_envs, device=self.device)
        self._actual_root_xy_drift_max = torch.zeros((), device=self.device)
        # new change
        self._actual_root_xy_drift_sum = torch.zeros((), device=self.device)
        self._actual_root_xy_drift_count = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        # new change
        # Establish one immutable world/reference alignment per episode.  A
        # dynamic root-local transform hides sliding, while no alignment at all
        # mistakes the motion-file origin for physical translation.
        self._world_reference_alignment = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._world_reference_alignment_valid = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        # new change
        # Store the measured robot root at the beginning of each episode.
        # This is deliberately independent of motion-file root translation:
        # physical base sliding must be measured in the simulator world frame.
        self._robot_root_episode_anchor = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        # new change
        # A stationary task must distinguish harmless pelvis sway from actual
        # support-foot translation.  Anchor both feet in world XY per episode.
        self._robot_foot_episode_anchor_xy = torch.zeros(
            self.num_envs, 2, 2, device=self.device
        )
        self.actual_foot_xy_slip_m = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self._actual_foot_xy_slip_max = torch.zeros((), device=self.device)
        # new change
        self._actual_foot_xy_slip_sum = torch.zeros((), device=self.device)
        self._actual_foot_xy_slip_count = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        # new change
        # Physical closed-loop stability metrics.  Support is the midpoint of
        # the two foot-link COMs (not a force-derived CoP), matching the debug
        # marker used in the validation videos.
        self.actual_com_support_xy_distance_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self._actual_com_support_xy_distance_sum = torch.zeros(
            (), device=self.device
        )
        self._actual_com_support_xy_distance_count = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self._actual_com_support_xy_distance_max = torch.zeros(
            (), device=self.device
        )
        self.actual_root_tilt_rad = torch.zeros(self.num_envs, device=self.device)
        self._actual_root_tilt_sum = torch.zeros((), device=self.device)
        self._actual_root_tilt_count = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self._actual_root_tilt_max = torch.zeros((), device=self.device)
        self._actual_all_keypoint_error_sum = torch.zeros((), device=self.device)
        self._actual_all_keypoint_error_count = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self._actual_all_keypoint_error_max = torch.zeros((), device=self.device)

        self.force_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_displacement_world = torch.zeros_like(self.force_world)
        self.desired_stiffness = self._event_stiffness.clone()
        self.force_body_onehot = torch.zeros(
            self.num_envs, int(cfg.force_body_context_size), device=self.device
        )
        self.online_phase = torch.zeros(self.num_envs, 2, device=self.device)
        self.active_force_mask = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.body_ids = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self.forcefield_forces_b = torch.zeros_like(self.force_world)
        self.forcefield_torques_b = torch.zeros_like(self.force_world)

        self._online_dof_pos = self._dof_pos.clone()
        self._online_dof_vel = self._dof_vel.clone()
        self._online_keypoints = (
            None if self._keypoints is None else self._keypoints.clone()
        )
        self._online_keypoint_rotations = (
            None
            if self._keypoint_rotations is None
            else self._keypoint_rotations.clone()
        )

        self.metrics["online_ik_feasible_scale"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["online_ik_active_residual_m"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["online_ik_max_foot_motion_m"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["online_ik_com_residual_m"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["online_force_active"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["online_ik_direction_cosine"] = torch.zeros(
            self.num_envs, device=self.device
        )

        self._joint_delta_limits = self._make_joint_delta_limits()
        self._posture_diagonal = self._make_posture_diagonal()
        self._online_ready = True
        self._sample_events(self._env_ids)
        self._build_online_reference()

    def _uniform(
        self, count: int, value_range: tuple[float, float]
    ) -> torch.Tensor:
        low, high = value_range
        return low + (high - low) * torch.rand(
            count, device=self.device, generator=self.torch_gen
        )

    # new change
    def _setup_reference_kinematics(self) -> None:
        """Load the released G1 MJCF as two GPU serial arm chains."""

        try:
            import pytorch_kinematics as pk
        except ImportError as exc:
            raise RuntimeError(
                "ArmLocalIKV2 requires pytorch_kinematics in the IsaacLab environment."
            ) from exc

        mjcf_path = Path(self.cfg.reference_mjcf_path).expanduser()
        if not mjcf_path.is_absolute():
            for parent in Path(__file__).resolve().parents:
                candidate = parent / mjcf_path
                if candidate.is_file():
                    mjcf_path = candidate
                    break
        if not mjcf_path.is_file():
            raise FileNotFoundError(
                f"Reference MJCF does not exist: {self.cfg.reference_mjcf_path}"
            )
        mesh_dir = mjcf_path.parent / "meshes"
        assets = {
            f"meshes/{path.name}": path.read_bytes()
            for path in mesh_dir.iterdir()
            if path.is_file()
        }
        full_chain = pk.build_chain_from_mjcf(
            mjcf_path.read_text(), assets=assets
        )
        dtype = self.robot.data.joint_pos.dtype
        for side, arm_ids, keypoint_names in (
            (
                "left",
                G1_SOFTMIMIC_LEFT_ARM_ACTION_IDS,
                ("left_shoulder_yaw_link", "left_elbow_link", "left_wrist_yaw_link"),
            ),
            (
                "right",
                G1_SOFTMIMIC_RIGHT_ARM_ACTION_IDS,
                ("right_shoulder_yaw_link", "right_elbow_link", "right_wrist_yaw_link"),
            ),
        ):
            chain = pk.SerialChain(
                full_chain, f"{side}_wrist_yaw_link", "pelvis"
            ).to(dtype=dtype, device=self.device)
            expected = (
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
                *(self.robot.joint_names[self.control_joint_ids[index]] for index in arm_ids),
            )
            if tuple(chain.get_joint_parameter_names()) != expected:
                raise RuntimeError(
                    f"{side} reference chain order mismatch: "
                    f"{chain.get_joint_parameter_names()} != {expected}"
                )
            self._reference_serial_chains[side] = chain
            self._reference_chain_action_ids[side] = (
                2,
                5,
                8,
                *arm_ids,
            )
            self._reference_chain_keypoint_ids[side] = tuple(
                self.keypoint_body_ids.index(
                    self.robot.find_bodies([name], preserve_order=True)[0][0]
                )
                for name in keypoint_names
            )

    def _make_joint_delta_limits(self) -> torch.Tensor:
        limits = torch.full(
            (self.num_joints,),
            float(self.cfg.hip_delta_limit_rad),
            device=self.device,
        )
        limits[[9, 10]] = float(self.cfg.knee_delta_limit_rad)
        limits[[13, 14, 17, 18]] = float(self.cfg.ankle_delta_limit_rad)
        limits[[2, 5, 8]] = float(self.cfg.waist_delta_limit_rad)
        arm_ids = list(
            G1_SOFTMIMIC_LEFT_ARM_ACTION_IDS
            + G1_SOFTMIMIC_RIGHT_ARM_ACTION_IDS
        )
        limits[arm_ids] = float(self.cfg.arm_delta_limit_rad)
        return limits

    def _make_posture_diagonal(self) -> torch.Tensor:
        diagonal = torch.full(
            (self.num_joints,),
            float(self.cfg.posture_weight + self.cfg.ik_damping**2),
            device=self.device,
        )
        diagonal[[2, 5, 8, 9, 10]] += float(
            self.cfg.waist_knee_posture_weight
        )
        return diagonal

    def _sample_events(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        count = env_ids.numel()
        wait = self._uniform(count, self.cfg.wait_time_range_s)
        self._event_time[env_ids] = -wait
        self._ramp_time[env_ids] = self._uniform(
            count, self.cfg.ramp_time_range_s
        )
        self._hold_time[env_ids] = self._uniform(
            count, self.cfg.hold_time_range_s
        )

        slots = torch.randint(
            0,
            len(self._candidate_body_ids),
            (count,),
            device=self.device,
            generator=self.torch_gen,
        )
        self._sampled_body_slot[env_ids] = slots
        self.selected_body_ids[env_ids] = self._candidate_body_ids[slots]
        self.selected_keypoint_ids[env_ids] = self._candidate_keypoint_ids[slots]

        direction = torch.randn(
            count, 3, device=self.device, generator=self.torch_gen
        )
        direction[:, 2] *= float(self.cfg.vertical_direction_scale)
        direction = direction / torch.clamp(
            torch.linalg.vector_norm(direction, dim=-1, keepdim=True), min=1.0e-6
        )
        # The torso entry follows the original augmentation's downward-only
        # convention; arm links retain random low-vertical directions.
        # torso_mask = slots == 0
        # direction[torso_mask] = torch.tensor(
        #     [0.0, 0.0, -1.0], device=self.device
        # )
        # new change
        # Slot zero is not necessarily the torso.  In wrist-only V2 it is the
        # left wrist, so the legacy positional check accidentally forced every
        # left-hand event straight downward.  Apply this convention by body
        # name only and preserve random directions for both wrists.
        torso_slots = tuple(
            index
            for index, body_name in enumerate(self.cfg.force_body_names)
            if body_name == "torso_link"
        )
        if torso_slots:
            torso_mask = torch.zeros(count, device=self.device, dtype=torch.bool)
            for torso_slot in torso_slots:
                torso_mask |= slots == torso_slot
            direction[torso_mask] = torch.tensor(
                [0.0, 0.0, -1.0], device=self.device
            )
        magnitude = self._uniform(count, self.cfg.force_range_n).unsqueeze(-1)
        stiffness = self._uniform(
            count, self.cfg.stiffness_range_n_per_m
        ).unsqueeze(-1)
        peak_force = direction * magnitude
        raw_displacement = peak_force / stiffness
        displacement_norm = torch.linalg.vector_norm(
            raw_displacement, dim=-1, keepdim=True
        )
        displacement_scale = torch.clamp(
            float(self.cfg.max_displacement_m)
            / torch.clamp(displacement_norm, min=1.0e-8),
            max=1.0,
        )
        peak_force *= displacement_scale

        self._sampled_peak_force_world[env_ids] = peak_force
        self._accepted_peak_force_world[env_ids] = 0.0
        self._accepted_peak_displacement_world[env_ids] = 0.0
        self._event_stiffness[env_ids] = stiffness
        self._peak_joint_delta[env_ids] = 0.0
        self._peak_keypoint_delta[env_ids] = 0.0
        self._solution_pending[env_ids] = True
        self.ik_feasible_scale[env_ids] = 0.0
        self.ik_active_residual_m[env_ids] = 0.0
        self.ik_max_foot_motion_m[env_ids] = 0.0
        self.ik_com_residual_m[env_ids] = 0.0
        self.ik_direction_cosine[env_ids] = 0.0
        self.ik_active_relative_residual[env_ids] = 0.0
        self.ik_peak_joint_speed_rad_s[env_ids] = 0.0

    def _body_jacobian_w(
        self, env_ids: torch.Tensor, body_ids: torch.Tensor
    ) -> torch.Tensor:
        all_jacobians = self.robot.root_physx_view.get_jacobians()
        if self.robot.is_fixed_base:
            jacobian_body_ids = body_ids - 1
        else:
            jacobian_body_ids = body_ids
        return all_jacobians[
            env_ids,
            jacobian_body_ids,
            :,
        ].index_select(-1, self._jacobian_joint_columns)

    def _all_body_jacobians_w(self, env_ids: torch.Tensor) -> torch.Tensor:
        jacobians = self.robot.root_physx_view.get_jacobians()[env_ids]
        jacobians = jacobians.index_select(-1, self._jacobian_joint_columns)
        if not self.robot.is_fixed_base:
            return jacobians
        root_zeros = torch.zeros(
            jacobians.shape[0],
            1,
            6,
            jacobians.shape[-1],
            device=self.device,
            dtype=jacobians.dtype,
        )
        return torch.cat((root_zeros, jacobians), dim=1)

    @staticmethod
    def _add_task(
        hessian: torch.Tensor,
        rhs: torch.Tensor,
        jacobian: torch.Tensor,
        target: torch.Tensor,
        weight: float,
    ) -> None:
        jt = jacobian.transpose(-1, -2)
        hessian.add_(float(weight) * torch.bmm(jt, jacobian))
        rhs.add_(
            float(weight)
            * torch.bmm(jt, target.unsqueeze(-1)).squeeze(-1)
        )

    def _whole_body_com_jacobian_w(
        self, env_ids: torch.Tensor
    ) -> torch.Tensor:
        body_jacobians = self._all_body_jacobians_w(env_ids)
        body_pos = self.robot.data.body_pos_w[env_ids]
        body_com_pos = self.robot.data.body_com_pos_w[env_ids]
        offset = body_com_pos - body_pos
        omega_columns = body_jacobians[:, :, 3:, :].permute(0, 1, 3, 2)
        point_term = torch.cross(
            omega_columns, offset.unsqueeze(2).expand_as(omega_columns), dim=-1
        ).permute(0, 1, 3, 2)
        linear = body_jacobians[:, :, :3, :] + point_term
        # new change
        # IsaacLab keeps default inertial properties on CPU in this version,
        # whereas live poses/Jacobians are CUDA tensors.
        masses = self.robot.data.default_mass.to(
            device=self.device, dtype=linear.dtype
        )[env_ids]
        return (linear * masses[:, :, None, None]).sum(dim=1) / torch.clamp(
            masses.sum(dim=1)[:, None, None], min=1.0e-6
        )

    # new change
    def _solve_reference_arm_events(self, env_ids: torch.Tensor) -> None:
        """Iterative wrist IK at q_ref; only the selected seven-joint arm moves."""

        count = env_ids.numel()
        dtype = self.robot.data.joint_pos.dtype
        horizon_index = 1 if self._dof_pos.shape[1] > 1 else 0
        nominal_all = self._dof_pos[env_ids, horizon_index].to(dtype)
        root_quat_wxyz = self._root_rot[env_ids, horizon_index].index_select(
            -1,
            torch.tensor([3, 0, 1, 2], device=self.device, dtype=torch.long),
        ).to(dtype)
        requested_world = (
            self._sampled_peak_force_world[env_ids]
            / self._event_stiffness[env_ids]
        ).to(dtype)
        requested_local = math_utils.quat_rotate_inverse(
            root_quat_wxyz, requested_world
        )

        final_delta = torch.zeros(
            count, self.num_joints, device=self.device, dtype=dtype
        )
        exact_keypoint_delta_local = torch.zeros(
            count,
            len(self.keypoint_body_ids),
            3,
            device=self.device,
            dtype=dtype,
        )
        predicted_local = torch.zeros(count, 3, device=self.device, dtype=dtype)

        for side_index, side in enumerate(("left", "right")):
            side_rows = torch.where(self._sampled_body_slot[env_ids] == side_index)[0]
            if side_rows.numel() == 0:
                continue
            chain = self._reference_serial_chains[side]
            action_ids = self._reference_chain_action_ids[side]
            arm_action_ids = action_ids[3:]
            q_nominal = nominal_all[side_rows][:, action_ids]
            q_work = q_nominal.clone()
            nominal_frames = chain.forward_kinematics(q_nominal, end_only=False)
            nominal_wrist = nominal_frames[f"{side}_wrist_yaw_link"].get_matrix()[
                :, :3, 3
            ]
            target_wrist = nominal_wrist + requested_local[side_rows]
            eye = torch.eye(7, device=self.device, dtype=dtype).expand(
                side_rows.numel(), -1, -1
            )
            for _ in range(int(self.cfg.reference_ik_iterations)):
                wrist = chain.forward_kinematics(q_work).get_matrix()[:, :3, 3]
                jacobian = chain.jacobian(q_work)[:, :3, 3:]
                error = target_wrist - wrist
                hessian = (
                    float(self.cfg.active_link_position_weight)
                    * torch.bmm(jacobian.transpose(1, 2), jacobian)
                    + float(self.cfg.ik_damping**2 + self.cfg.posture_weight) * eye
                )
                rhs = (
                    float(self.cfg.active_link_position_weight)
                    * torch.bmm(
                        jacobian.transpose(1, 2), error.unsqueeze(-1)
                    ).squeeze(-1)
                    - float(self.cfg.posture_weight) * (q_work[:, 3:] - q_nominal[:, 3:])
                )
                increment = torch.linalg.solve(hessian, rhs.unsqueeze(-1)).squeeze(-1)
                q_work[:, 3:] += increment.clamp(-0.15, 0.15)
                arm_delta = (q_work[:, 3:] - q_nominal[:, 3:]).clamp(
                    -float(self.cfg.arm_delta_limit_rad),
                    float(self.cfg.arm_delta_limit_rad),
                )
                q_work[:, 3:] = q_nominal[:, 3:] + arm_delta

            # Respect the exact Isaac articulation limits in canonical order.
            canonical_arm_ids = torch.as_tensor(
                arm_action_ids, device=self.device, dtype=torch.long
            )
            articulation_ids = self._canonical_joint_ids[canonical_arm_ids]
            limits = self.robot.data.soft_joint_pos_limits[
                env_ids[side_rows, None], articulation_ids[None, :]
            ]
            q_work[:, 3:] = torch.maximum(
                torch.minimum(q_work[:, 3:], limits[..., 1]), limits[..., 0]
            )
            raw_delta = q_work[:, 3:] - q_nominal[:, 3:]

            velocity_limits = self.robot.data.soft_joint_vel_limits[
                env_ids[side_rows, None], articulation_ids[None, :]
            ].to(dtype)
            nominal_speed = torch.abs(
                self._dof_vel[env_ids[side_rows], horizon_index][:, arm_action_ids]
            )
            available_speed = torch.clamp(
                float(self.cfg.joint_velocity_limit_fraction) * velocity_limits
                - nominal_speed,
                min=0.0,
            )
            correction_speed = torch.abs(raw_delta) / self._ramp_time[
                env_ids[side_rows], None
            ]
            velocity_scale = torch.where(
                correction_speed > 1.0e-8,
                available_speed / torch.clamp(correction_speed, min=1.0e-8),
                torch.ones_like(correction_speed),
            ).amin(dim=1).clamp(max=1.0)
            raw_delta *= velocity_scale[:, None]
            q_work[:, 3:] = q_nominal[:, 3:] + raw_delta
            final_delta[side_rows[:, None], canonical_arm_ids[None, :]] = raw_delta

            adapted_frames = chain.forward_kinematics(q_work, end_only=False)
            adapted_wrist = adapted_frames[f"{side}_wrist_yaw_link"].get_matrix()[
                :, :3, 3
            ]
            predicted_local[side_rows] = adapted_wrist - nominal_wrist
            for keypoint_slot, frame_name in zip(
                self._reference_chain_keypoint_ids[side],
                (
                    f"{side}_shoulder_yaw_link",
                    f"{side}_elbow_link",
                    f"{side}_wrist_yaw_link",
                ),
                strict=True,
            ):
                exact_keypoint_delta_local[side_rows, keypoint_slot] = (
                    adapted_frames[frame_name].get_matrix()[:, :3, 3]
                    - nominal_frames[frame_name].get_matrix()[:, :3, 3]
                )

        # The velocity limiter scales the accepted F/K target together with q.
        requested_norm = torch.linalg.vector_norm(requested_local, dim=-1)
        predicted_norm = torch.linalg.vector_norm(predicted_local, dim=-1)
        feasible_scale = torch.where(
            requested_norm > 1.0e-8,
            predicted_norm / torch.clamp(requested_norm, min=1.0e-8),
            torch.ones_like(requested_norm),
        ).clamp(max=1.0)
        accepted_target_local = requested_local * feasible_scale[:, None]
        active_residual = torch.linalg.vector_norm(
            predicted_local - accepted_target_local, dim=-1
        )
        accepted_norm = torch.linalg.vector_norm(accepted_target_local, dim=-1)
        relative_residual = torch.where(
            accepted_norm > 1.0e-8,
            active_residual / torch.clamp(accepted_norm, min=1.0e-8),
            torch.zeros_like(active_residual),
        )
        direction_cosine = (predicted_local * accepted_target_local).sum(dim=-1) / torch.clamp(
            predicted_norm * accepted_norm, min=1.0e-8
        )
        finite = torch.isfinite(final_delta).all(dim=-1)
        accepted = (
            finite
            & (feasible_scale >= float(self.cfg.min_feasible_scale))
            & (active_residual <= float(self.cfg.max_active_link_residual_m))
            & (relative_residual <= float(self.cfg.max_active_link_relative_residual))
            & (direction_cosine >= float(self.cfg.min_active_direction_cosine))
        )
        final_delta = torch.where(
            accepted[:, None], final_delta, torch.zeros_like(final_delta)
        )
        exact_keypoint_delta_local = torch.where(
            accepted[:, None, None],
            exact_keypoint_delta_local,
            torch.zeros_like(exact_keypoint_delta_local),
        )
        exact_keypoint_delta_world = math_utils.quat_rotate(
            root_quat_wxyz[:, None, :].expand(-1, len(self.keypoint_body_ids), -1).reshape(-1, 4),
            exact_keypoint_delta_local.reshape(-1, 3),
        ).reshape(count, len(self.keypoint_body_ids), 3)
        accepted_scale = torch.where(
            accepted, feasible_scale, torch.zeros_like(feasible_scale)
        )
        self._peak_joint_delta[env_ids] = final_delta
        self._peak_keypoint_delta[env_ids] = exact_keypoint_delta_world
        self._accepted_peak_force_world[env_ids] = (
            self._sampled_peak_force_world[env_ids] * accepted_scale[:, None]
        )
        self._accepted_peak_displacement_world[env_ids] = (
            requested_world * accepted_scale[:, None]
        )
        self.ik_feasible_scale[env_ids] = accepted_scale
        self.ik_active_residual_m[env_ids] = torch.where(
            accepted, active_residual, torch.zeros_like(active_residual)
        )
        self.ik_active_relative_residual[env_ids] = torch.where(
            accepted, relative_residual, torch.zeros_like(relative_residual)
        )
        self.ik_direction_cosine[env_ids] = torch.where(
            accepted, direction_cosine, torch.zeros_like(direction_cosine)
        )
        self.ik_max_foot_motion_m[env_ids] = 0.0
        self.ik_com_residual_m[env_ids] = 0.0
        peak_speed = (
            torch.abs(final_delta) / self._ramp_time[env_ids, None]
        ).amax(dim=1)
        self.ik_peak_joint_speed_rad_s[env_ids] = peak_speed
        self.ik_solve_count[env_ids] += 1
        self.ik_accept_count[env_ids] += accepted.long()
        # new change
        event_slots = self._sampled_body_slot[env_ids]
        self._solve_count_by_slot.scatter_add_(
            0, event_slots, torch.ones_like(event_slots)
        )
        self._accept_count_by_slot.scatter_add_(
            0, event_slots, accepted.long()
        )
        if torch.any(accepted):
            # new change
            self._max_accepted_force_seen = torch.maximum(
                self._max_accepted_force_seen,
                torch.linalg.vector_norm(
                    self._accepted_peak_force_world[env_ids][accepted], dim=-1
                ).max(),
            )
            self._accepted_direction_sum += direction_cosine[accepted].sum()
            self._accepted_direction_count += accepted.sum()
            self._accepted_direction_min = torch.minimum(
                self._accepted_direction_min, direction_cosine[accepted].min()
            )
            self._max_active_residual_seen = torch.maximum(
                self._max_active_residual_seen, active_residual[accepted].max()
            )
            self._max_joint_speed_seen = torch.maximum(
                self._max_joint_speed_seen, peak_speed[accepted].max()
            )
            self._max_joint_delta_seen = torch.maximum(
                self._max_joint_delta_seen,
                torch.abs(final_delta[accepted]).max(),
            )
            self._max_keypoint_delta_seen = torch.maximum(
                self._max_keypoint_delta_seen,
                torch.linalg.vector_norm(
                    exact_keypoint_delta_world[accepted], dim=-1
                ).max(),
            )
            # new change
            for slot in range(len(self.cfg.force_body_names)):
                slot_mask = accepted & (event_slots == slot)
                if torch.any(slot_mask):
                    self._max_force_by_slot[slot] = torch.maximum(
                        self._max_force_by_slot[slot],
                        torch.linalg.vector_norm(
                            self._accepted_peak_force_world[env_ids][slot_mask], dim=-1
                        ).max(),
                    )
                    self._max_joint_delta_by_slot[slot] = torch.maximum(
                        self._max_joint_delta_by_slot[slot],
                        torch.abs(final_delta[slot_mask]).max(),
                    )
                    self._max_keypoint_delta_by_slot[slot] = torch.maximum(
                        self._max_keypoint_delta_by_slot[slot],
                        torch.linalg.vector_norm(
                            exact_keypoint_delta_world[slot_mask], dim=-1
                        ).max(),
                    )
                    self._max_residual_by_slot[slot] = torch.maximum(
                        self._max_residual_by_slot[slot],
                        active_residual[slot_mask].max(),
                    )
        self._solution_pending[env_ids] = False

    def _solve_pending_events(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        # new change
        if self._reference_kinematics_enabled:
            self._solve_reference_arm_events(env_ids)
            return
        count = env_ids.numel()
        dtype = self.robot.data.joint_pos.dtype
        diagonal = self._posture_diagonal.to(dtype).expand(count, -1)
        hessian = torch.diag_embed(diagonal).clone()
        rhs = torch.zeros(count, self.num_joints, device=self.device, dtype=dtype)

        selected_bodies = self.selected_body_ids[env_ids]
        active_jacobian = self._body_jacobian_w(env_ids, selected_bodies)
        # new change
        if self.cfg.arm_local_ik_enabled:
            local_joint_mask = self._candidate_local_joint_masks[
                self._sampled_body_slot[env_ids]
            ]
            active_jacobian = active_jacobian * local_joint_mask[:, None, :]
        else:
            local_joint_mask = torch.ones(
                count, self.num_joints, device=self.device, dtype=torch.bool
            )
        requested_displacement = (
            self._sampled_peak_force_world[env_ids]
            / self._event_stiffness[env_ids]
        ).to(dtype)
        self._add_task(
            hessian,
            rhs,
            active_jacobian[:, :3],
            requested_displacement,
            self.cfg.active_link_position_weight,
        )

        repeated_env_ids = env_ids[:, None].expand(-1, 2).reshape(-1)
        repeated_foot_ids = self._foot_body_ids[None].expand(count, -1).reshape(-1)
        foot_jacobian = self._body_jacobian_w(
            repeated_env_ids, repeated_foot_ids
        ).reshape(count, 2, 6, self.num_joints)
        # new change
        if self.cfg.arm_local_ik_enabled:
            foot_jacobian = foot_jacobian * local_joint_mask[:, None, None, :]
        zero_foot = torch.zeros(count, 6, device=self.device, dtype=dtype)
        self._add_task(
            hessian,
            rhs,
            foot_jacobian[:, :, :3].reshape(count, 6, self.num_joints),
            zero_foot,
            self.cfg.foot_position_weight,
        )
        self._add_task(
            hessian,
            rhs,
            foot_jacobian[:, :, 3:].reshape(count, 6, self.num_joints),
            zero_foot,
            self.cfg.foot_orientation_weight,
        )

        keypoint_count = len(self.keypoint_body_ids)
        repeated_env_ids = env_ids[:, None].expand(-1, keypoint_count).reshape(-1)
        repeated_keypoint_ids = self.keypoint_body_ids_tensor[None].expand(
            count, -1
        ).reshape(-1)
        keypoint_jacobian = self._body_jacobian_w(
            repeated_env_ids, repeated_keypoint_ids
        ).reshape(count, keypoint_count, 6, self.num_joints)
        # new change
        if self.cfg.arm_local_ik_enabled:
            keypoint_jacobian = (
                keypoint_jacobian * local_joint_mask[:, None, None, :]
            )
        if self.cfg.exclude_active_link_from_keypoint_task:
            keypoint_jacobian[
                torch.arange(count, device=self.device),
                self.selected_keypoint_ids[env_ids],
            ] = 0.0
        zero_keypoint = torch.zeros(
            count, 3 * keypoint_count, device=self.device, dtype=dtype
        )
        self._add_task(
            hessian,
            rhs,
            keypoint_jacobian[:, :, :3].reshape(
                count, 3 * keypoint_count, self.num_joints
            ),
            zero_keypoint,
            self.cfg.keypoint_position_weight,
        )
        self._add_task(
            hessian,
            rhs,
            keypoint_jacobian[:, :, 3:].reshape(
                count, 3 * keypoint_count, self.num_joints
            ),
            zero_keypoint,
            self.cfg.keypoint_orientation_weight,
        )

        pelvis_ids = torch.full(
            (count,), self._pelvis_body_id, device=self.device, dtype=torch.long
        )
        pelvis_jacobian = self._body_jacobian_w(env_ids, pelvis_ids)
        # new change
        if self.cfg.arm_local_ik_enabled:
            pelvis_jacobian = pelvis_jacobian * local_joint_mask[:, None, :]
        self._add_task(
            hessian,
            rhs,
            pelvis_jacobian[:, 3:],
            torch.zeros(count, 3, device=self.device, dtype=dtype),
            self.cfg.pelvis_orientation_weight,
        )

        com_jacobian = self._whole_body_com_jacobian_w(env_ids)
        support_midpoint = self.robot.data.body_com_pos_w[
            env_ids[:, None], self._foot_body_ids[None], :
        ].mean(dim=1)
        application_pos = self.robot.data.body_pos_w[
            env_ids, selected_bodies
        ]
        moment = torch.cross(
            application_pos - support_midpoint,
            self._sampled_peak_force_world[env_ids],
            dim=-1,
        )
        total_mass = self.robot.data.default_mass.to(
            device=self.device, dtype=dtype
        )[env_ids].sum(dim=1)
        # new change
        if self.cfg.cop_aware_com_target_enabled:
            com_target = torch.stack(
                (-moment[:, 1], moment[:, 0]), dim=-1
            ) / torch.clamp(total_mass[:, None] * 9.81, min=1.0e-6)
        else:
            # ArmLocalIKV2 does not command the whole body to lean with the
            # disturbance.  Small COM changes caused by the moving arm remain
            # regularized toward zero.
            com_target = torch.zeros(count, 2, device=self.device, dtype=dtype)
        if self.cfg.arm_local_ik_enabled:
            com_jacobian = com_jacobian * local_joint_mask[:, None, :]
        self._add_task(
            hessian,
            rhs,
            com_jacobian[:, :2],
            com_target.to(dtype),
            self.cfg.com_xy_weight,
        )

        correction = torch.linalg.solve(hessian, rhs.unsqueeze(-1)).squeeze(-1)
        # new change
        correction = torch.where(local_joint_mask, correction, torch.zeros_like(correction))
        correction = torch.maximum(
            torch.minimum(correction, self._joint_delta_limits),
            -self._joint_delta_limits,
        )

        nominal = self._dof_pos[env_ids]
        joint_limits = self.robot.data.soft_joint_pos_limits[
            env_ids[:, None], self._canonical_joint_ids[None, :]
        ]
        lower_margin = (joint_limits[:, None, :, 0] - nominal).amax(dim=1)
        upper_margin = (joint_limits[:, None, :, 1] - nominal).amin(dim=1)
        feasible_lower = torch.maximum(-self._joint_delta_limits, lower_margin)
        feasible_upper = torch.minimum(self._joint_delta_limits, upper_margin)
        invalid_limit = feasible_lower > feasible_upper
        correction = torch.maximum(
            torch.minimum(correction, feasible_upper), feasible_lower
        )
        correction[invalid_limit] = 0.0

        predicted_active = torch.bmm(
            active_jacobian[:, :3], correction.unsqueeze(-1)
        ).squeeze(-1)
        active_residual = torch.linalg.vector_norm(
            predicted_active - requested_displacement, dim=-1
        )
        requested_norm = torch.linalg.vector_norm(
            requested_displacement, dim=-1
        )
        predicted_norm = torch.linalg.vector_norm(predicted_active, dim=-1)
        direction_cosine = (
            predicted_active * requested_displacement
        ).sum(dim=-1) / torch.clamp(
            predicted_norm * requested_norm, min=1.0e-8
        )
        zero_request = requested_norm <= 1.0e-8
        direction_cosine = torch.where(
            zero_request, torch.ones_like(direction_cosine), direction_cosine
        )
        relative_residual = torch.where(
            zero_request,
            torch.zeros_like(active_residual),
            active_residual / torch.clamp(requested_norm, min=1.0e-8),
        )
        predicted_feet = torch.einsum(
            "nbij,nj->nbi", foot_jacobian[:, :, :3], correction
        )
        foot_motion = torch.linalg.vector_norm(predicted_feet, dim=-1).amax(dim=1)
        predicted_com = torch.bmm(
            com_jacobian[:, :2], correction.unsqueeze(-1)
        ).squeeze(-1)
        com_residual = torch.linalg.vector_norm(
            predicted_com - com_target, dim=-1
        )

        one = torch.ones_like(active_residual)
        scale = torch.minimum(
            one,
            float(self.cfg.max_active_link_residual_m)
            / torch.clamp(active_residual, min=1.0e-8),
        )
        scale = torch.minimum(
            scale,
            float(self.cfg.max_foot_motion_m)
            / torch.clamp(foot_motion, min=1.0e-8),
        )
        scale = torch.minimum(
            scale,
            float(self.cfg.max_com_residual_m)
            / torch.clamp(com_residual, min=1.0e-8),
        )
        # new change
        # Mirror the original Mink velocity-limit task explicitly.  The force,
        # Cartesian displacement and q correction share this scale, so a
        # velocity-infeasible reference can never be paired with a full wrench.
        velocity_limits = self.robot.data.soft_joint_vel_limits.to(
            device=self.device, dtype=dtype
        )[env_ids[:, None], self._canonical_joint_ids[None, :]]
        nominal_speed = torch.abs(self._dof_vel[env_ids]).amax(dim=1)
        available_speed = torch.clamp(
            float(self.cfg.joint_velocity_limit_fraction) * velocity_limits
            - nominal_speed,
            min=0.0,
        )
        correction_speed = torch.abs(correction) / self._ramp_time[env_ids, None]
        per_joint_velocity_scale = torch.where(
            correction_speed > 1.0e-8,
            available_speed / torch.clamp(correction_speed, min=1.0e-8),
            torch.ones_like(correction_speed),
        )
        velocity_scale = torch.clamp(
            per_joint_velocity_scale.amin(dim=1), max=1.0
        )
        scale = torch.minimum(scale, velocity_scale)
        finite = torch.isfinite(correction).all(dim=-1) & torch.isfinite(scale)
        accepted = (
            finite
            & (scale >= float(self.cfg.min_feasible_scale))
            & (
                relative_residual
                <= float(self.cfg.max_active_link_relative_residual)
            )
            & (
                direction_cosine
                >= float(self.cfg.min_active_direction_cosine)
            )
        )
        scale = torch.where(accepted, scale, torch.zeros_like(scale))
        correction = torch.where(
            accepted[:, None], correction * scale[:, None], torch.zeros_like(correction)
        )

        predicted_keypoints = torch.einsum(
            "nbij,nj->nbi", keypoint_jacobian[:, :, :3], correction
        )
        peak_force = self._sampled_peak_force_world[env_ids] * scale[:, None]
        self._accepted_peak_force_world[env_ids] = peak_force
        self._accepted_peak_displacement_world[env_ids] = (
            peak_force / self._event_stiffness[env_ids]
        )
        self._peak_joint_delta[env_ids] = correction
        self._peak_keypoint_delta[env_ids] = predicted_keypoints
        self.ik_feasible_scale[env_ids] = scale
        self.ik_active_residual_m[env_ids] = active_residual * scale
        self.ik_max_foot_motion_m[env_ids] = foot_motion * scale
        self.ik_com_residual_m[env_ids] = com_residual * scale
        self.ik_active_relative_residual[env_ids] = torch.where(
            accepted, relative_residual, torch.zeros_like(relative_residual)
        )
        final_joint_speed = (
            torch.abs(correction) / self._ramp_time[env_ids, None]
        ).amax(dim=1)
        self.ik_peak_joint_speed_rad_s[env_ids] = final_joint_speed
        selected_keypoint_slots = self.selected_keypoint_ids[env_ids]
        predicted_selected = predicted_keypoints[
            torch.arange(count, device=self.device), selected_keypoint_slots
        ]
        accepted_displacement = self._accepted_peak_displacement_world[env_ids]
        self.ik_direction_cosine[env_ids] = (
            predicted_selected * accepted_displacement
        ).sum(dim=-1) / torch.clamp(
            torch.linalg.vector_norm(predicted_selected, dim=-1)
            * torch.linalg.vector_norm(accepted_displacement, dim=-1),
            min=1.0e-8,
        )
        self.ik_solve_count[env_ids] += 1
        self.ik_accept_count[env_ids] += accepted.long()
        # new change
        event_slots = self._sampled_body_slot[env_ids]
        self._solve_count_by_slot.scatter_add_(
            0, event_slots, torch.ones_like(event_slots)
        )
        self._accept_count_by_slot.scatter_add_(
            0, event_slots, accepted.long()
        )
        if torch.any(accepted):
            # new change
            self._max_accepted_force_seen = torch.maximum(
                self._max_accepted_force_seen,
                torch.linalg.vector_norm(peak_force[accepted], dim=-1).max(),
            )
            accepted_direction = direction_cosine[accepted]
            self._accepted_direction_sum += accepted_direction.sum()
            self._accepted_direction_count += accepted_direction.numel()
            self._accepted_direction_min = torch.minimum(
                self._accepted_direction_min, accepted_direction.min()
            )
            self._max_active_residual_seen = torch.maximum(
                self._max_active_residual_seen,
                (active_residual[accepted] * scale[accepted]).max(),
            )
            self._max_foot_motion_seen = torch.maximum(
                self._max_foot_motion_seen,
                (foot_motion[accepted] * scale[accepted]).max(),
            )
            self._max_com_residual_seen = torch.maximum(
                self._max_com_residual_seen,
                (com_residual[accepted] * scale[accepted]).max(),
            )
            self._max_joint_speed_seen = torch.maximum(
                self._max_joint_speed_seen, final_joint_speed[accepted].max()
            )
            # new change
            for slot in range(len(self.cfg.force_body_names)):
                slot_mask = accepted & (event_slots == slot)
                if torch.any(slot_mask):
                    self._max_force_by_slot[slot] = torch.maximum(
                        self._max_force_by_slot[slot],
                        torch.linalg.vector_norm(peak_force[slot_mask], dim=-1).max(),
                    )
                    self._max_joint_delta_by_slot[slot] = torch.maximum(
                        self._max_joint_delta_by_slot[slot],
                        torch.abs(correction[slot_mask]).max(),
                    )
                    self._max_keypoint_delta_by_slot[slot] = torch.maximum(
                        self._max_keypoint_delta_by_slot[slot],
                        torch.linalg.vector_norm(
                            predicted_keypoints[slot_mask], dim=-1
                        ).max(),
                    )
                    self._max_residual_by_slot[slot] = torch.maximum(
                        self._max_residual_by_slot[slot],
                        (active_residual[slot_mask] * scale[slot_mask]).max(),
                    )
        self._solution_pending[env_ids] = False

    def _envelope(
        self, event_time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ramp = self._ramp_time[:, None]
        hold = self._hold_time[:, None]
        time = event_time
        up = (time / ramp).clamp(0.0, 1.0)
        down = (1.0 - (time - ramp - hold) / ramp).clamp(0.0, 1.0)
        envelope = torch.where(
            time < 0.0,
            torch.zeros_like(time),
            torch.where(time < ramp, up, torch.where(time < ramp + hold, 1.0, down)),
        )
        envelope = torch.where(
            time < 2.0 * ramp + hold, envelope, torch.zeros_like(envelope)
        )
        derivative = torch.where(
            (time >= 0.0) & (time < ramp),
            1.0 / ramp,
            torch.where(
                (time >= ramp + hold) & (time < 2.0 * ramp + hold),
                -1.0 / ramp,
                torch.zeros_like(time),
            ),
        )
        # new change
        # The original linear profile remains the default for checkpoint/task
        # compatibility.  The half-cosine A/B has exactly the same wait, ramp,
        # hold, duration and peak force, but its slope is zero at force onset,
        # full-force entry, release onset and release completion.
        profile = getattr(self.cfg, "force_envelope_profile", "linear")
        if profile == "linear":
            return envelope, derivative
        if profile != "half_cosine":
            raise ValueError(
                "Unsupported force_envelope_profile "
                f"'{profile}'. Expected 'linear' or 'half_cosine'."
            )

        up_phase = (time / ramp).clamp(0.0, 1.0)
        down_phase = ((time - ramp - hold) / ramp).clamp(0.0, 1.0)
        up_cosine = 0.5 - 0.5 * torch.cos(torch.pi * up_phase)
        down_cosine = 0.5 + 0.5 * torch.cos(torch.pi * down_phase)
        up_cosine_derivative = (
            0.5 * torch.pi / ramp * torch.sin(torch.pi * up_phase)
        )
        down_cosine_derivative = (
            -0.5 * torch.pi / ramp * torch.sin(torch.pi * down_phase)
        )
        envelope = torch.where(
            time < 0.0,
            torch.zeros_like(time),
            torch.where(
                time < ramp,
                up_cosine,
                torch.where(time < ramp + hold, 1.0, down_cosine),
            ),
        )
        envelope = torch.where(
            time < 2.0 * ramp + hold, envelope, torch.zeros_like(envelope)
        )
        derivative = torch.where(
            (time >= 0.0) & (time < ramp),
            up_cosine_derivative,
            torch.where(
                (time >= ramp + hold) & (time < 2.0 * ramp + hold),
                down_cosine_derivative,
                torch.zeros_like(time),
            ),
        )
        return envelope, derivative

    def _build_online_reference(self) -> None:
        future_offsets = torch.arange(
            self._horizon_steps, device=self.device, dtype=self._event_time.dtype
        ) * float(self.future_dt)
        horizon_time = self._event_time[:, None] + future_offsets[None]
        envelope, envelope_derivative = self._envelope(horizon_time)

        self._online_dof_pos = self._dof_pos + (
            envelope[:, :, None] * self._peak_joint_delta[:, None]
        )
        self._online_dof_vel = self._dof_vel + (
            envelope_derivative[:, :, None] * self._peak_joint_delta[:, None]
        )
        if self._keypoints is not None:
            self._online_keypoints = self._keypoints.clone()
            self._online_keypoints[..., :3] += (
                envelope[:, :, None, None]
                * self._peak_keypoint_delta[:, None]
            )
        if self._keypoint_rotations is not None:
            self._online_keypoint_rotations = self._keypoint_rotations.clone()

        current_envelope = envelope[:, 0]
        self.force_world = (
            self._accepted_peak_force_world * current_envelope[:, None]
        )
        self.target_displacement_world = (
            self._accepted_peak_displacement_world * current_envelope[:, None]
        )
        self.desired_stiffness = self._event_stiffness
        # new change
        # A rejected IK event must never be exposed as an active force event.
        # The rejected target remains visible through the dedicated purple
        # marker, while physics and policy observations receive zero wrench.
        self.active_force_mask = (
            (current_envelope > 1.0e-6) & (self.ik_feasible_scale > 0.0)
        )
        self.body_ids = torch.where(
            self.active_force_mask,
            self.selected_body_ids,
            torch.full_like(self.selected_body_ids, -1),
        )
        self.force_body_onehot.zero_()
        self.force_body_onehot.scatter_(
            1,
            self._candidate_context_slots[self._sampled_body_slot][:, None],
            1.0,
        )

        duration = 2.0 * self._ramp_time + self._hold_time
        normalized_phase = torch.clamp(
            self._event_time / torch.clamp(duration, min=1.0e-6), 0.0, 1.0
        )
        phase_angle = 2.0 * torch.pi * normalized_phase
        self.online_phase = torch.stack(
            (torch.sin(phase_angle), torch.cos(phase_angle)), dim=-1
        )

        selected_quat = self.robot.data.body_quat_w[
            self._env_ids, self.selected_body_ids
        ]
        self.forcefield_forces_b = math_utils.quat_rotate_inverse(
            selected_quat, self.force_world
        )
        self.forcefield_torques_b.zero_()

    def _advance_events(self) -> None:
        self._event_time += self._step_dt
        duration = 2.0 * self._ramp_time + self._hold_time
        ended = torch.where(self._event_time >= duration)[0]
        if ended.numel() > 0:
            self._sample_events(ended)
        pending = torch.where(
            self._solution_pending & (self._event_time >= 0.0)
        )[0]
        if pending.numel() > 0:
            self._solve_pending_events(pending)

    def _update_command(
        self,
        step: bool = True,
        motion_res: dict[str, torch.Tensor] | None = None,
        env_origins: torch.Tensor | None = None,
    ):
        super()._update_command(step=step, motion_res=motion_res, env_origins=env_origins)
        if not self._online_ready:
            return
        if step:
            self._advance_events()
        self._build_online_reference()

    def _update_metrics(self):
        super()._update_metrics()
        if not self._online_ready:
            return
        # new change
        reference_root = self._root_pos[:, 1, :3]
        needs_anchor = ~self._world_reference_alignment_valid
        if torch.any(needs_anchor):
            self._world_reference_alignment[needs_anchor] = (
                self.robot.data.root_pos_w[needs_anchor] - reference_root[needs_anchor]
            )
            # new change
            self._robot_root_episode_anchor[needs_anchor] = (
                self.robot.data.root_pos_w[needs_anchor]
            )
            # new change
            self._robot_foot_episode_anchor_xy[needs_anchor] = (
                self.robot.data.body_pos_w[needs_anchor][
                    :, self._foot_body_ids, :2
                ]
            )
            self._world_reference_alignment_valid[needs_anchor] = True
        # new change
        if self._online_keypoints is not None:
            # new change
            # Stand quality must be evaluated in the commanded world frame.
            # Re-anchoring to the measured root would hide whole-body sliding.
            # adapted_world = (
            #     self._online_keypoints[:, 1, :, :3]
            #     + self._world_reference_alignment[:, None, :]
            # )
            # new change
            # SoftMimic tracks articulation relative to the current root pose.
            # Evaluate arm/keypoint execution in that same local frame, while
            # the independent root-displacement metric below detects sliding.
            adapted_world = self.localize_keypoints(
                self._online_keypoints[:, 1, :, :3]
            )
            actual_world = self.robot.data.body_pos_w[
                :, self.keypoint_body_ids_tensor, :3
            ]
            keypoint_error = torch.linalg.vector_norm(
                actual_world - adapted_world, dim=-1
            )
            self.actual_keypoint_mean_error_m = keypoint_error.mean(dim=-1)
            # new change
            self._actual_all_keypoint_error_sum += keypoint_error.sum()
            self._actual_all_keypoint_error_count += keypoint_error.numel()
            self._actual_all_keypoint_error_max = torch.maximum(
                self._actual_all_keypoint_error_max, keypoint_error.max()
            )
            self.actual_active_link_error_m = keypoint_error[
                self._env_ids, self.selected_keypoint_ids
            ]
            if torch.any(self.active_force_mask):
                active_error = self.actual_active_link_error_m[
                    self.active_force_mask
                ]
                # new change
                active_slots = self._sampled_body_slot[self.active_force_mask]
                self._actual_active_error_sum += active_error.sum()
                self._actual_active_error_count += active_error.numel()
                self._actual_active_error_max = torch.maximum(
                    self._actual_active_error_max, active_error.max()
                )
                self._actual_active_error_sum_by_slot.scatter_add_(
                    0, active_slots, active_error
                )
                self._actual_active_error_count_by_slot.scatter_add_(
                    0, active_slots, torch.ones_like(active_slots)
                )
                for slot in range(len(self.cfg.force_body_names)):
                    slot_mask = active_slots == slot
                    if torch.any(slot_mask):
                        self._actual_active_error_max_by_slot[slot] = torch.maximum(
                            self._actual_active_error_max_by_slot[slot],
                            active_error[slot_mask].max(),
                        )
        # new change
        # root_reference_xy = (
        #     reference_root[:, :2] + self._world_reference_alignment[:, :2]
        # )
        # new change
        # Do not subtract the motion-file root trajectory here.  The question
        # is simply whether the physical robot base moved in the world.
        root_reference_xy = self._robot_root_episode_anchor[:, :2]
        self.actual_root_xy_drift_m = torch.linalg.vector_norm(
            self.robot.data.root_pos_w[:, :2] - root_reference_xy, dim=-1
        )
        self._actual_root_xy_drift_max = torch.maximum(
            self._actual_root_xy_drift_max, self.actual_root_xy_drift_m.max()
        )
        # new change
        self._actual_root_xy_drift_sum += self.actual_root_xy_drift_m.sum()
        self._actual_root_xy_drift_count += self.actual_root_xy_drift_m.numel()
        # new change
        current_foot_xy = self.robot.data.body_pos_w[
            :, self._foot_body_ids, :2
        ]
        self.actual_foot_xy_slip_m = torch.linalg.vector_norm(
            current_foot_xy - self._robot_foot_episode_anchor_xy, dim=-1
        )
        self._actual_foot_xy_slip_max = torch.maximum(
            self._actual_foot_xy_slip_max, self.actual_foot_xy_slip_m.max()
        )
        # new change
        self._actual_foot_xy_slip_sum += self.actual_foot_xy_slip_m.sum()
        self._actual_foot_xy_slip_count += self.actual_foot_xy_slip_m.numel()

        masses = self.robot.data.default_mass.to(
            device=self.device, dtype=self.robot.data.body_com_pos_w.dtype
        )
        actual_com = (
            self.robot.data.body_com_pos_w * masses[:, :, None]
        ).sum(dim=1) / masses.sum(dim=1, keepdim=True)
        support_midpoint = self.robot.data.body_com_pos_w[
            self._env_ids[:, None], self._foot_body_ids[None], :
        ].mean(dim=1)
        self.actual_com_support_xy_distance_m = torch.linalg.vector_norm(
            actual_com[:, :2] - support_midpoint[:, :2], dim=-1
        )
        self._actual_com_support_xy_distance_sum += (
            self.actual_com_support_xy_distance_m.sum()
        )
        self._actual_com_support_xy_distance_count += (
            self.actual_com_support_xy_distance_m.numel()
        )
        self._actual_com_support_xy_distance_max = torch.maximum(
            self._actual_com_support_xy_distance_max,
            self.actual_com_support_xy_distance_m.max(),
        )

        upright_cosine = torch.clamp(
            -self.robot.data.projected_gravity_b[:, 2], min=-1.0, max=1.0
        )
        self.actual_root_tilt_rad = torch.acos(upright_cosine)
        self._actual_root_tilt_sum += self.actual_root_tilt_rad.sum()
        self._actual_root_tilt_count += self.actual_root_tilt_rad.numel()
        self._actual_root_tilt_max = torch.maximum(
            self._actual_root_tilt_max, self.actual_root_tilt_rad.max()
        )
        self.metrics["online_ik_feasible_scale"] = self.ik_feasible_scale
        self.metrics["online_ik_active_residual_m"] = self.ik_active_residual_m
        self.metrics["online_ik_max_foot_motion_m"] = self.ik_max_foot_motion_m
        self.metrics["online_ik_com_residual_m"] = self.ik_com_residual_m
        self.metrics["online_force_active"] = self.active_force_mask.float()
        self.metrics["online_ik_direction_cosine"] = self.ik_direction_cosine

    def reset_motions(self, env_ids: Sequence[int] | None = None):
        super().reset_motions(env_ids)
        if not self._online_ready:
            return
        # new change
        reset_ids = (
            self._env_ids
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        self._world_reference_alignment_valid[reset_ids] = False
        if env_ids is None:
            tensor_ids = self._env_ids
        else:
            tensor_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._sample_events(tensor_ids)
        self._build_online_reference()

    # new change
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create the dedicated online-reference quality visualization."""

        if debug_vis:
            visualizers = (
                ("online_nominal_visualizer", self.cfg.online_nominal_keypoints_visualizer_cfg),
                ("online_adapted_visualizer", self.cfg.online_adapted_keypoints_visualizer_cfg),
                ("online_actual_visualizer", self.cfg.online_actual_keypoints_visualizer_cfg),
                ("online_force_visualizer", self.cfg.online_force_visualizer_cfg),
                ("online_setpoint_visualizer", self.cfg.online_setpoint_visualizer_cfg),
                (
                    "online_rejected_visualizer",
                    self.cfg.online_rejected_setpoint_visualizer_cfg,
                ),
                ("online_com_support_visualizer", self.cfg.online_com_support_visualizer_cfg),
            )
            for attribute, marker_cfg in visualizers:
                if not hasattr(self, attribute):
                    setattr(self, attribute, VisualizationMarkers(marker_cfg))
                getattr(self, attribute).set_visibility(True)
        else:
            for attribute in (
                "online_nominal_visualizer",
                "online_adapted_visualizer",
                "online_actual_visualizer",
                "online_force_visualizer",
                "online_setpoint_visualizer",
                "online_rejected_visualizer",
                "online_com_support_visualizer",
            ):
                if hasattr(self, attribute):
                    getattr(self, attribute).set_visibility(False)

    # new change
    def _debug_vis_callback(self, event):
        """Draw measured/nominal/online keypoints and the synchronized force."""

        # new change
        # Isaac Sim can issue one final debug-draw callback after env.close()
        # has removed the scene.  Treat that shutdown callback as a no-op.
        try:
            scene_alive = hasattr(self._env, "scene")
        except (AttributeError, ReferenceError):
            scene_alive = False
        if not scene_alive or not self._online_ready or self._keypoints is None:
            return
        identity = torch.zeros(
            self.num_envs * len(self.keypoint_body_ids), 4, device=self.device
        )
        identity[:, 0] = 1.0
        # new change
        # Keep the red/yellow references fixed to the commanded Stand world
        # frame so any blue whole-body drift is immediately visible.
        nominal = self._keypoints[:, 1, :, :3].reshape(-1, 3)
        adapted = self._online_keypoints[:, 1, :, :3].reshape(-1, 3)
        actual = self.robot.data.body_pos_w[
            :, self.keypoint_body_ids_tensor, :3
        ].reshape(-1, 3)
        self.online_nominal_visualizer.visualize(
            translations=nominal, orientations=identity
        )
        self.online_adapted_visualizer.visualize(
            translations=adapted, orientations=identity
        )
        self.online_actual_visualizer.visualize(
            translations=actual, orientations=identity
        )

        env_ids = self._env_ids
        active = self.active_force_mask
        if torch.any(active):
            active_envs = env_ids[active]
            active_force = self.force_world[active]
            application_pos = self.robot.data.body_pos_w[
                active_envs, self.selected_body_ids[active]
            ]
            force_norm = torch.linalg.vector_norm(active_force, dim=-1)
            arrow_orientation = quat_from_axis_to_vector(active_force, axis="x")
            default_scale = self.online_force_visualizer.cfg.markers["force"].scale
            arrow_scale = torch.tensor(
                default_scale, device=self.device, dtype=active_force.dtype
            ).repeat(active_force.shape[0], 1)
            # new change
            # Keep the safe 2--6 N curriculum visually legible in recorded
            # frames.  This changes markers only, never the applied wrench.
            arrow_scale[:, 0] *= torch.clamp(force_norm / 2.0, min=1.0, max=3.0)
            arrow_position = application_pos + (
                math_utils.normalize(active_force) * arrow_scale[:, :1] * 0.25
            )
            self.online_force_visualizer.set_visibility(True)
            self.online_force_visualizer.visualize(
                translations=arrow_position,
                orientations=arrow_orientation,
                scales=arrow_scale,
            )

            adapted_by_env = adapted.reshape(
                self.num_envs, len(self.keypoint_body_ids), 3
            )
            setpoint = adapted_by_env[
                active_envs, self.selected_keypoint_ids[active]
            ]
            setpoint_quat = torch.zeros(
                setpoint.shape[0], 4, device=self.device
            )
            setpoint_quat[:, 0] = 1.0
            self.online_setpoint_visualizer.set_visibility(True)
            self.online_setpoint_visualizer.visualize(
                translations=setpoint, orientations=setpoint_quat
            )
        else:
            self.online_force_visualizer.set_visibility(False)
            self.online_setpoint_visualizer.set_visibility(False)

        rejected = (
            (self._event_time >= 0.0)
            & (~self._solution_pending)
            & (self.ik_feasible_scale <= 0.0)
        )
        if torch.any(rejected):
            rejected_envs = env_ids[rejected]
            nominal_by_env = nominal.reshape(
                self.num_envs, len(self.keypoint_body_ids), 3
            )
            rejected_target = nominal_by_env[
                rejected_envs, self.selected_keypoint_ids[rejected]
            ] + (
                self._sampled_peak_force_world[rejected]
                / self._event_stiffness[rejected]
            )
            rejected_quat = torch.zeros(
                rejected_target.shape[0], 4, device=self.device
            )
            rejected_quat[:, 0] = 1.0
            self.online_rejected_visualizer.set_visibility(True)
            self.online_rejected_visualizer.visualize(
                translations=rejected_target, orientations=rejected_quat
            )
        else:
            self.online_rejected_visualizer.set_visibility(False)

        masses = self.robot.data.default_mass.to(
            device=self.device, dtype=self.robot.data.body_com_pos_w.dtype
        )
        com = (
            self.robot.data.body_com_pos_w * masses[:, :, None]
        ).sum(dim=1) / masses.sum(dim=1, keepdim=True)
        support = self.robot.data.body_com_pos_w[
            env_ids[:, None], self._foot_body_ids[None], :
        ].mean(dim=1)
        com_support_positions = torch.cat((com, support), dim=0)
        com_support_quat = torch.zeros(
            2 * self.num_envs, 4, device=self.device
        )
        com_support_quat[:, 0] = 1.0
        marker_indices = torch.cat(
            (
                torch.zeros(self.num_envs, device=self.device, dtype=torch.long),
                torch.ones(self.num_envs, device=self.device, dtype=torch.long),
            )
        )
        self.online_com_support_visualizer.visualize(
            translations=com_support_positions,
            orientations=com_support_quat,
            marker_indices=marker_indices,
        )

    @property
    def adapted_dof_pos(self) -> torch.Tensor:
        return self._online_dof_pos

    @property
    def adapted_dof_vel(self) -> torch.Tensor:
        return self._online_dof_vel

    @property
    def adapted_keypoints(self) -> torch.Tensor | None:
        return self._online_keypoints

    @property
    def adapted_keypoint_rotations(self) -> torch.Tensor | None:
        return self._online_keypoint_rotations

    @property
    def dof_pos(self) -> torch.Tensor:
        return self._online_dof_pos

    @property
    def dof_vel(self) -> torch.Tensor:
        return self._online_dof_vel

    @property
    def keypoints(self) -> torch.Tensor | None:
        return self._online_keypoints

    @property
    def keypoint_rotations(self) -> torch.Tensor | None:
        return self._online_keypoint_rotations

    # new change
    def diagnostic_summary(self) -> dict[str, float | int]:
        """Return compact synchronization/feasibility diagnostics for test.py."""

        inactive = ~self.active_force_mask
        if torch.any(inactive):
            zero_q_error = torch.max(
                torch.abs(self._online_dof_pos[inactive] - self._dof_pos[inactive])
            )
            if self._keypoints is None:
                zero_keypoint_error = torch.zeros((), device=self.device)
            else:
                zero_keypoint_error = torch.max(
                    torch.abs(
                        self._online_keypoints[inactive] - self._keypoints[inactive]
                    )
                )
        else:
            zero_q_error = torch.zeros((), device=self.device)
            zero_keypoint_error = torch.zeros((), device=self.device)
        solved = int(self.ik_solve_count.sum().item())
        accepted = int(self.ik_accept_count.sum().item())
        mean_cosine = (
            float(
                (
                    self._accepted_direction_sum
                    / self._accepted_direction_count.clamp(min=1)
                ).item()
            )
            if int(self._accepted_direction_count.item()) > 0
            else 0.0
        )
        # new change
        per_body = {}
        for slot, body_name in enumerate(self.cfg.force_body_names):
            slot_solved = int(self._solve_count_by_slot[slot].item())
            slot_accepted = int(self._accept_count_by_slot[slot].item())
            execution_count = int(
                self._actual_active_error_count_by_slot[slot].item()
            )
            per_body[body_name] = {
                "solves": slot_solved,
                "accepted": slot_accepted,
                "acceptance_rate": slot_accepted / max(slot_solved, 1),
                "max_force_n": float(self._max_force_by_slot[slot].item()),
                "max_joint_delta_rad": float(
                    self._max_joint_delta_by_slot[slot].item()
                ),
                "max_keypoint_delta_m": float(
                    self._max_keypoint_delta_by_slot[slot].item()
                ),
                "max_ik_residual_m": float(
                    self._max_residual_by_slot[slot].item()
                ),
                "actual_active_error_mean_m": (
                    float(
                        (
                            self._actual_active_error_sum_by_slot[slot]
                            / max(execution_count, 1)
                        ).item()
                    )
                    if execution_count > 0
                    else 0.0
                ),
                "actual_active_error_max_m": float(
                    self._actual_active_error_max_by_slot[slot].item()
                ),
            }
        return {
            "solves": solved,
            "accepted": accepted,
            "acceptance_rate": accepted / max(solved, 1),
            "active_envs": int(self.active_force_mask.sum().item()),
            # new change
            "max_force_n": float(self._max_accepted_force_seen.item()),
            "current_force_n": float(
                torch.linalg.vector_norm(self.force_world, dim=-1).max().item()
            ),
            # new change
            # Historical maxima remain meaningful even when the final video
            # frame happens to be in a quiet interval.
            "max_joint_delta_rad": float(self._max_joint_delta_seen.item()),
            "max_keypoint_delta_m": float(self._max_keypoint_delta_seen.item()),
            "max_active_residual_m": float(self._max_active_residual_seen.item()),
            "max_foot_motion_m": float(self._max_foot_motion_seen.item()),
            "max_com_residual_m": float(self._max_com_residual_seen.item()),
            "max_joint_speed_rad_s": float(self._max_joint_speed_seen.item()),
            # new change
            "actual_active_link_error_mean_m": (
                float(
                    (
                        self._actual_active_error_sum
                        / self._actual_active_error_count.clamp(min=1)
                    ).item()
                )
                if int(self._actual_active_error_count.item()) > 0
                else 0.0
            ),
            "actual_active_link_error_max_m": float(
                self._actual_active_error_max.item()
            ),
            "actual_all_keypoints_error_mean_m": (
                float(
                    (
                        self._actual_all_keypoint_error_sum
                        / self._actual_all_keypoint_error_count.clamp(min=1)
                    ).item()
                )
                if int(self._actual_all_keypoint_error_count.item()) > 0
                else 0.0
            ),
            "actual_all_keypoints_error_max_m": float(
                self._actual_all_keypoint_error_max.item()
            ),
            # new change
            "actual_root_xy_drift_max_m": float(
                self._actual_root_xy_drift_max.item()
            ),
            "actual_root_xy_drift_mean_m": float(
                (
                    self._actual_root_xy_drift_sum
                    / self._actual_root_xy_drift_count.clamp(min=1)
                ).item()
            ),
            # new change
            "actual_foot_xy_slip_max_m": float(
                self._actual_foot_xy_slip_max.item()
            ),
            "actual_foot_xy_slip_mean_m": float(
                (
                    self._actual_foot_xy_slip_sum
                    / self._actual_foot_xy_slip_count.clamp(min=1)
                ).item()
            ),
            # new change
            "actual_com_support_xy_distance_mean_m": float(
                (
                    self._actual_com_support_xy_distance_sum
                    / self._actual_com_support_xy_distance_count.clamp(min=1)
                ).item()
            ),
            "actual_com_support_xy_distance_max_m": float(
                self._actual_com_support_xy_distance_max.item()
            ),
            "actual_root_tilt_mean_rad": float(
                (
                    self._actual_root_tilt_sum
                    / self._actual_root_tilt_count.clamp(min=1)
                ).item()
            ),
            "actual_root_tilt_max_rad": float(
                self._actual_root_tilt_max.item()
            ),
            "mean_direction_cosine": mean_cosine,
            "min_direction_cosine": (
                float(self._accepted_direction_min.item())
                if accepted > 0
                else 0.0
            ),
            "inactive_q_equivalence_max_rad": float(zero_q_error.item()),
            "inactive_keypoint_equivalence_max_m": float(
                zero_keypoint_error.item()
            ),
            # new change
            "per_body": per_body,
        }
