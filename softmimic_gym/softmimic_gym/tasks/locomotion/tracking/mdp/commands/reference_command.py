# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import numpy as np

# new change
from softmimic_gym.assets.robots.g1_compat import (
    G1_SOFTMIMIC_29_JOINT_NAMES,
    G1_SOFTMIMIC_NUM_JOINTS,
)

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers.visualization_markers import VisualizationMarkers

# new change
from ..keypoints import resolve_keypoint_body_ids

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformJointPosCommandCfg, TimeVaryingJointPosCommandCfg, WholeExoCommandCfg, AMASSCommandCfg

class ReferenceCommand(CommandTerm):
    """Command generator that samples motions from AMASS dataset using MotionLibH1.
    """

    def __init__(self, cfg: ReferenceCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)

        random_seed = 42
        # random_seed = 43
        # new change
        # The old hard-coded value made --seed change physics but not the
        # sampled force body/direction/magnitude/timing sequence.  Keep 42 as
        # the config default and permit explicit unseen event-seed validation.
        random_seed = int(getattr(cfg, "command_random_seed", random_seed))
        self.torch_gen = torch.Generator(device=self.device)
        # if random_seed is not None:
        self.torch_gen.manual_seed(random_seed)

        # obtain the robot asset
        self.robot = env.scene[cfg.asset_name]

        self.robot_type = cfg.joint_config.robot_type
        self.num_joints = cfg.joint_config.num_joints

        print(f"Using robot {self.robot_type}")

        # new change
        # The official articulation has 43 joints; every SoftMimic tensor has
        # 29.  Resolve once and reuse the canonical ordering everywhere.
        if self.robot_type == "g1" and self.num_joints == G1_SOFTMIMIC_NUM_JOINTS:
            self.control_joint_ids, resolved_names = self.robot.find_joints(
                list(G1_SOFTMIMIC_29_JOINT_NAMES), preserve_order=True
            )
            if list(resolved_names) != list(G1_SOFTMIMIC_29_JOINT_NAMES):
                raise RuntimeError(
                    "The G1 articulation did not resolve the canonical 29DOF interface: "
                    f"resolved={list(resolved_names)}."
                )
        else:
            self.control_joint_ids = None

        # # store the joint names
        # if cfg.demo_joint_order_mode == "default":
        #     # self.joint_names = ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle', 'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle', 'torso', 'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow', 'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow']
        #     # self.joint_names = ['left_hip_yaw', 'right_hip_yaw', 'torso', 'left_hip_roll', 'right_hip_roll', 'left_shoulder_pitch', 'right_shoulder_pitch', 'left_hip_pitch', 'right_hip_pitch', 'left_shoulder_roll', 'right_shoulder_roll', 'left_knee', 'right_knee', 'left_shoulder_yaw', 'right_shoulder_yaw', 'left_ankle', 'right_ankle', 'left_elbow', 'right_elbow']
        #     pass
        # else:
        #     # TODO(get rid of this)
        #     raise ValueError(f"Invalid joint order mode: {cfg.demo_joint_order_mode}")
        #     # joint_names = ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle', 'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle', 'torso', 'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow', 'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow']
        #     # remap_order = [10,  0,  1,  2,  3, 11,  5,  6,  7,  8, 15, 12, 13, 14,  4, 16, 17, 18, 9]
        #     joint_names = ['left_hip_yaw', 'right_hip_yaw', 'torso', 'left_hip_roll', 'right_hip_roll', 'left_shoulder_pitch', 'right_shoulder_pitch', 'left_hip_pitch', 'right_hip_pitch', 'left_shoulder_roll', 'right_shoulder_roll', 'left_knee', 'right_knee', 'left_shoulder_yaw', 'right_shoulder_yaw', 'left_ankle', 'right_ankle', 'left_elbow', 'right_elbow']
        #     remap_order = [ 2,  5,  6,  0,  1,  9, 10,  3,  4, 13, 14,  7,  8, 17, 18, 11, 12, 15, 16]
        #     self.joint_names = [joint_names[i] for i in remap_order]

        from softmimic_deploy.src.motion_lib.motion_lib_from_multi_csv import JointConfig

        joint_config = JointConfig(
            num_joints=cfg.joint_config.num_joints,
            left_leg_indices={
                "hip_yaw": cfg.joint_config.left_leg_indices.hip_yaw,
                "hip_roll": cfg.joint_config.left_leg_indices.hip_roll,
                "hip_pitch": cfg.joint_config.left_leg_indices.hip_pitch,
                "knee": cfg.joint_config.left_leg_indices.knee,
                "ankle_pitch": cfg.joint_config.left_leg_indices.ankle_pitch,
            },
            right_leg_indices={
                "hip_yaw": cfg.joint_config.right_leg_indices.hip_yaw,
                "hip_roll": cfg.joint_config.right_leg_indices.hip_roll,
                "hip_pitch": cfg.joint_config.right_leg_indices.hip_pitch,
                "knee": cfg.joint_config.right_leg_indices.knee,
                "ankle_pitch": cfg.joint_config.right_leg_indices.ankle_pitch,
            },
            left_arm_indices={
                "shoulder_pitch": cfg.joint_config.left_arm_indices.shoulder_pitch,
                "shoulder_roll": cfg.joint_config.left_arm_indices.shoulder_roll,
                "shoulder_yaw": cfg.joint_config.left_arm_indices.shoulder_yaw,
                "elbow": cfg.joint_config.left_arm_indices.elbow,
                "wrist_pitch": cfg.joint_config.left_arm_indices.wrist_pitch,
                "wrist_roll": cfg.joint_config.left_arm_indices.wrist_roll,
                "wrist_yaw": cfg.joint_config.left_arm_indices.wrist_yaw,
            },
            right_arm_indices={
                "shoulder_pitch": cfg.joint_config.right_arm_indices.shoulder_pitch,
                "shoulder_roll": cfg.joint_config.right_arm_indices.shoulder_roll,
                "shoulder_yaw": cfg.joint_config.right_arm_indices.shoulder_yaw,
                "elbow": cfg.joint_config.right_arm_indices.elbow,
                "wrist_pitch": cfg.joint_config.right_arm_indices.wrist_pitch,
                "wrist_roll": cfg.joint_config.right_arm_indices.wrist_roll,
                "wrist_yaw": cfg.joint_config.right_arm_indices.wrist_yaw,
            },
            thigh_length=cfg.joint_config.thigh_length,
            calf_length=cfg.joint_config.calf_length,
            # robot_type=cfg.joint_config.robot_type
        )

        # Initialize motion library
        if cfg.demo_recording_path is not None:
            reindex_mapping=[0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
            feet_contacts = True
            
            from softmimic_deploy.src.motion_lib import ProceduralMotionLibFromDemo, AugmentedMotionLibFromDemo

            motion_lib_classes = {
                "ProceduralMotionLibFromDemo": ProceduralMotionLibFromDemo,
                "AugmentedMotionLibFromDemo": AugmentedMotionLibFromDemo,
            }
            try:
                demo_loader_class = motion_lib_classes[cfg.demo_loader_class]
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported demo_loader_class '{cfg.demo_loader_class}'. "
                    f"Supported options: {', '.join(motion_lib_classes.keys())}"
                ) from exc

            self.motion_lib = demo_loader_class(
                cfg.demo_recording_path,
                motion_dt=env.step_dt,
                start_range=cfg.demo_start_range,
                n_future_steps=cfg.n_future_steps,
                demo_playback_mode="references",
                device=self.device,
                joint_config=joint_config,
                reindex_mapping=reindex_mapping,
                feet_contacts=feet_contacts,
                upper_demo_only=cfg.demo_upper_body_only,
            )
        else:
            try:
                from softmimic_motions.raibert_heuristic.motion_lib_h1_proc import ProceduralMotionLib
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "Procedural motion generation requires the optional 'softmimic_motions' package. "
                    "Install it or provide a demo_recording_path."
                ) from exc
            self.motion_lib = ProceduralMotionLib(
                motion_dt=env.step_dt,
                # motion_dt=env.physics_dt,
                device=self.device,
                fix_height=False,
                multi_thread=False,
                extend_head=False,
                motion_length_s=cfg.resampling_time_range[1],
                random_arms=cfg.proc_ranges.random_arms,
                natural_arms=False,#True, # TODO(gmargo): True is broken for now
                joint_config=joint_config,
            )
        
        self.motion_lib.set_motion_parameters(
            freq_range=[cfg.proc_ranges.min_freq, cfg.proc_ranges.max_freq],
            height_range=[cfg.proc_ranges.min_height, cfg.proc_ranges.max_height],
            vel_range=[cfg.proc_ranges.min_vel, cfg.proc_ranges.max_vel],
            yaw_vel_range=[cfg.proc_ranges.min_yaw_vel, cfg.proc_ranges.max_yaw_vel],
            pitch_range=[cfg.proc_ranges.min_pitch, cfg.proc_ranges.max_pitch],
            standing_prob=cfg.proc_ranges.standing_prob,
            stance_duration=cfg.proc_ranges.stance_duration,
            swing_height=cfg.proc_ranges.swing_height,
            num_steps_to_stand=cfg.proc_ranges.num_steps_to_stand,
            dynamic_standing_only=cfg.proc_ranges.dynamic_standing_only,
            standing_pitch_range=[cfg.proc_ranges.min_standing_pitch, cfg.proc_ranges.max_standing_pitch],
            standing_height_range=[cfg.proc_ranges.min_standing_height, cfg.proc_ranges.max_standing_height],
        )

        self.n_future_steps = cfg.n_future_steps
        self.future_dt = cfg.future_dt
        
        # new change
        self.keypoint_body_ids = resolve_keypoint_body_ids(self.robot, cfg.keypoint_body_ids)
        self._keypoint_body_ids_tensor = (
            torch.tensor(self.keypoint_body_ids, device=self.device, dtype=torch.long)
            if len(self.keypoint_body_ids) > 0
            else None
        )

        # Initialize motion IDs and offsets
        self.motion_ids = torch.arange(self.num_envs, device=self.device)
        self.offset = torch.zeros(self.num_envs, 3, device=self.device)
        self.motion_count = torch.zeros(self.num_envs, device=self.device)
        self.motion_start = torch.zeros(self.num_envs, device = self.device)
        self.motion_durations = torch.rand(self.num_envs, device=self.device, generator=self.torch_gen) * (cfg.resampling_time_range[1] - cfg.resampling_time_range[0]) + cfg.resampling_time_range[0]
        self.update_ids = torch.empty(0, device=self.device, dtype=torch.long)
        
        if self.cfg.demo_random_init:
            
            # linearly space
            # self.motion_count[:] = torch.linspace(0, int(cfg.resampling_time_range[0] / self.motion_lib._motion_dt), self.num_envs, device=self.device).long()
            # randomly shuffle
            # self.motion_count = self.motion_count[torch.randperm(self.num_envs, generator=self.torch_gen, device=self.device)]

            # randomly sample within bounds
            max_times = self.motion_lib.get_max_times(self.motion_ids)
            max_frames = (max_times / self.motion_lib._motion_dt).long() - 1
            max_frames = torch.min(max_frames, torch.tensor(int(self.cfg.resampling_time_range[0] / self.motion_lib._motion_dt), device=self.device, dtype=torch.float))
            min_frames = torch.zeros_like(max_frames, device=self.device, dtype=torch.long)
            random_floats = torch.rand(
                (self.num_envs,), 
                device=self.device, 
                generator=self.torch_gen
            )
            self.motion_count[:] = (
                min_frames + random_floats * (max_frames - min_frames).float()
            ).long()
            
            
            # print(self.motion_count)
            # print(max_frames)
            # input()
        if self.cfg.clip_length:
            max_times = self.motion_lib.get_max_times(self.motion_ids)
            max_frames = (max_times / self.motion_lib._motion_dt).long() - 1
            max_frames = torch.min(max_frames, torch.tensor(int(self.cfg.clip_length / self.motion_lib._motion_dt), device=self.device, dtype=torch.float))
            min_frames = torch.zeros_like(max_frames, device=self.device, dtype=torch.long)

            # motion durations is effectively end times of the motion clip
            self.motion_durations[:] = torch.rand(self.num_envs, device=self.device, generator=self.torch_gen)*(max_times-self.cfg.clip_length) + self.cfg.clip_length
            #motion count is effectively the start index (and then we keep incrementing it)
            self.motion_count[:]=torch.max((self.motion_durations-self.cfg.clip_length)/self.motion_lib._motion_dt, min_frames)
            self.motion_start = self.motion_count.clone()
            # print("motion count init ", self.motion_count)
            # new change
            # Avoid dumping one value per environment during large training runs.

        # Load motions
        self.motion_lib.load_motions(
            env_ids=torch.arange(self.num_envs, device=self.device),
            durations=self.motion_durations,
            limb_weights=[torch.zeros(10, device=self.device)] * self.num_envs,
            random_sample=True
        )

        # Store resulting motion durations
        max_times = self.motion_lib.get_max_times(self.motion_ids)
        max_frames = (max_times / self.motion_lib._motion_dt).long() - 1
        self.motion_durations = torch.min(
            self.motion_durations,
            max_frames * self.motion_lib._motion_dt,
        )
        # print("max frames seconds", max_frames * self.motion_lib._motion_dt)
        # Store motion dt
        self.motion_dt = self.motion_lib._motion_dt

        # # initialize the command buffers
        # self._command = torch.zeros(self.num_envs, self.num_joints * (self.n_future_steps+1), device=self.device)
        # self._next_command = torch.zeros(self.num_envs, self.num_joints * (self.n_future_steps+1), device=self.device)
        # self._command_update_velocity = torch.zeros(self.num_envs, self.num_joints * (self.n_future_steps+1), device=self.device)

        # buffer of past true root poses and commanded root poses
        self.horizon_length = 100
        self.current_root_pose_buf = torch.zeros((self.num_envs, self.horizon_length, 7), device=self.device)
        self.commanded_root_pose_buf = torch.zeros((self.num_envs, self.horizon_length, 7), device=self.device)

        # metrics for logging
        self.metrics = {
            str("error_joint_" + str(name)): torch.zeros(self.num_envs, device=self.device) 
            for name in range(self.num_joints)
        }
        self.metrics[str("keypoint_error_mae")] = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.horizon_length // 10):
            h = i * 10
            self.metrics[str("keypoint_error_mae_horizon_" + str(h))] = torch.zeros(self.num_envs, device=self.device)

        # self._true_dof_pos = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        # self._true_dof_vel = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        # self._prev_true_dof_pos = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        # self._prev_true_dof_vel = torch.zeros(self.num_envs, self.num_joints, device=self.device)

        
        # 
        # # load the base policy if applicable
        # if cfg.demo_base_policy_path is not None:
        #     self.base_policy = torch.jit.load(cfg.demo_base_policy_path).to(self.device)
        #     self.base_policy_decimation = 4
        #     self.last_policy_actions = torch.zeros(self._env.num_envs, 19, device=self._env.device)
        #     self.obs_history_buf = torch.zeros(self._env.num_envs, 63*3, device=self._env.device)
        #     self._policy_outputs = torch.zeros(self._env.num_envs, 19, device=self._env.device)
        # else:
        #     self.base_policy = None
            
        # Initialize keypoint extractor
        self.keypoint_extractor = None

        self._update_command(step=False)

    def __str__(self) -> str:
        """Return a string representation of the command."""
        msg = f"AMASSCommand:\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The motion command."""
        return self._command

    @property
    def keypoint_body_ids_tensor(self) -> torch.Tensor | None:
        """Keypoint body ids cached on the correct device."""
        return self._keypoint_body_ids_tensor

    def _update_metrics(self):
        """Update tracking error metrics."""
        # time for which the command was executed
        max_command_step = self._env.max_episode_length
        # logs data
        for joint_number in range(self.num_joints):
            # new change
            articulation_joint_id = (
                joint_number
                if self.control_joint_ids is None
                else self.control_joint_ids[joint_number]
            )
            self.metrics[str("error_joint_" + str(joint_number))] += torch.abs(
                self.original_dof_pos[:, 1, joint_number]
                - self.robot.data.joint_pos[:, articulation_joint_id]
            ) / max_command_step

        # log the keypoint error across the horizon
        if self._keypoints is not None:
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
            current_keypoints = self._env.scene["robot"].data.body_pos_w[:, self.keypoint_body_ids, :3]
            
            keypoint_error = self._keypoints[:,0] - current_keypoints
            # print(torch.norm(keypoint_error, dim=-1))

            keypoint_error_mae = torch.norm(keypoint_error, dim=-1).mean(dim=-1)
            self.metrics[str("keypoint_error_mae")] = keypoint_error_mae
            
            for i in range(self.horizon_length // 10):
                h = i * 10
                # Get the past root positions and rotations from the command manager
                horizon_past_current_root_pos = self.current_root_pose_buf[:, h, :3].clone()
                horizon_past_current_root_pos[:, 2] = 0.0 # ignore z position
                horizon_past_current_root_rot = self.current_root_pose_buf[:, h, 3:7].clone()
                _, _, horizon_past_current_yaw = math_utils.euler_xyz_from_quat(horizon_past_current_root_rot)
                
                # Get the past commanded root positions and rotations from the command manager
                horizon_past_commanded_root_pos = self.commanded_root_pose_buf[:, h, :3].clone()
                horizon_past_commanded_root_pos[:, 2] = 0.0 # ignore z position
                horizon_past_commanded_root_rot = self.commanded_root_pose_buf[:, h, 3:7].clone()
                horizon_past_commanded_root_rot = horizon_past_commanded_root_rot[:, [3, 0, 1, 2]]
                _, _, horizon_past_commanded_yaw = math_utils.euler_xyz_from_quat(horizon_past_commanded_root_rot)

                # Transform the keypoints into the horizon past local frame
                yaw_diff = math_utils.wrap_to_pi(horizon_past_current_yaw - horizon_past_commanded_yaw)
                yaw_rotation = math_utils.quat_from_euler_xyz(torch.zeros_like(horizon_past_current_yaw), torch.zeros_like(horizon_past_current_yaw), yaw_diff)
                
                commanded_keypoints = self._keypoints[:,0, :, :3].clone()#.reshape(self.num_envs * 14, 3)
                commanded_keypoints_local_horizon = commanded_keypoints - horizon_past_commanded_root_pos.unsqueeze(1)
                # Rotate the commanded keypoints to the local frame

                commanded_keypoints_local_horizon = math_utils.quat_rotate(
                    yaw_rotation.unsqueeze(1).repeat(1, commanded_keypoints.shape[1], 1),
                    commanded_keypoints_local_horizon,
                )

                keypoints_transformed_h = (commanded_keypoints_local_horizon + horizon_past_current_root_pos.unsqueeze(1))
                keypoint_error_h = current_keypoints - keypoints_transformed_h
                keypoint_error_h_mae = torch.norm(keypoint_error_h, dim=-1).mean(dim=-1)

                self.metrics[str("keypoint_error_mae_horizon_" + str(h))] = keypoint_error_h_mae

        # print(keypoint_error_h_mae[:10])

        # # log the error between joint pos and demo's true dof pos
        # if self.cfg.demo_recording_path is not None:
        #     current_joint_pos = self.robot.data.joint_pos[:, :]
        #     # demo_joint_pos = self._prev_true_dof_pos
        #     demo_joint_pos = self._next_true_dof_pos
            
        #     angle = current_joint_pos - demo_joint_pos

        #     for joint_number in range(self.num_joints):
        #         self.metrics[str("demo_error_pos_" + str(joint_number))] += angle.abs()[:, joint_number] / max_command_step
        #         self.metrics[str("demo_error_vel_" + str(joint_number))] += torch.abs(self.robot.data.joint_vel[:, joint_number] - self._next_true_dof_vel[:, joint_number]) / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the specified environments."""
        pass
        
    def reset_motions(self, env_ids: Sequence[int] | None = None):
        # if len(env_ids) > 0: print("RESET MOTIONS")
        if self.cfg.demo_recording_path is not None:
            if self.cfg.demo_random_reset:
                # self.motion_count[env_ids] = torch.randint(0, int(self.cfg.resampling_time_range[0] / self.motion_lib._motion_dt), (env_ids.shape[0],), device=self.device, dtype=torch.float, generator=self.torch_gen)
                max_times = self.motion_lib.get_max_times(env_ids)
                max_frames = (max_times / self.motion_lib._motion_dt).long() - 1
                max_frames = torch.min(max_frames, torch.tensor(int(self.cfg.resampling_time_range[0] / self.motion_lib._motion_dt), device=self.device, dtype=torch.float))
                min_frames = torch.zeros_like(max_frames, device=self.device, dtype=torch.long)
                random_floats = torch.rand(
                    (env_ids.shape[0],), 
                    device=self.device, 
                    generator=self.torch_gen
                )
                self.motion_count[env_ids] = (
                    min_frames + random_floats * (max_frames - min_frames).float()
                ).long()
            else:
                if self.cfg.demo_zero_reset:
                    self.motion_count[env_ids] = 0
                
                if self.cfg.clip_length:
                    self.motion_count[env_ids] = self.motion_start[env_ids].clone()
                    # print("reset env ids", env_ids)
                    # print("motion count in reset motions", self.motion_count)
                    # print('motion durations', self.motion_durations)
                    # new change
                    # These tensors can contain thousands of entries and are
                    # intentionally silent in the general training task.
            self._update_command(step=False)

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the command generator."""
        self.motion_lib.reset_positions(env_ids)

        # reset the buffers
        # self.current_root_pose_buf[env_ids] = torch.zeros((len(env_ids), self.horizon_length, 7), device=self.device)
        self.current_root_pose_buf[env_ids, :, :3] = self._env.scene["robot"].data.root_pos_w[env_ids, :3].clone().unsqueeze(1)
        self.current_root_pose_buf[env_ids, :, 3:] = self._env.scene["robot"].data.root_quat_w[env_ids, :4].clone().unsqueeze(1)
        self.commanded_root_pose_buf[env_ids, :, :3] = self._root_pos[env_ids, 0, :3].clone().unsqueeze(1)
        self.commanded_root_pose_buf[env_ids, :, 3:] = self._root_rot[env_ids,0, :4].clone().unsqueeze(1)

        return super().reset(env_ids)

    def _populate_reference_buffers(self, motion_res: dict[str, torch.Tensor], env_origins: torch.Tensor) -> None:
        env_offsets = env_origins.unsqueeze(1)
        if self.cfg.demo_zero_xy:
            root_pos = motion_res["root_pos"].clone() + env_offsets
            root_pos[:, :, 0:2] = env_offsets[:, :, 0:2]
        else:
            root_pos = motion_res["root_pos"].clone() + env_offsets

        self._command = torch.cat(
            [
                motion_res["dof_pos"].reshape(self.num_envs, -1),
                motion_res["root_vel"][:, :, 0:2].reshape(self.num_envs, -1),
                motion_res["root_ang_vel"][:, :, 2:3].reshape(self.num_envs, -1),
                motion_res["foot_contacts"][:, :, 0:2].reshape(self.num_envs, -1),
                root_pos[:, :, 0:3].reshape(self.num_envs, -1),
                motion_res["root_yaw"][:, :, 0:1].reshape(self.num_envs, -1),
                motion_res["root_pitch"][:, :, 0:1].reshape(self.num_envs, -1),
                motion_res["gait_parameters"].reshape(self.num_envs, -1),
                motion_res["gravity_vec"][:, :, 0:3].reshape(self.num_envs, -1),
            ],
            dim=1,
        )

        self._root_pos = root_pos
        self._root_rot = motion_res["root_rot"]
        self._root_vel = motion_res["root_vel"]
        self._root_ang_vel = motion_res["root_ang_vel"]
        self._root_vel_global = motion_res["root_vel_global"]
        self._root_ang_vel_global = motion_res["root_ang_vel_global"]
        self._foot_contacts = motion_res["foot_contacts"]
        self._dof_pos = motion_res["dof_pos"]
        self._dof_vel = motion_res["dof_vel"]
        self._root_pitch = motion_res["root_pitch"]
        self._root_yaw = motion_res["root_yaw"]
        self._root_gravity_vec = motion_res["gravity_vec"]
        self._root_roll = motion_res.get("root_roll", torch.zeros_like(self._root_pitch))
        self._relative_timestamp = (self.motion_count - self.motion_start) * self.motion_lib._motion_dt

        self._keypoints = motion_res.get("keypoints")
        self._keypoint_rotations = motion_res.get("keypoint_rotations")
        if self._keypoints is not None:
            if self.cfg.demo_zero_xy:
                self._keypoints[:, :, :, 0:2] -= motion_res["root_pos"][:, :, 0:2].unsqueeze(2)
                self._keypoints[:, :, :, 0:2] += root_pos[:, :, 0:2].unsqueeze(2)
            else:
                self._keypoints += env_offsets.unsqueeze(1)

    def _finalize_update(self, step: bool) -> None:
        if not step:
            return

        if (
            self.cfg.demo_recording_path is not None
            and self.cfg.demo_reset_robot_on_clip_end
            and len(self.update_ids) > 0
        ):
            positions = self._root_pos[self.update_ids, 0].clone()
            orientations = self._root_rot[self.update_ids, 0].clone()[:, [3, 0, 1, 2]]
            lin_vel_global = self._root_vel_global[self.update_ids, 0].clone()
            ang_vel_global = self._root_ang_vel_global[self.update_ids, 0].clone()
            velocities = torch.cat([lin_vel_global, ang_vel_global], dim=-1)
            self._env.scene["robot"].write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=self.update_ids
            )
            self._env.scene["robot"].write_root_velocity_to_sim(velocities, env_ids=self.update_ids)
            self._env.scene["robot"].write_joint_state_to_sim(
                self._dof_pos[self.update_ids, 0].float().clone(),
                self._dof_vel[self.update_ids, 0].float().clone(),
                # new change
                joint_ids=self.control_joint_ids,
                env_ids=self.update_ids,
            )
            if hasattr(self, "_last_ff_stiffness"):
                self._last_ff_stiffness[self.update_ids] = 0.0
            if hasattr(self, "_last_ff_rotational_stiffness"):
                self._last_ff_rotational_stiffness[self.update_ids] = 0.0

        self.motion_count += 1
        self.update_ids = torch.where(self.motion_count >= (self.motion_durations / self.motion_lib._motion_dt))[0]

        if len(self.update_ids) > 0:
            self.motion_durations[self.update_ids] = torch.rand(
                len(self.update_ids),
                device=self.device,
                generator=self.torch_gen,
            ) * (self.cfg.resampling_time_range[1] - self.cfg.resampling_time_range[0]) + self.cfg.resampling_time_range[0]

            max_times = self.motion_lib.get_max_times(self.update_ids)
            max_frames = (max_times / self.motion_lib._motion_dt).long() - 1
            self.motion_durations[self.update_ids] = torch.min(
                self.motion_durations[self.update_ids],
                max_frames * self.motion_lib._motion_dt,
            )

            if self.cfg.demo_recording_path is not None and self.cfg.demo_random_reset:
                max_frames = torch.min(
                    max_frames,
                    torch.tensor(
                        int(self.cfg.resampling_time_range[0] / self.motion_lib._motion_dt),
                        device=self.device,
                        dtype=torch.float,
                    ),
                )
                min_frames = torch.zeros_like(max_frames, device=self.device, dtype=torch.long)
                random_floats = torch.rand(len(self.update_ids), device=self.device, generator=self.torch_gen)
                self.motion_count[self.update_ids] = (
                    min_frames + random_floats * (max_frames - min_frames).float()
                ).long()
            elif self.cfg.demo_recording_path is not None and self.cfg.clip_length:
                self.motion_durations[self.update_ids] = torch.rand(
                    len(self.update_ids),
                    device=self.device,
                    generator=self.torch_gen,
                ) * (max_times - self.cfg.clip_length) + self.cfg.clip_length
                min_frames = torch.zeros_like(max_frames, device=self.device, dtype=torch.long)
                self.motion_count[self.update_ids] = torch.max(
                    (self.motion_durations[self.update_ids] - self.cfg.clip_length) / self.motion_lib._motion_dt,
                    min_frames,
                )
                self.motion_start[self.update_ids] = self.motion_count[self.update_ids].clone()
            else:
                self.motion_count[self.update_ids] = 0

            self.motion_lib.load_motions(
                env_ids=self.update_ids,
                durations=self.motion_durations[self.update_ids],
                limb_weights=[torch.zeros(10, device=self.device)] * self.num_envs,
                random_sample=True,
            )

        if len(self.update_ids) > 0:
            self.current_root_pose_buf[self.update_ids, :, :3] = (
                self._env.scene["robot"].data.root_pos_w[self.update_ids, :3].clone().unsqueeze(1)
            )
            self.current_root_pose_buf[self.update_ids, :, 3:] = (
                self._env.scene["robot"].data.root_quat_w[self.update_ids, :4].clone().unsqueeze(1)
            )
            self.commanded_root_pose_buf[self.update_ids, :, :3] = (
                self._root_pos[self.update_ids, 0, :3].clone().unsqueeze(1)
            )
            self.commanded_root_pose_buf[self.update_ids, :, 3:] = (
                self._root_rot[self.update_ids, 0, :4].clone().unsqueeze(1)
            )

        roll_envs = self.motion_ids
        self.current_root_pose_buf[roll_envs] = torch.roll(self.current_root_pose_buf[roll_envs], shifts=-1, dims=1)
        self.current_root_pose_buf[roll_envs, -1, :3] = self._env.scene["robot"].data.root_pos_w[roll_envs, :3].clone()
        self.current_root_pose_buf[roll_envs, -1, 3:] = self._env.scene["robot"].data.root_quat_w[roll_envs, :4].clone()
        self.commanded_root_pose_buf[roll_envs] = torch.roll(
            self.commanded_root_pose_buf[roll_envs], shifts=-1, dims=1
        )
        self.commanded_root_pose_buf[roll_envs, -1, :3] = self._root_pos[roll_envs, 0, :3].clone()
        self.commanded_root_pose_buf[roll_envs, -1, 3:] = self._root_rot[roll_envs, 0, :4].clone()

    def _update_command(
        self,
        step: bool = True,
        motion_res: dict[str, torch.Tensor] | None = None,
        env_origins: torch.Tensor | None = None,
    ):
        """Update commands based on current motion time."""

        motion_times = self.motion_count * self.motion_dt
        if motion_res is None:
            motion_res = self.motion_lib.get_motion_state(
                self.motion_ids,
                motion_times,
                offset=self.offset,
                future_frame_dt=self.future_dt,
            )
        if env_origins is None:
            env_origins = self._env.scene.env_origins[self.motion_ids, :3]

        self._populate_reference_buffers(motion_res, env_origins)
        self._finalize_update(step)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Implementation for debug visualization."""
        # create markers if necessary for the first tome
        # return
        if debug_vis:
            # if not hasattr(self, "goal_pose_visualizer"):
            #     self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # # set their visibility to true
            # self.goal_pose_visualizer.set_visibility(True)

            if not hasattr(self, "current_pose_visualizer"):
                # register the callback for debug visualization
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.current_pose_visualizer.set_visibility(True)

            if not hasattr(self, "local_goal_pose_visualizer"):
                self.local_goal_pose_visualizer = VisualizationMarkers(self.cfg.local_goal_pose_visualizer_cfg)
            # set their visibility to true
            self.local_goal_pose_visualizer.set_visibility(True)

            # if not hasattr(self, "local_horizon_goal_pose_visualizer"):
            #     self.local_horizon_goal_pose_visualizer = VisualizationMarkers(self.cfg.local_horizon_goal_pose_visualizer_cfg)
            # # set their visibility to true
            # self.local_horizon_goal_pose_visualizer.set_visibility(True)

        else:
            # if hasattr(self, "goal_pose_visualizer"):
            #     self.goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "cur_pose_visualizer"):
                self.current_pose_visualizer.set_visibility(False)
            if hasattr(self, "local_goal_pose_visualizer"):
                self.local_goal_pose_visualizer.set_visibility(False)
            # if hasattr(self, "local_horizon_goal_pose_visualizer"):
            #     self.local_horizon_goal_pose_visualizer.set_visibility(False)

    def localize_keypoints(self, keypoints):
        """Convert global keypoints to local frame based on current root pose."""
        if self._keypoints is None:
            return None

        current_root_pos = self.robot.data.root_pos_w[:, :3].clone()
        current_root_pos[:, 2] = 0.0  # ignore z position
        current_root_rot = self.robot.data.root_quat_w[:, :4].clone()
        _, _, current_yaw = math_utils.euler_xyz_from_quat(current_root_rot)
        commanded_root_pos = self._root_pos[:,0, :3].clone()
        commanded_root_pos[:, 2] = 0.0  # ignore z position
        commanded_root_rot = self._root_rot[:,0, :4].clone()
        commanded_root_rot = commanded_root_rot[:, [3, 0, 1, 2]]
        _, _, commanded_yaw = math_utils.euler_xyz_from_quat(commanded_root_rot)
        yaw_diff = math_utils.wrap_to_pi(current_yaw - commanded_yaw)
        yaw_rotation = math_utils.quat_from_euler_xyz(torch.zeros_like(current_yaw), torch.zeros_like(current_yaw), yaw_diff)
        commanded_keypoints_local = keypoints.clone() - commanded_root_pos.unsqueeze(1)
        commanded_keypoints_local = math_utils.quat_rotate(
            yaw_rotation.unsqueeze(1).repeat(1, commanded_keypoints_local.shape[1], 1),
            commanded_keypoints_local
        )
        local_keypoints = commanded_keypoints_local + current_root_pos.unsqueeze(1)
        return local_keypoints
   
    # def localize_keypoints(self, keypoints: torch.Tensor) -> torch.Tensor:
    #     """
    #     Convert global keypoints to a local frame.

    #     This method anchors the commanded motion to the robot's current position. It defines a
    #     reference point on the commanded motion (the average position of feet in contact) and
    #     aligns this point with the robot's current root position on the XY plane. The orientation
    #     is also aligned based on the difference between commanded and current root yaw.

    #     If no feet are commanded to be in contact, it falls back to using the commanded root
    #     position as the anchor.

    #     Args:
    #         keypoints (torch.Tensor): A tensor of globally commanded keypoint positions for the
    #                                   current timestep. Shape: (num_envs, num_keypoints, 3).

    #     Returns:
    #         torch.Tensor: The localized keypoint positions. Shape: (num_envs, num_keypoints, 3).
    #     """
    #     # -- 1. Define the TARGET reference frame: The robot's CURRENT root pose --
    #     current_root_pos = self.robot.data.root_pos_w[:, :3].clone()
    #     current_root_pos[:, 2] = 0.0  # Project to XY plane for 2D alignment
    #     current_root_quat = self.robot.data.root_quat_w[:, :4].clone()
    #     _, _, current_root_yaw = math_utils.euler_xyz_from_quat(current_root_quat)

    #     # -- 2. Define the SOURCE reference frame: The COMMANDED center of contact --
    #     # Get desired contact state from the command manager for the current timestep [:, 0]
    #     desired_contacts_prob = self.foot_contacts[:, 0]
    #     commanded_contacts_bool = desired_contacts_prob > 0.5
    #     num_commanded_contacts = commanded_contacts_bool.float().sum(dim=1, keepdim=True)
    #     has_commanded_contact = (num_commanded_contacts.squeeze() > 0)

    #     # The origin of the source frame is the commanded center-of-contact.
    #     # Default to commanded root position if no feet are supposed to be in contact.
    #     commanded_center_pos = self._root_pos[:, 0, :3].clone()
    #     if torch.any(has_commanded_contact):
    #         commanded_foot_pos = keypoints[:, self.foot_keypoint_ids, :]
    #         sum_commanded_pos = (commanded_foot_pos * commanded_contacts_bool.unsqueeze(-1)).sum(dim=1)
    #         # Use clamp(min=1.0) to avoid division by zero
    #         avg_commanded_pos = sum_commanded_pos / torch.clamp(num_commanded_contacts, min=1.0)
    #         # Overwrite the default position for environments with commanded contacts
    #         commanded_center_pos[has_commanded_contact] = avg_commanded_pos[has_commanded_contact]

    #     commanded_center_pos[:, 2] = 0.0 # Project to XY plane

    #     # The orientation of the source frame is always the commanded root orientation.
    #     commanded_root_rot = self._root_rot[:, 0, :4].clone()
    #     commanded_root_rot = commanded_root_rot[:, [3, 0, 1, 2]]  # xyzw -> wxyz
    #     _, _, commanded_root_yaw = math_utils.euler_xyz_from_quat(commanded_root_rot)

    #     # -- 3. Calculate and apply the transformation --
    #     # Find the yaw difference to align orientations
    #     yaw_diff = math_utils.wrap_to_pi(current_root_yaw - commanded_root_yaw)
    #     yaw_rotation = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw_diff), torch.zeros_like(yaw_diff), yaw_diff)

    #     # Apply the transformation:
    #     # 1. Translate keypoints relative to the source frame's origin (commanded contact center)
    #     keypoints_in_origin = keypoints.clone() - commanded_center_pos.unsqueeze(1)
    #     # 2. Rotate them by the yaw difference
    #     rotated_keypoints = math_utils.quat_rotate(
    #         yaw_rotation.unsqueeze(1).repeat(1, keypoints_in_origin.shape[1], 1),
    #         keypoints_in_origin
    #     )
    #     # 3. Translate the rotated points to the target frame's origin (current root position)
    #     local_keypoints = rotated_keypoints + current_root_pos.unsqueeze(1)

    #     return local_keypoints
        
    def _debug_vis_callback(self, event):
        # return
        """Callback for debug visualization."""
        # add an offset to the marker position to visualize the goal
        marker_pos = self._keypoints[:,0, :, :3].reshape(self.num_envs * 14, 3).clone()
        marker_quat = torch.zeros(self.num_envs * 14, 4, device=self.device)
        marker_quat[:, 0] = 1.0  # identity quaternion for no rotation
        
        # keypoint_body_indices = [
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
        # keypoint_body_names = ['pelvis', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_roll_link', 'torso_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_yaw_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_yaw_link']
        current_pos = self.robot.data.body_pos_w[:, self.keypoint_body_ids, :].reshape(-1, 3)
        current_quat = self.robot.data.body_quat_w[:, self.keypoint_body_ids, :].reshape(-1, 4)
        
        # # local corrections
        # current_root_pos = self.robot.data.root_pos_w[:, :3].clone()
        # current_root_pos[:, 2] = 0.0  # ignore z position
        # current_root_rot = self.robot.data.root_quat_w[:, :4].clone()
        # _, _, current_yaw = math_utils.euler_xyz_from_quat(current_root_rot)
        # commanded_root_pos = self._root_pos[:,0, :3].clone()
        # commanded_root_pos[:, 2] = 0.0  # ignore z position
        # commanded_root_rot = self._root_rot[:,0, :4].clone()
        # commanded_root_rot = commanded_root_rot[:, [3, 0, 1, 2]]
        # _, _, commanded_yaw = math_utils.euler_xyz_from_quat(commanded_root_rot)
        # yaw_diff = math_utils.wrap_to_pi(current_yaw - commanded_yaw)
        # yaw_rotation = math_utils.quat_from_euler_xyz(torch.zeros_like(current_yaw), torch.zeros_like(current_yaw), yaw_diff)
        # commanded_keypoints_local = self._keypoints[:,0, :, :3].clone() - commanded_root_pos.unsqueeze(1)
        # commanded_keypoints_local = math_utils.quat_rotate(
        #     yaw_rotation.unsqueeze(1).repeat(1, commanded_keypoints_local.shape[1], 1),
        #     commanded_keypoints_local
        # )
        # local_marker_pos = commanded_keypoints_local + current_root_pos.unsqueeze(1)
        local_marker_pos = self.localize_keypoints(self._keypoints[:,0, :, :3].clone())
        local_marker_pos = local_marker_pos.reshape(self.num_envs * 14, 3)
        local_marker_quat = torch.zeros(self.num_envs * 14, 4, device=self.device)
        local_marker_quat[:, 0] = 1.0  # identity quaternion for no rotation

        # # local corrections with a horizon
        # num_marker_steps = self.horizon_length // 10
        # local_horizon_marker_pos = torch.zeros(self.num_envs * 14 * num_marker_steps, 3, device=self.device)
        # local_horizon_marker_quat = torch.zeros(self.num_envs * 14 * num_marker_steps, 4, device=self.device)
        # for i in range(num_marker_steps):
        #     h = i * (self.horizon_length // num_marker_steps)
        #     # Get the past root positions and rotations from the command manager
        #     horizon_past_current_root_pos = self.current_root_pose_buf[:, h, :3].clone()
        #     horizon_past_current_root_pos[:, 2] = 0.0 # ignore z position
        #     horizon_past_current_root_rot = self.current_root_pose_buf[:, h, 3:7].clone()
        #     _, _, horizon_past_current_yaw = math_utils.euler_xyz_from_quat(horizon_past_current_root_rot)
            
        #     # Get the past commanded root positions and rotations from the command manager
        #     horizon_past_commanded_root_pos = self.commanded_root_pose_buf[:, h, :3].clone()
        #     horizon_past_commanded_root_pos[:, 2] = 0.0 # ignore z position
        #     horizon_past_commanded_root_rot = self.commanded_root_pose_buf[:, h, 3:7].clone()
        #     horizon_past_commanded_root_rot = horizon_past_commanded_root_rot[:, [3, 0, 1, 2]]
        #     _, _, horizon_past_commanded_yaw = math_utils.euler_xyz_from_quat(horizon_past_commanded_root_rot)

        #     # Transform the keypoints into the horizon past local frame
        #     yaw_diff = math_utils.wrap_to_pi(horizon_past_current_yaw - horizon_past_commanded_yaw)
        #     yaw_rotation = math_utils.quat_from_euler_xyz(torch.zeros_like(horizon_past_current_yaw), torch.zeros_like(horizon_past_current_yaw), yaw_diff)
            
        #     commanded_keypoints = self._keypoints[:, 0, :, :3].clone()#.reshape(self.num_envs * 14, 3)
        #     commanded_keypoints_local_horizon = commanded_keypoints - horizon_past_commanded_root_pos.unsqueeze(1)
        #     # Rotate the commanded keypoints to the local frame

        #     commanded_keypoints_local_horizon = math_utils.quat_rotate(
        #         yaw_rotation.unsqueeze(1).repeat(1, commanded_keypoints.shape[1], 1),
        #         commanded_keypoints_local_horizon,
        #     )

        #     keypoint_error_h_mae = torch.norm(commanded_keypoints_local_horizon + horizon_past_current_root_pos.unsqueeze(1) - self.robot.data.body_pos_w[:, self.keypoint_body_ids, :] , dim=-1).mean(dim=-1)
        #     # print('b', keypoint_error_h_mae)

        #     local_horizon_marker_pos[i * self.num_envs * 14:(i + 1) * self.num_envs * 14, :] = (commanded_keypoints_local_horizon + horizon_past_current_root_pos.unsqueeze(1)).reshape(self.num_envs * 14, 3)
        #     local_horizon_marker_quat[i * self.num_envs * 14:(i + 1) * self.num_envs * 14, :] = torch.zeros(self.num_envs * 14, 4, device=self.device)
        #     local_horizon_marker_quat[i * self.num_envs * 14:(i + 1) * self.num_envs * 14, 0] = 1.0

        # print('b', keypoint_error_h_mae[:10])
        #     print(h, horizon_past_current_root_pos, horizon_past_commanded_root_pos)

        # input()

        # visualize the markers
        # self.goal_pose_visualizer.visualize(translations=marker_pos, orientations=marker_quat)
        self.local_goal_pose_visualizer.visualize(translations=local_marker_pos, orientations=local_marker_quat)
        # self.local_horizon_goal_pose_visualizer.visualize(translations=local_horizon_marker_pos, orientations=local_horizon_marker_quat)
        self.current_pose_visualizer.visualize(translations=current_pos, orientations=current_quat)

    # def _build_wbc_obs(self, reference_command):
    #     if self._env.common_step_counter % self.base_policy_decimation == 0:
    #         obs = [
    #             self.robot.data.projected_gravity_b, # gravity vector
    #             reference_command, # reference command
    #             self.robot.data.joint_pos - self.robot.data.default_joint_pos, # joint pos
    #             # self._env.action_manager.action, # last action
    #             self.last_policy_actions,
    #         ]
    #         obs = torch.cat(obs, dim=1)
    #         self.obs_history_buf = torch.cat((self.obs_history_buf[:, 63:], obs), dim=1)
        
    #     # print(self.obs_history_buf[0, :63])
        
    #     return self.obs_history_buf
    
    @property
    def policy_outputs(self):
        return self._policy_outputs

    @property
    def root_state(self):
        """Return the current root state of the motion."""
        return {
            "root_pos": self._root_pos,
            "root_rot": self._root_rot,
            "root_vel": self._root_vel,
            "root_ang_vel": self._root_ang_vel,
            "root_vel_global": self._root_vel_global,
            "root_ang_vel_global": self._root_ang_vel_global,
            "root_pitch": self._root_pitch,
            "root_roll": self._root_roll,
            "root_yaw": self._root_yaw,
            "projected_gravity_b": self._root_gravity_vec,
        }
        
    @property
    def foot_contacts(self):
        """Return the current foot contacts."""
        return self._foot_contacts
    
    @property
    def dof_pos(self):
        """Return the target joint positions."""
        return self._dof_pos

    @property
    def dof_vel(self):
        """Return the target joint velocities."""
        return self._dof_vel
    
    @property
    def relative_timestamp(self):
        return self._relative_timestamp
    
    # @property
    # def true_dof_pos(self):
    #     """Return the true joint positions."""
    #     return self._true_dof_pos
    
    # @property
    # def true_dof_vel(self):
    #     """Return the true joint velocities."""
    #     return self._true_dof_vel

    # @property
    # def next_true_dof_pos(self):
    #     """Return the true joint positions."""
    #     return self._next_true_dof_pos
    
    # @property
    # def next_true_dof_vel(self):
    #     """Return the true joint velocities."""
    #     return self._next_true_dof_vel
    
    # @property
    # def prev_true_dof_pos(self):
    #     """Return the true joint positions."""
    #     return self._prev_true_dof_pos
    
    # @property
    # def prev_true_dof_vel(self):
    #     """Return the true joint velocities."""
    #     return self._prev_true_dof_vel
    
    @property
    def keypoints(self):
        """Return the keypoints."""
        return self._keypoints
    
    @property
    def keypoint_rotations(self):
        """Return the keypoint rotations."""
        return self._keypoint_rotations

    # if the above gets overridden, provide the original reference data through separate properties
    @property
    def original_root_state(self) -> dict[str, torch.Tensor]:
        """Return the current **original** root state of the motion."""
        return {
            "root_pos": self._root_pos,
            "root_rot": self._root_rot,
            "root_vel": self._root_vel,
            "root_ang_vel": self._root_ang_vel,
            "root_vel_global": self._root_vel_global,
            "root_ang_vel_global": self._root_ang_vel_global,
            "root_pitch": self._root_pitch,
            "root_roll": self._root_roll,
            "root_yaw": self._root_yaw,
            "projected_gravity_b": self._root_gravity_vec,
        }
    
    @property
    def original_dof_pos(self) -> torch.Tensor:
        """Return the **original** target joint positions."""
        return self._dof_pos
    
    @property
    def original_dof_vel(self) -> torch.Tensor:
        """Return the **original** target joint velocities."""
        return self._dof_vel
    
    @property
    def original_foot_contacts(self) -> torch.Tensor:
        """Return the **original** foot contact states."""
        return self._foot_contacts
    
    @property
    def original_keypoints(self):
        """Return the **original** keypoints."""
        return self._keypoints

    @property
    def original_keypoint_rotations(self):
        """Return the **original** keypoint rotations."""
        return self._keypoint_rotations
