"""Generate diagnostic-only low-disturbance transitions with a frozen policy.

This script writes a fixed ``.pt`` dataset and is not the online compliance
training path.  The StandCoordinationOnlineCompliant task generates targets
and consumes them on-policy inside the same PPO rollout.
"""

# new change
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from collections import defaultdict

import torch


# new change
_PROJECT_ROOT = pathlib.Path(__file__).absolute().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "softmimic_gym"))

from isaaclab.app import AppLauncher


# new change
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-G1-StandCoordinationLowDisturbanceData-100Pct-29DOF-v0",
)
parser.add_argument("--jit_policy", type=pathlib.Path, required=True)
parser.add_argument(
    "--jit_extra_obs_groups",
    nargs="*",
    default=("left_arm_adapter", "right_arm_adapter"),
)
parser.add_argument("--output_dir", type=pathlib.Path, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--clean_fraction", type=float, default=0.40)
parser.add_argument("--force_min_n", type=float, default=2.0)
parser.add_argument("--force_max_n", type=float, default=8.0)
parser.add_argument("--warmup_steps", type=int, default=50)
parser.add_argument("--burst_min_steps", type=int, default=10)
parser.add_argument("--burst_max_steps", type=int, default=25)
parser.add_argument("--gap_min_steps", type=int, default=50)
parser.add_argument("--gap_max_steps", type=int, default=100)
parser.add_argument("--recovery_window_steps", type=int, default=50)
parser.add_argument("--chunk_steps", type=int, default=25)
parser.add_argument(
    "--force_body_names",
    nargs="*",
    default=(
        "torso_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# new change
def _validate_args() -> None:
    if args.num_envs <= 0 or args.steps <= 0 or args.chunk_steps <= 0:
        raise ValueError("--num_envs, --steps and --chunk_steps must be positive.")
    if not 0.0 <= args.clean_fraction <= 1.0:
        raise ValueError("--clean_fraction must be in [0, 1].")
    if args.force_min_n < 0.0 or args.force_max_n < args.force_min_n:
        raise ValueError("Expected 0 <= force_min_n <= force_max_n.")
    if args.burst_min_steps <= 0 or args.burst_max_steps < args.burst_min_steps:
        raise ValueError("Invalid burst step range.")
    if args.gap_min_steps <= 0 or args.gap_max_steps < args.gap_min_steps:
        raise ValueError("Invalid gap step range.")
    if args.warmup_steps < 0 or args.recovery_window_steps < 0:
        raise ValueError("Warmup and recovery steps cannot be negative.")
    if not args.jit_policy.is_file():
        raise FileNotFoundError(args.jit_policy)


# new change
def _rand_uniform(
    shape: tuple[int, ...],
    low: float,
    high: float,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return low + (high - low) * torch.rand(
        shape, device=device, generator=generator
    )


# new change
def _randint(
    low: int,
    high_inclusive: int,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randint(
        low,
        high_inclusive + 1,
        shape,
        device=device,
        generator=generator,
    )


# new change
def _flatten_steps(values: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(values, dim=0)
    return stacked.flatten(0, 1).contiguous()


# new change
def _save_chunk(
    output_dir: pathlib.Path,
    chunk_index: int,
    buffers: dict[str, list[torch.Tensor]],
    first_step: int,
    last_step: int,
) -> pathlib.Path:
    payload = {
        "schema_version": 1,
        "chunk_index": chunk_index,
        "first_rollout_step": first_step,
        "last_rollout_step": last_step,
        "data": {name: _flatten_steps(values) for name, values in buffers.items()},
    }
    path = output_dir / f"transitions_{chunk_index:05d}.pt"
    torch.save(payload, path)
    return path


# new change
def main() -> None:
    import gymnasium as gym

    import isaaclab.utils.math as math_utils
    from isaaclab_tasks.utils import parse_env_cfg

    import softmimic_gym.tasks  # noqa: F401
    from softmimic_gym.assets.robots.g1_compat import (
        G1_SOFTMIMIC_29_JOINT_NAMES,
    )

    _validate_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.recorders = None
    env = gym.make(args.task, cfg=env_cfg)

    try:
        observations, _ = env.reset()
        unwrapped = env.unwrapped
        device = torch.device(unwrapped.device)
        robot = unwrapped.scene["robot"]
        command = unwrapped.command_manager.get_term("upper_body_joints")
        policy = torch.jit.load(str(args.jit_policy), map_location=device)
        policy.eval()

        control_ids_list, control_names = robot.find_joints(
            list(G1_SOFTMIMIC_29_JOINT_NAMES), preserve_order=True
        )
        if tuple(control_names) != tuple(G1_SOFTMIMIC_29_JOINT_NAMES):
            raise RuntimeError(f"Canonical 29DOF mapping mismatch: {control_names}")
        control_ids = torch.as_tensor(
            control_ids_list, device=device, dtype=torch.long
        )

        force_body_ids_list, force_body_names = robot.find_bodies(
            list(args.force_body_names), preserve_order=True
        )
        if tuple(force_body_names) != tuple(args.force_body_names):
            raise RuntimeError(
                f"Could not resolve requested force bodies: {force_body_names}"
            )
        force_body_ids = torch.as_tensor(
            force_body_ids_list, device=device, dtype=torch.long
        )

        torso_ids, torso_names = robot.find_bodies(
            ["torso_link"], preserve_order=True
        )
        foot_ids, foot_names = robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        if torso_names != ["torso_link"] or foot_names != [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]:
            raise RuntimeError("Could not resolve torso/support bodies.")
        torso_id = int(torso_ids[0])
        foot_body_ids = torch.as_tensor(foot_ids, device=device, dtype=torch.long)

        contact_sensor = unwrapped.scene.sensors["contact_forces"]
        contact_ids_list, contact_names = contact_sensor.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        if contact_names != ["left_ankle_roll_link", "right_ankle_roll_link"]:
            raise RuntimeError("Could not resolve support contact entries.")
        contact_ids = torch.as_tensor(
            contact_ids_list, device=device, dtype=torch.long
        )

        keypoint_ids = command.keypoint_body_ids_tensor.to(device=device)
        arm_keypoint_reference_ids = torch.tensor(
            [8, 9, 10, 11, 12, 13], device=device, dtype=torch.long
        )
        masses = robot.data.default_mass.to(device=device, dtype=torch.float32)
        total_mass = masses.sum(dim=1, keepdim=True)
        num_motions = len(command.cfg.demo_recording_path)

        rng = torch.Generator(device=device)
        rng.manual_seed(args.seed + 17041)
        all_env_ids = torch.arange(args.num_envs, device=device, dtype=torch.long)
        clean_count = int(round(args.num_envs * args.clean_fraction))
        permutation = torch.randperm(
            args.num_envs, device=device, generator=rng
        )
        clean_cohort = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
        clean_cohort[permutation[:clean_count]] = True

        episode_step = torch.zeros(args.num_envs, device=device, dtype=torch.long)
        episode_index = torch.zeros_like(episode_step)
        force_event_index = torch.full_like(episode_step, -1)
        active_body_slot = torch.full_like(episode_step, -1)
        force_end_step = torch.zeros_like(episode_step)
        recovery_age = torch.full_like(episode_step, -1)
        force_vector_w = torch.zeros(
            (args.num_envs, 3), device=device, dtype=torch.float32
        )

        def sample_next_start(mask: torch.Tensor) -> None:
            ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if ids.numel() == 0:
                return
            gap = _randint(
                args.gap_min_steps,
                args.gap_max_steps,
                (ids.numel(),),
                device=device,
                generator=rng,
            )
            next_force_start[ids] = episode_step[ids] + gap

        next_force_start = torch.full_like(episode_step, args.warmup_steps)
        perturbed_mask = ~clean_cohort
        initial_ids = torch.nonzero(perturbed_mask, as_tuple=False).squeeze(-1)
        if initial_ids.numel() > 0:
            initial_delay = _randint(
                0,
                args.gap_max_steps - args.gap_min_steps,
                (initial_ids.numel(),),
                device=device,
                generator=rng,
            )
            next_force_start[initial_ids] = args.warmup_steps + initial_delay

        metadata = {
            "schema_version": 1,
            "task": args.task,
            "policy_path": str(args.jit_policy.resolve()),
            "policy_sha256": hashlib.sha256(args.jit_policy.read_bytes()).hexdigest(),
            "seed": args.seed,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "control_dt_s": float(unwrapped.step_dt),
            "clean_fraction_requested": args.clean_fraction,
            "clean_env_count": int(clean_cohort.sum().item()),
            "perturbed_env_count": int((~clean_cohort).sum().item()),
            "force_min_n": args.force_min_n,
            "force_max_n": args.force_max_n,
            "force_frame": "world",
            "force_body_names": list(force_body_names),
            "force_body_ids": [int(value) for value in force_body_ids_list],
            "warmup_steps": args.warmup_steps,
            "burst_step_range": [args.burst_min_steps, args.burst_max_steps],
            "gap_step_range": [args.gap_min_steps, args.gap_max_steps],
            "recovery_window_steps": args.recovery_window_steps,
            "joint_names_29dof": list(G1_SOFTMIMIC_29_JOINT_NAMES),
            "arm_keypoint_names": [
                "left_shoulder",
                "left_elbow",
                "left_wrist",
                "right_shoulder",
                "right_elbow",
                "right_wrist",
            ],
            "motion_names": [
                pathlib.Path(path).stem
                for path in command.cfg.demo_recording_path
            ],
            "jit_extra_obs_groups": list(args.jit_extra_obs_groups),
            "data_layout": "step-major then environment-major within each chunk",
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        buffers: dict[str, list[torch.Tensor]] = defaultdict(list)
        chunk_index = 0
        chunk_first_step = 0
        saved_paths: list[str] = []

        for rollout_step in range(args.steps):
            ended = (
                perturbed_mask
                & (active_body_slot >= 0)
                & (episode_step >= force_end_step)
            )
            if torch.any(ended):
                force_vector_w[ended] = 0.0
                active_body_slot[ended] = -1
                recovery_age[ended] = 0
                sample_next_start(ended)

            start = (
                perturbed_mask
                & (active_body_slot < 0)
                & (episode_step >= next_force_start)
                & (args.force_max_n > 0.0)
            )
            start_ids = torch.nonzero(start, as_tuple=False).squeeze(-1)
            if start_ids.numel() > 0:
                active_body_slot[start_ids] = _randint(
                    0,
                    len(force_body_names) - 1,
                    (start_ids.numel(),),
                    device=device,
                    generator=rng,
                )
                magnitude = _rand_uniform(
                    (start_ids.numel(),),
                    args.force_min_n,
                    args.force_max_n,
                    device=device,
                    generator=rng,
                )
                angle = _rand_uniform(
                    (start_ids.numel(),),
                    -math.pi,
                    math.pi,
                    device=device,
                    generator=rng,
                )
                force_vector_w[start_ids, 0] = magnitude * torch.cos(angle)
                force_vector_w[start_ids, 1] = magnitude * torch.sin(angle)
                force_vector_w[start_ids, 2] = 0.0
                duration = _randint(
                    args.burst_min_steps,
                    args.burst_max_steps,
                    (start_ids.numel(),),
                    device=device,
                    generator=rng,
                )
                force_end_step[start_ids] = episode_step[start_ids] + duration
                force_event_index[start_ids] += 1
                recovery_age[start_ids] = -1

            force_active = active_body_slot >= 0
            force_tensor = torch.zeros(
                (args.num_envs, len(force_body_names), 3),
                device=device,
                dtype=torch.float32,
            )
            active_ids = torch.nonzero(force_active, as_tuple=False).squeeze(-1)
            if active_ids.numel() > 0:
                force_tensor[
                    active_ids, active_body_slot[active_ids]
                ] = force_vector_w[active_ids]
            torque_tensor = torch.zeros_like(force_tensor)
            robot.permanent_wrench_composer.set_forces_and_torques(
                forces=force_tensor,
                torques=torque_tensor,
                body_ids=force_body_ids,
                env_ids=all_env_ids,
                is_global=True,
            )

            with torch.inference_mode():
                policy_input_parts = [observations["policy"]]
                policy_input_parts.extend(
                    observations[group_name]
                    for group_name in args.jit_extra_obs_groups
                )
                policy_input = torch.cat(policy_input_parts, dim=-1)
                nominal_action = policy(policy_input)

            actual_q = robot.data.joint_pos.index_select(1, control_ids)
            actual_dq = robot.data.joint_vel.index_select(1, control_ids)
            reference_q = command.dof_pos[:, 1]
            reference_dq = command.dof_vel[:, 1]
            root_state = robot.data.root_state_w
            torso_state = robot.data.body_state_w[:, torso_id]

            desired_keypoints = command.keypoints[:, 1]
            current_keypoints = robot.data.body_pos_w[:, keypoint_ids, :3]
            current_root = robot.data.root_pos_w[:, :3].clone()
            desired_root = command.root_state["root_pos"][:, 1, :3].clone()
            current_root[:, 2] = 0.0
            desired_root[:, 2] = 0.0
            _, _, current_yaw = math_utils.euler_xyz_from_quat(
                robot.data.root_quat_w
            )
            desired_quat = command.root_state["root_rot"][:, 1, [3, 0, 1, 2]]
            _, _, desired_yaw = math_utils.euler_xyz_from_quat(desired_quat)
            yaw_delta = math_utils.wrap_to_pi(current_yaw - desired_yaw)
            yaw_rotation = math_utils.quat_from_euler_xyz(
                torch.zeros_like(yaw_delta),
                torch.zeros_like(yaw_delta),
                yaw_delta,
            )
            desired_local = math_utils.quat_apply(
                yaw_rotation.unsqueeze(1).expand(-1, desired_keypoints.shape[1], -1),
                desired_keypoints - desired_root.unsqueeze(1),
            )
            current_local = current_keypoints - current_root.unsqueeze(1)
            arm_keypoint_error = torch.linalg.vector_norm(
                current_local.index_select(1, arm_keypoint_reference_ids)
                - desired_local.index_select(1, arm_keypoint_reference_ids),
                dim=-1,
            )

            body_com = robot.data.body_com_pos_w.to(dtype=torch.float32)
            whole_body_com = (body_com * masses.unsqueeze(-1)).sum(dim=1) / total_mass
            foot_pos = robot.data.body_com_pos_w.index_select(1, foot_body_ids)
            support_center = foot_pos[:, :, :2].mean(dim=1)
            com_offset_w = whole_body_com[:, :2] - support_center
            com_offset_3d = torch.cat(
                (com_offset_w, torch.zeros_like(com_offset_w[:, :1])), dim=1
            )
            robot_yaw_quat = math_utils.quat_from_euler_xyz(
                torch.zeros_like(current_yaw),
                torch.zeros_like(current_yaw),
                current_yaw,
            )
            com_support_offset_local = math_utils.quat_apply_inverse(
                robot_yaw_quat, com_offset_3d
            )[:, :2]

            gravity_w = torch.zeros(
                (args.num_envs, 3), device=device, dtype=torch.float32
            )
            gravity_w[:, 2] = -1.0
            torso_gravity = math_utils.quat_apply_inverse(
                robot.data.body_quat_w[:, torso_id], gravity_w
            )
            torso_tilt_rad = torch.acos(
                torch.clamp(-torso_gravity[:, 2], min=-1.0, max=1.0)
            )
            foot_xy_speed = torch.linalg.vector_norm(
                robot.data.body_com_lin_vel_w.index_select(1, foot_body_ids)[
                    :, :, :2
                ],
                dim=-1,
            )
            foot_contact_force = (
                contact_sensor.data.net_forces_w_history.index_select(2, contact_ids)
                .norm(dim=-1)
                .max(dim=1)
                .values
            )

            source_time_s = command.motion_count * command.motion_dt
            motion_id = command.motion_ids % num_motions
            pre_step = {
                "rollout_step": torch.full_like(episode_step, rollout_step),
                "env_id": all_env_ids,
                "episode_index": episode_index,
                "episode_step": episode_step,
                "motion_id": motion_id,
                "source_time_s": source_time_s,
                "clean_cohort": clean_cohort,
                "force_active": force_active,
                "force_body_slot": active_body_slot,
                "force_event_index": force_event_index,
                "force_w": force_vector_w,
                "torque_w": torch.zeros_like(force_vector_w),
                "recovery_age_steps": recovery_age,
                "policy_observation": observations["policy"],
                "left_arm_adapter_observation": observations["left_arm_adapter"],
                "right_arm_adapter_observation": observations["right_arm_adapter"],
                "nominal_action": nominal_action,
                "joint_q": actual_q,
                "joint_dq": actual_dq,
                "reference_q": reference_q,
                "reference_dq": reference_dq,
                "root_state_w": root_state,
                "torso_state_w": torso_state,
                "arm_keypoint_error_m": arm_keypoint_error,
                "whole_body_com_w": whole_body_com,
                "com_support_offset_local_m": com_support_offset_local,
                "torso_projected_gravity": torso_gravity,
                "torso_tilt_rad": torso_tilt_rad,
                "foot_xy_speed_mps": foot_xy_speed,
                "foot_contact_force_n": foot_contact_force,
            }

            next_observations, reward, terminated, truncated, _ = env.step(
                nominal_action
            )
            bad_orientation = unwrapped.termination_manager.get_term(
                "bad_orientation"
            ).bool()
            base_contact = unwrapped.termination_manager.get_term(
                "base_contact"
            ).bool()
            done = terminated | truncated

            post_step = {
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "bad_orientation": bad_orientation,
                "base_contact": base_contact,
                "next_policy_observation": next_observations["policy"],
                "next_joint_q": robot.data.joint_pos.index_select(1, control_ids),
                "next_joint_dq": robot.data.joint_vel.index_select(1, control_ids),
                "next_root_state_w": robot.data.root_state_w,
                "joint_position_target": robot.data.joint_pos_target.index_select(
                    1, control_ids
                ),
            }
            pre_step.update(post_step)
            for name, value in pre_step.items():
                buffers[name].append(value.detach().cpu())

            episode_step += 1
            recovering = (
                perturbed_mask
                & (~force_active)
                & (recovery_age >= 0)
                & (recovery_age < args.recovery_window_steps)
            )
            recovery_age[recovering] += 1

            if torch.any(done):
                episode_index[done] += 1
                episode_step[done] = 0
                force_event_index[done] = -1
                active_body_slot[done] = -1
                force_end_step[done] = 0
                force_vector_w[done] = 0.0
                recovery_age[done] = -1
                done_perturbed = done & perturbed_mask
                done_ids = torch.nonzero(
                    done_perturbed, as_tuple=False
                ).squeeze(-1)
                if done_ids.numel() > 0:
                    reset_delay = _randint(
                        0,
                        args.gap_max_steps - args.gap_min_steps,
                        (done_ids.numel(),),
                        device=device,
                        generator=rng,
                    )
                    next_force_start[done_ids] = args.warmup_steps + reset_delay

            observations = next_observations

            should_flush = (
                len(next(iter(buffers.values()))) >= args.chunk_steps
                or rollout_step + 1 == args.steps
            )
            if should_flush:
                chunk_path = _save_chunk(
                    args.output_dir,
                    chunk_index,
                    buffers,
                    chunk_first_step,
                    rollout_step,
                )
                saved_paths.append(chunk_path.name)
                samples = len(next(iter(buffers.values()))) * args.num_envs
                print(
                    f"[LOW_DIST_DATA] saved={chunk_path.name} samples={samples} "
                    f"steps={chunk_first_step}-{rollout_step}",
                    flush=True,
                )
                buffers = defaultdict(list)
                chunk_index += 1
                chunk_first_step = rollout_step + 1

        zero_force_tensor = torch.zeros(
            (args.num_envs, len(force_body_names), 3),
            device=device,
            dtype=torch.float32,
        )
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=zero_force_tensor,
            torques=zero_force_tensor,
            body_ids=force_body_ids,
            env_ids=all_env_ids,
            is_global=True,
        )
        metadata["chunks"] = saved_paths
        metadata["total_transitions"] = args.steps * args.num_envs
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[LOW_DIST_DATA] complete transitions={metadata['total_transitions']} "
            f"chunks={len(saved_paths)} output_dir={args.output_dir}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
