"""Construct GeneralTrackingV1 and execute a short zero-residual rollout."""

from __future__ import annotations

import argparse
import pathlib
import sys

# new change
# The user's IsaacLab environment may have the original softmimic checkout
# installed editable.  Prefer this project's source without changing that
# external installation or breaking the archived project.
_LOCAL_EXTENSION_ROOT = pathlib.Path(__file__).absolute().parents[1] / "softmimic_gym"
sys.path.insert(0, str(_LOCAL_EXTENSION_ROOT))

from isaaclab.app import AppLauncher


# new change
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--verbose_steps", action="store_true")
# new change
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-G1-GeneralTrackingV1-29DOF-v0",
    help="GeneralTrackingV1 curriculum task to validate.",
)
# new change
parser.add_argument("--jit_policy", type=str, default=None)
# new change
parser.add_argument(
    "--compare_jit_policy",
    type=str,
    default=None,
    help="Optional 705D baseline JIT used only to measure action deltas.",
)
# new change
parser.add_argument(
    "--compare_jit_extra_obs_groups",
    nargs="*",
    default=(),
    help="Structured observation groups appended for the comparison JIT.",
)
# new change
parser.add_argument(
    "--adapter_feature_pt",
    type=str,
    default=None,
    help="Save left/right 37D structured features for offline distillation.",
)
# new change
parser.add_argument(
    "--support_adapter_feature_pt",
    type=str,
    default=None,
    help=(
        "Save online 79D support features and bounded 10DOF analytic teacher "
        "targets for offline support-adapter distillation."
    ),
)
# new change
parser.add_argument(
    "--support_teacher_gain",
    type=float,
    default=0.10,
    help="Analytic teacher gain applied to the online q-q_ref support error.",
)
# new change
parser.add_argument(
    "--support_teacher_limit_rad",
    type=float,
    default=0.006,
    help="Per-joint absolute physical target limit for the support teacher.",
)
# new change
parser.add_argument(
    "--jit_extra_obs_groups",
    nargs="*",
    default=(),
    help="Append structured groups to policy observations in the listed order.",
)
# new change
parser.add_argument(
    "--residual_sample_hold_steps",
    type=int,
    default=0,
    help=(
        "With --compare_jit_policy, sample candidate-minus-baseline once per "
        "N steps and hold it, matching a local finite-horizon teacher."
    ),
)
# new change
parser.add_argument(
    "--residual_active_steps",
    type=int,
    default=0,
    help="Optional active part of each sample/hold cycle; zero means full cycle.",
)
# new change
parser.add_argument(
    "--temporal_support_jit",
    type=str,
    default=None,
    help=(
        "Standalone 98D support adapter.  Its 10DOF output is added to the "
        "frozen base policy without modifying any PPO checkpoint."
    ),
)
# new change
parser.add_argument(
    "--temporal_support_update_steps",
    type=int,
    default=25,
    help="Number of policy steps between time-context support updates.",
)
# new change
parser.add_argument("--seed", type=int, default=42)
# new change
parser.add_argument(
    "--state_stats_json",
    type=str,
    default=None,
    help="Optionally save aggregate actual/reference joint and root state statistics.",
)
# new change
parser.add_argument(
    "--state_stats_start",
    type=int,
    default=0,
    help="First zero-based rollout step included in --state_stats_json.",
)
# new change
parser.add_argument(
    "--exact_reference_reset",
    action="store_true",
    help="Remove reset perturbations for a q_ref-to-USD kinematic audit.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import torch
    import traceback
    # new change
    from pathlib import Path

    from isaaclab_tasks.utils import parse_env_cfg
    # new change
    from isaaclab.utils import math as math_utils

    import softmimic_gym.tasks  # noqa: F401
    # new change
    from softmimic_gym.assets.robots.g1_compat import G1_SOFTMIMIC_29_JOINT_NAMES

    # new change
    task_id = args.task
    env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=args.num_envs)
    # new change
    env_cfg.seed = args.seed
    # new change
    if args.exact_reference_reset:
        env_cfg.events.reset_robot.params["jpos_range"] = (0.0, 0.0)
        env_cfg.events.reset_robot.params["jvel_range"] = (0.0, 0.0)
        for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
            env_cfg.events.reset_robot.params["pose_range"][axis] = (0.0, 0.0)
        print("[SMOKE] exact_reference_reset=True")
    env_cfg.recorders = None
    env = gym.make(task_id, cfg=env_cfg)
    try:
        observations, _ = env.reset()
        policy_obs = observations["policy"]
        action_dim = env.unwrapped.action_manager.total_action_dim
        print(
            f"[SMOKE] policy_obs={tuple(policy_obs.shape)} "
            f"action_dim={action_dim} robot_joints={env.unwrapped.scene['robot'].num_joints}"
        )
        if action_dim != 29:
            raise RuntimeError(f"Expected 29 policy actions, got {action_dim}.")
        if env.unwrapped.scene["robot"].num_joints != 43:
            raise RuntimeError("Expected the official 43-joint G1 articulation.")

        action = torch.zeros((args.num_envs, action_dim), device=env.unwrapped.device)
        # new change
        policy = None
        # new change
        compare_policy = None
        if args.jit_policy is not None:
            policy = torch.jit.load(args.jit_policy, map_location=env.unwrapped.device)
            policy.eval()
            print(f"[SMOKE] loaded_jit_policy={args.jit_policy}")
        # new change
        if args.compare_jit_policy is not None:
            compare_policy = torch.jit.load(
                args.compare_jit_policy, map_location=env.unwrapped.device
            )
            compare_policy.eval()
            print(f"[SMOKE] compare_jit_policy={args.compare_jit_policy}")
        # new change
        temporal_support_policy = None
        if args.temporal_support_jit is not None:
            temporal_support_policy = torch.jit.load(
                args.temporal_support_jit, map_location=env.unwrapped.device
            ).eval()
            print(f"[SMOKE] temporal_support_jit={args.temporal_support_jit}")
        if args.temporal_support_update_steps < 1:
            raise ValueError("--temporal_support_update_steps must be positive.")
        action_delta_sum = torch.zeros(action_dim, device=env.unwrapped.device)
        action_delta_max = torch.zeros(action_dim, device=env.unwrapped.device)
        # new change
        held_residual = torch.zeros_like(action)
        # new change
        support_action_ids = torch.tensor(
            (0, 1, 9, 10, 13, 14, 3, 4, 17, 18),
            device=env.unwrapped.device,
            dtype=torch.long,
        )
        temporal_held_actor = torch.zeros(
            (args.num_envs, 10), device=env.unwrapped.device
        )
        temporal_previous_rad = torch.zeros_like(temporal_held_actor)
        temporal_elapsed_steps = torch.zeros(
            args.num_envs, device=env.unwrapped.device, dtype=torch.long
        )
        temporal_needs_update = torch.ones(
            args.num_envs, device=env.unwrapped.device, dtype=torch.bool
        )
        temporal_saturation_count = torch.zeros(
            10, device=env.unwrapped.device
        )
        if args.residual_sample_hold_steps < 0:
            raise ValueError("--residual_sample_hold_steps must be non-negative.")
        if args.residual_sample_hold_steps and compare_policy is None:
            raise ValueError("Sample/hold residual requires --compare_jit_policy.")
        if args.residual_active_steps < 0 or (
            args.residual_sample_hold_steps
            and args.residual_active_steps > args.residual_sample_hold_steps
        ):
            raise ValueError("Invalid --residual_active_steps.")
        # new change
        adapter_feature_batches = []
        # new change
        support_feature_batches = []
        support_target_actor_batches = []
        support_target_rad_batches = []
        # new change
        adapter_motion_id_batches = []
        adapter_source_time_batches = []
        terminated_total = torch.zeros(args.num_envs, device=env.unwrapped.device)
        truncated_total = torch.zeros(args.num_envs, device=env.unwrapped.device)
        # new change
        termination_counts = {
            name: torch.zeros(args.num_envs, device=env.unwrapped.device)
            for name in env.unwrapped.termination_manager.active_terms
        }
        # new change
        # Keep the first failure separate from repeated post-reset failures.
        # This makes motion-phase diagnosis meaningful for dynamic clips.
        first_bad_step = torch.full(
            (args.num_envs,), args.steps + 1, device=env.unwrapped.device, dtype=torch.long
        )
        first_bad_phase = torch.full(
            (args.num_envs,), float("nan"), device=env.unwrapped.device
        )
        # new change
        q_error_sum = torch.zeros((args.num_envs, 29), device=env.unwrapped.device)
        q_error_max = torch.zeros((args.num_envs, 29), device=env.unwrapped.device)
        # new change
        state_stats_count = 0
        state_q_sum = torch.zeros(29, device=env.unwrapped.device, dtype=torch.float64)
        state_q_sq_sum = torch.zeros_like(state_q_sum)
        state_ref_sum = torch.zeros_like(state_q_sum)
        state_ref_sq_sum = torch.zeros_like(state_q_sum)
        state_root_pos_sum = torch.zeros(3, device=env.unwrapped.device, dtype=torch.float64)
        state_gravity_sum = torch.zeros_like(state_root_pos_sum)
        # new change
        keypoint_error_sum = torch.zeros(args.num_envs, device=env.unwrapped.device)
        keypoint_error_max = torch.zeros(args.num_envs, device=env.unwrapped.device)
        minimum_root_height = torch.full(
            (args.num_envs,), float("inf"), device=env.unwrapped.device
        )
        maximum_tilt_deg = torch.zeros(args.num_envs, device=env.unwrapped.device)
        # new change
        reference_tilt_error_sum = torch.zeros(args.num_envs, device=env.unwrapped.device)
        reference_tilt_error_max = torch.zeros(args.num_envs, device=env.unwrapped.device)
        root_velocity_error_sum = torch.zeros(args.num_envs, device=env.unwrapped.device)
        root_velocity_error_max = torch.zeros(args.num_envs, device=env.unwrapped.device)
        command = env.unwrapped.command_manager.get_term("upper_body_joints")
        initial_reference_phase = command.relative_timestamp.clone()
        # new change
        initial_source_time = command.motion_count.clone() * command.motion_lib._motion_dt
        first_bad_source_time = torch.full(
            (args.num_envs,), float("nan"), device=env.unwrapped.device
        )
        for step_index in range(args.steps):
            if args.verbose_steps:
                print(f"[SMOKE] stepping={step_index}", flush=True)
            try:
                # new change
                phase_before_step = command.relative_timestamp.clone()
                source_time_before_step = (
                    command.motion_count.clone() * command.motion_lib._motion_dt
                )
                if policy is not None:
                    # new change
                    # JIT evaluation must not retain one autograd graph per
                    # simulation step.  The action is fed back into the next
                    # observation, so leaving grad enabled grows GPU memory
                    # until long population rollouts fail near their end.
                    with torch.inference_mode():
                        # new change
                        if args.adapter_feature_pt is not None:
                            adapter_feature_batches.append(
                                (
                                    observations["left_arm_adapter"].detach().cpu(),
                                    observations["right_arm_adapter"].detach().cpu(),
                                )
                            )
                        # new change
                        if (
                            args.adapter_feature_pt is not None
                            or args.support_adapter_feature_pt is not None
                        ):
                            # Preserve motion context once even when both arm
                            # and support datasets are requested.
                            adapter_motion_id_batches.append(
                                (
                                    torch.arange(args.num_envs, device=env.unwrapped.device)
                                    % len(command.cfg.demo_recording_path)
                                ).detach().cpu()
                            )
                            adapter_source_time_batches.append(
                                (
                                    command.motion_count * command.motion_dt
                                ).detach().cpu()
                            )
                        # new change
                        if args.support_adapter_feature_pt is not None:
                            support_features = observations["support_adapter"].detach()
                            if support_features.shape[-1] != 79:
                                raise RuntimeError(
                                    "Support teacher expects 79D features, got "
                                    f"{support_features.shape[-1]}."
                                )
                            # Feature order begins q10,q_ref10,q-q_ref10.
                            support_error = support_features[:, 20:30]
                            support_target_rad = torch.clamp(
                                -args.support_teacher_gain * support_error,
                                min=-args.support_teacher_limit_rad,
                                max=args.support_teacher_limit_rad,
                            )
                            support_feature_batches.append(
                                support_features.cpu()
                            )
                            support_target_rad_batches.append(
                                support_target_rad.cpu()
                            )
                            support_target_actor_batches.append(
                                (support_target_rad / 0.30).cpu()
                            )
                        # new change
                        policy_input = policy_obs
                        if args.jit_extra_obs_groups:
                            policy_input = torch.cat(
                                [policy_obs]
                                + [observations[name] for name in args.jit_extra_obs_groups],
                                dim=-1,
                            )
                        action = policy(policy_input).detach()
                        # new change
                        if temporal_support_policy is not None:
                            support_features = observations["support_adapter"].detach()
                            if support_features.shape[-1] != 87:
                                raise RuntimeError(
                                    "Temporal support adapter expects 87D physical "
                                    f"features, got {support_features.shape[-1]}."
                                )
                            periodic_update = (
                                temporal_elapsed_steps
                                >= args.temporal_support_update_steps
                            )
                            update_mask = temporal_needs_update | periodic_update
                            if torch.any(update_mask):
                                elapsed_fraction = (
                                    temporal_elapsed_steps.to(support_features.dtype)
                                    / float(args.temporal_support_update_steps)
                                ).clamp(0.0, 1.0).unsqueeze(-1)
                                temporal_input = torch.cat(
                                    (
                                        support_features,
                                        temporal_previous_rad,
                                        elapsed_fraction,
                                    ),
                                    dim=-1,
                                )
                                proposed = temporal_support_policy(temporal_input).detach()
                                if proposed.shape != temporal_held_actor.shape:
                                    raise RuntimeError(
                                        "Temporal support output must be [N,10], got "
                                        f"{tuple(proposed.shape)}."
                                    )
                                temporal_held_actor[update_mask] = proposed[update_mask]
                                temporal_previous_rad[update_mask] = (
                                    proposed[update_mask] * 0.30
                                )
                                temporal_elapsed_steps[update_mask] = 0
                                temporal_needs_update[update_mask] = False
                            action[:, support_action_ids] += temporal_held_actor
                            temporal_saturation_count += (
                                temporal_held_actor.abs() > 0.0199
                            ).float().sum(0)
                        # new change
                        if compare_policy is not None:
                            compare_input = policy_obs
                            if args.compare_jit_extra_obs_groups:
                                compare_input = torch.cat(
                                    [policy_obs]
                                    + [
                                        observations[name]
                                        for name in args.compare_jit_extra_obs_groups
                                    ],
                                    dim=-1,
                                )
                            baseline_action = compare_policy(compare_input).detach()
                            # new change
                            if args.residual_sample_hold_steps:
                                cycle_step = step_index % args.residual_sample_hold_steps
                                if cycle_step == 0:
                                    held_residual.copy_(action - baseline_action)
                                active_steps = (
                                    args.residual_active_steps
                                    if args.residual_active_steps
                                    else args.residual_sample_hold_steps
                                )
                                action = (
                                    baseline_action + held_residual
                                    if cycle_step < active_steps
                                    else baseline_action
                                )
                            delta = torch.abs(action - baseline_action)
                            action_delta_sum += delta.mean(dim=0)
                            action_delta_max = torch.maximum(
                                action_delta_max, delta.max(dim=0).values
                            )
                observations, rewards, terminated, truncated, _ = env.step(action)
                # new change
                policy_obs = observations["policy"]
            except BaseException as exc:
                print(
                    f"[SMOKE][ERROR] step={step_index} "
                    f"type={type(exc).__name__} message={exc}",
                    flush=True,
                )
                # new change
                traceback.print_exc()
                raise
            terminated_total += terminated
            truncated_total += truncated
            # new change
            # Never carry a correction sampled in a failed episode into the
            # freshly reset state of that environment.
            held_residual[terminated | truncated] = 0.0
            # new change
            if temporal_support_policy is not None:
                temporal_elapsed_steps += 1
                reset_mask = terminated | truncated
                temporal_held_actor[reset_mask] = 0.0
                temporal_previous_rad[reset_mask] = 0.0
                temporal_elapsed_steps[reset_mask] = 0
                temporal_needs_update[reset_mask] = True
            # new change
            for name in termination_counts:
                termination_counts[name] += env.unwrapped.termination_manager.get_term(name)
            # new change
            bad_orientation = env.unwrapped.termination_manager.get_term("bad_orientation").bool()
            first_bad_mask = bad_orientation & (first_bad_step > args.steps)
            first_bad_step[first_bad_mask] = step_index + 1
            first_bad_phase[first_bad_mask] = phase_before_step[first_bad_mask]
            first_bad_source_time[first_bad_mask] = source_time_before_step[first_bad_mask]
            robot = env.unwrapped.scene["robot"]
            minimum_root_height = torch.minimum(
                minimum_root_height, robot.data.root_pos_w[:, 2]
            )
            upright_cosine = torch.clamp(
                -robot.data.projected_gravity_b[:, 2], -1.0, 1.0
            )
            maximum_tilt_deg = torch.maximum(
                maximum_tilt_deg, torch.rad2deg(torch.acos(upright_cosine))
            )
            # new change
            reference_gravity = torch.nn.functional.normalize(
                command.root_state["projected_gravity_b"][:, 1], dim=-1
            )
            current_gravity = torch.nn.functional.normalize(
                robot.data.projected_gravity_b, dim=-1
            )
            reference_tilt_error = torch.rad2deg(
                torch.acos(torch.sum(current_gravity * reference_gravity, dim=-1).clamp(-1.0, 1.0))
            )
            reference_tilt_error_sum += reference_tilt_error
            reference_tilt_error_max = torch.maximum(
                reference_tilt_error_max, reference_tilt_error
            )
            root_velocity_error = torch.linalg.vector_norm(
                robot.data.root_lin_vel_b - command.root_state["root_vel"][:, 1], dim=-1
            )
            root_velocity_error_sum += root_velocity_error
            root_velocity_error_max = torch.maximum(root_velocity_error_max, root_velocity_error)
            # new change
            control_ids = torch.as_tensor(
                command.control_joint_ids, device=env.unwrapped.device, dtype=torch.long
            )
            step_q_error = torch.abs(
                robot.data.joint_pos.index_select(1, control_ids) - command.dof_pos[:, 1]
            )
            q_error_sum += step_q_error
            q_error_max = torch.maximum(q_error_max, step_q_error)
            # new change
            if args.state_stats_json is not None and step_index >= args.state_stats_start:
                actual_q = robot.data.joint_pos.index_select(1, control_ids).double()
                reference_q = command.dof_pos[:, 1].double()
                state_q_sum += actual_q.sum(dim=0)
                state_q_sq_sum += actual_q.square().sum(dim=0)
                state_ref_sum += reference_q.sum(dim=0)
                state_ref_sq_sum += reference_q.square().sum(dim=0)
                state_root_pos_sum += robot.data.root_pos_w[:, :3].double().sum(dim=0)
                state_gravity_sum += robot.data.projected_gravity_b.double().sum(dim=0)
                state_stats_count += args.num_envs
            # new change
            keypoint_ids = command.keypoint_body_ids_tensor
            desired_keypoints = command.keypoints[:, 1]
            current_keypoints = robot.data.body_pos_w[:, keypoint_ids, :3]
            current_root = robot.data.root_pos_w[:, :3].clone()
            current_root[:, 2] = 0.0
            desired_root = command.root_state["root_pos"][:, 1, :3].clone()
            desired_root[:, 2] = 0.0
            _, _, current_yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w[:, :4])
            desired_quat = command.root_state["root_rot"][:, 1, [3, 0, 1, 2]]
            _, _, desired_yaw = math_utils.euler_xyz_from_quat(desired_quat)
            yaw_delta = math_utils.wrap_to_pi(current_yaw - desired_yaw)
            yaw_rotation = math_utils.quat_from_euler_xyz(
                torch.zeros_like(yaw_delta), torch.zeros_like(yaw_delta), yaw_delta
            )
            desired_local = math_utils.quat_rotate(
                yaw_rotation.unsqueeze(1).expand(-1, desired_keypoints.shape[1], -1),
                desired_keypoints - desired_root.unsqueeze(1),
            )
            current_local = current_keypoints - current_root.unsqueeze(1)
            keypoint_distance = torch.linalg.vector_norm(current_local - desired_local, dim=-1)
            step_keypoint_error = keypoint_distance.mean(dim=1)
            keypoint_error_sum += step_keypoint_error
            keypoint_error_max = torch.maximum(
                keypoint_error_max, keypoint_distance.max(dim=1).values
            )
            if args.verbose_steps:
                print(f"[SMOKE] stepped={step_index}", flush=True)
        robot = env.unwrapped.scene["robot"]
        command = env.unwrapped.command_manager.get_term("upper_body_joints")
        control_ids = torch.as_tensor(
            command.control_joint_ids, device=env.unwrapped.device, dtype=torch.long
        )
        q_error = torch.abs(
            robot.data.joint_pos.index_select(1, control_ids) - command.dof_pos[:, 1]
        )
        # new change
        rollout_q_mae = q_error_sum / args.steps
        # new change
        rollout_keypoint_mae = keypoint_error_sum / args.steps
        # new change
        rollout_reference_tilt_mae = reference_tilt_error_sum / args.steps
        rollout_root_velocity_mae = root_velocity_error_sum / args.steps
        print(
            f"[SMOKE] steps={args.steps} reward_mean={rewards.mean().item():.4f} "
            f"terminated_total={terminated_total.sum().item():.0f} "
            f"truncated_total={truncated_total.sum().item():.0f} "
            f"root_height_min={minimum_root_height.min().item():.4f} "
            f"tilt_max_deg={maximum_tilt_deg.max().item():.2f} "
            f"q_ref_mae_deg={torch.rad2deg(q_error).mean().item():.3f}",
            # new change
            flush=True,
        )
        # new change
        print(
            "[SMOKE][population] "
            + " ".join(
                f"{name}={counts.sum().item():.0f}" for name, counts in termination_counts.items()
            )
            + f" q_ref_rollout_mae_deg={torch.rad2deg(rollout_q_mae).mean().item():.3f}"
            + f" q_ref_rollout_max_deg={torch.rad2deg(q_error_max).max().item():.3f}"
            + f" keypoint_rollout_mae_m={rollout_keypoint_mae.mean().item():.4f}"
            + f" keypoint_rollout_max_m={keypoint_error_max.max().item():.4f}"
            + f" reference_tilt_mae_deg={rollout_reference_tilt_mae.mean().item():.2f}"
            + f" reference_tilt_max_deg={reference_tilt_error_max.max().item():.2f}"
            + f" root_velocity_mae_mps={rollout_root_velocity_mae.mean().item():.3f}"
            + f" root_velocity_max_mps={root_velocity_error_max.max().item():.3f}",
            # new change
            flush=True,
        )
        # new change
        if compare_policy is not None:
            action_delta_mae = action_delta_sum / args.steps
            arm_ids = torch.tensor(
                [11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                device=env.unwrapped.device,
            )
            print(
                "[SMOKE][action_delta] "
                f"all_mae_actor={action_delta_mae.mean().item():.9f} "
                f"arm_mae_actor={action_delta_mae.index_select(0, arm_ids).mean().item():.9f} "
                f"arm_max_actor={action_delta_max.index_select(0, arm_ids).max().item():.9f} "
                f"arm_mae_target_rad={(0.15 * action_delta_mae.index_select(0, arm_ids).mean()).item():.9f} "
                f"arm_max_target_rad={(0.15 * action_delta_max.index_select(0, arm_ids).max()).item():.9f}",
                flush=True,
            )
        # new change
        if temporal_support_policy is not None:
            print(
                "[SMOKE][temporal_support] "
                f"update_steps={args.temporal_support_update_steps} "
                f"residual_abs_mean_mrad="
                f"{(temporal_previous_rad.abs().mean() * 1000.0).item():.4f} "
                f"residual_abs_max_mrad="
                f"{(temporal_previous_rad.abs().max() * 1000.0).item():.4f} "
                f"saturation_fraction="
                f"{(temporal_saturation_count.sum() / max(args.steps * args.num_envs * 10, 1)).item():.6f}",
                flush=True,
            )
        # new change
        if args.adapter_feature_pt is not None:
            from pathlib import Path

            feature_path = Path(args.adapter_feature_pt)
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "left_arm_adapter": torch.cat(
                        [batch[0] for batch in adapter_feature_batches], dim=0
                    ),
                    "right_arm_adapter": torch.cat(
                        [batch[1] for batch in adapter_feature_batches], dim=0
                    ),
                    # new change
                    "motion_id": torch.cat(adapter_motion_id_batches, dim=0),
                    "source_time_s": torch.cat(adapter_source_time_batches, dim=0),
                    "feature_order": (
                        "q[7],q_ref[7],q-q_ref[7],dq[7],dq_ref[7],phase_sincos[2]"
                    ),
                    "seed": args.seed,
                    "steps": args.steps,
                    "num_envs": args.num_envs,
                },
                feature_path,
            )
            print(
                f"[SMOKE][adapter_features] saved={feature_path} "
                f"samples={len(adapter_feature_batches) * args.num_envs}",
                flush=True,
            )
        # new change
        if args.support_adapter_feature_pt is not None:
            if args.support_teacher_gain <= 0.0:
                raise ValueError("--support_teacher_gain must be positive.")
            if args.support_teacher_limit_rad <= 0.0:
                raise ValueError("--support_teacher_limit_rad must be positive.")
            support_feature_path = Path(args.support_adapter_feature_pt)
            support_feature_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "support_adapter": torch.cat(
                        support_feature_batches, dim=0
                    ),
                    "support_target_actor": torch.cat(
                        support_target_actor_batches, dim=0
                    ),
                    "support_target_rad": torch.cat(
                        support_target_rad_batches, dim=0
                    ),
                    "motion_id": torch.cat(adapter_motion_id_batches, dim=0),
                    "source_time_s": torch.cat(
                        adapter_source_time_batches, dim=0
                    ),
                    "feature_order": (
                        "q[10],q_ref[10],q-q_ref[10],dq[10],dq_ref[10],"
                        "phase[2],COM[2],gravity[3],ang_vel[3],root_xy[2],"
                        "base_vel[3],foot_vel[6],forces[2],contacts[2],"
                        "fixed_foot_xy[4]"
                    ),
                    "support_action_ids": (
                        0, 1, 9, 10, 13, 14, 3, 4, 17, 18
                    ),
                    "teacher_gain": args.support_teacher_gain,
                    "teacher_limit_rad": args.support_teacher_limit_rad,
                    "action_scale_rad": 0.30,
                    "seed": args.seed,
                    "steps": args.steps,
                    "num_envs": args.num_envs,
                },
                support_feature_path,
            )
            print(
                f"[SMOKE][support_features] saved={support_feature_path} "
                f"samples={len(support_feature_batches) * args.num_envs} "
                f"teacher_gain={args.support_teacher_gain} "
                f"teacher_limit_rad={args.support_teacher_limit_rad}",
                flush=True,
            )
        # new change
        if args.state_stats_json is not None:
            import json

            if state_stats_count <= 0:
                raise ValueError(
                    "No state samples collected; --state_stats_start must be smaller than --steps."
                )
            count = float(state_stats_count)
            actual_mean = state_q_sum / count
            reference_mean = state_ref_sum / count
            actual_std = torch.sqrt(
                torch.clamp(state_q_sq_sum / count - actual_mean.square(), min=0.0)
            )
            reference_std = torch.sqrt(
                torch.clamp(state_ref_sq_sum / count - reference_mean.square(), min=0.0)
            )
            state_path = Path(args.state_stats_json)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "task": task_id,
                        "seed": args.seed,
                        "num_envs": args.num_envs,
                        "steps": args.steps,
                        "start_step": args.state_stats_start,
                        "sample_count": state_stats_count,
                        "joint_names": list(G1_SOFTMIMIC_29_JOINT_NAMES),
                        "actual_q_mean_rad": actual_mean.cpu().tolist(),
                        "actual_q_std_rad": actual_std.cpu().tolist(),
                        "reference_q_mean_rad": reference_mean.cpu().tolist(),
                        "reference_q_std_rad": reference_std.cpu().tolist(),
                        "root_pos_mean_m": (state_root_pos_sum / count).cpu().tolist(),
                        "projected_gravity_mean": (state_gravity_sum / count).cpu().tolist(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"[SMOKE][state_stats] saved={state_path} samples={state_stats_count}",
                flush=True,
            )
        # new change
        # new change
        # ReferenceCommandCfg may intentionally leave joint_names empty because
        # the strict compatibility layer owns the canonical ordering.
        # configured_joint_names = command.cfg.joint_names
        # joint_names = (
        #     tuple(configured_joint_names)
        #     if configured_joint_names
        #     else G1_SOFTMIMIC_29_JOINT_NAMES
        # )
        # new change
        joint_names = G1_SOFTMIMIC_29_JOINT_NAMES
        for joint_index, joint_name in enumerate(joint_names):
            print(
                f"[SMOKE][joint={joint_name}] "
                f"mae_deg={torch.rad2deg(rollout_q_mae[:, joint_index]).mean().item():.3f} "
                f"max_deg={torch.rad2deg(q_error_max[:, joint_index]).max().item():.3f}",
                # new change
                flush=True,
            )
        # new change
        motion_names = tuple(Path(path).stem for path in command.cfg.demo_recording_path)
        motion_ids = torch.arange(args.num_envs, device=env.unwrapped.device) % len(motion_names)
        for motion_id, motion_name in enumerate(motion_names):
            mask = motion_ids == motion_id
            # new change
            failed_mask = mask & (first_bad_step <= args.steps)
            if torch.any(failed_mask):
                failure_stats = (
                    f" first_bad_count={failed_mask.sum().item()}"
                    f" first_bad_step_min={first_bad_step[failed_mask].min().item()}"
                    f" first_bad_step_median={first_bad_step[failed_mask].float().median().item():.1f}"
                    f" first_bad_step_max={first_bad_step[failed_mask].max().item()}"
                    f" first_bad_phase_median_s={first_bad_phase[failed_mask].median().item():.3f}"
                    f" first_bad_source_time_median_s={first_bad_source_time[failed_mask].median().item():.3f}"
                )
            else:
                failure_stats = " first_bad_count=0"
            print(
                f"[SMOKE][motion={motion_name} envs={mask.sum().item()}] "
                + " ".join(
                    f"{name}={counts[mask].sum().item():.0f}"
                    for name, counts in termination_counts.items()
                )
                + f" min_height={minimum_root_height[mask].min().item():.4f}"
                + f" max_tilt_deg={maximum_tilt_deg[mask].max().item():.2f}"
                + f" q_ref_rollout_mae_deg={torch.rad2deg(rollout_q_mae[mask]).mean().item():.3f}"
                + f" keypoint_rollout_mae_m={rollout_keypoint_mae[mask].mean().item():.4f}"
                + f" reference_tilt_mae_deg={rollout_reference_tilt_mae[mask].mean().item():.2f}"
                + f" root_velocity_mae_mps={rollout_root_velocity_mae[mask].mean().item():.3f}"
                + f" initial_phase_mean_s={initial_reference_phase[mask].mean().item():.3f}"
                + f" initial_source_time_min_s={initial_source_time[mask].min().item():.3f}"
                + f" initial_source_time_max_s={initial_source_time[mask].max().item():.3f}"
                + failure_stats,
                # new change
                flush=True,
            )
        # new change
        if args.verbose_steps and args.num_envs <= 16:
            for env_id in range(args.num_envs):
                print(
                    f"[SMOKE][env={env_id} motion={motion_names[env_id % len(motion_names)]}] "
                    f"terminated={terminated_total[env_id].item():.0f} "
                    f"truncated={truncated_total[env_id].item():.0f} "
                    f"min_height={minimum_root_height[env_id].item():.4f} "
                    f"max_tilt_deg={maximum_tilt_deg[env_id].item():.2f} "
                    f"q_ref_mae_deg={torch.rad2deg(q_error[env_id]).mean().item():.3f}",
                    # new change
                    flush=True,
                )
    finally:
        # new change
        # Kit may terminate before Python flushes block-buffered stdout when
        # the diagnostic is piped through tee or redirected to a file.
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        env.close()


if __name__ == "__main__":
    try:
        main()
    # new change
    # Isaac Kit's fast shutdown can terminate Python before an exception that
    # occurs during task imports/configuration is rendered.  Print and flush
    # it before closing the application so smoke failures remain actionable.
    except BaseException:
        import sys
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
