"""Diagnose frozen compliance control across force-event start/hold/stop stages."""

# new change
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys


# new change
_PROJECT_ROOT = pathlib.Path(__file__).absolute().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "softmimic_gym"))

from isaaclab.app import AppLauncher


# new change
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--jit_policy", required=True)
parser.add_argument("--baseline_jit", required=True)
parser.add_argument(
    "--jit_extra_obs_groups",
    nargs="*",
    default=("left_arm_adapter", "right_arm_adapter", "online_compliance"),
)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--seed", type=int, default=0)
# new change
parser.add_argument(
    "--event_seed",
    type=int,
    default=None,
    help="Independent command/force-event RNG seed; default preserves task value 42.",
)
parser.add_argument("--support_contact_threshold_n", type=float, default=20.0)
parser.add_argument("--output_dir", required=True)
# new change
parser.add_argument(
    "--teacher_dataset_pt",
    default=None,
    help="Optional path for active-event 71D features and 14D arm targets.",
)
parser.add_argument("--teacher_sample_stride", type=int, default=5)
parser.add_argument("--teacher_max_samples", type=int, default=200000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# new change
_STAGE_NAMES = (
    "waiting_early_0_25s",
    "waiting_late",
    "ramp_up_0_25",
    "ramp_up_25_100",
    "hold",
    "ramp_down_100_25",
    "ramp_down_25_0",
    "rejected_event",
)


# new change
def _stage_indices(command, previous_stage, waiting_age_steps):
    """Resolve event stages before the current action/physics step."""

    import torch

    event_time = command._event_time
    ramp = torch.clamp(command._ramp_time, min=1.0e-6)
    hold = command._hold_time
    accepted = command.ik_feasible_scale > 0.0
    waiting = event_time < 0.0
    entered_waiting = waiting & (previous_stage >= 2) & (previous_stage <= 6)
    waiting_age_steps = torch.where(
        entered_waiting,
        torch.zeros_like(waiting_age_steps),
        torch.where(waiting, waiting_age_steps + 1, torch.zeros_like(waiting_age_steps)),
    )

    stage = torch.full_like(previous_stage, 7)
    early_wait_steps = max(1, int(round(0.25 / float(command._step_dt))))
    stage[waiting & (waiting_age_steps <= early_wait_steps)] = 0
    stage[waiting & (waiting_age_steps > early_wait_steps)] = 1

    valid = (~waiting) & accepted
    up = valid & (event_time < ramp)
    up_fraction = event_time / ramp
    stage[up & (up_fraction < 0.25)] = 2
    stage[up & (up_fraction >= 0.25)] = 3

    hold_mask = valid & (event_time >= ramp) & (event_time < ramp + hold)
    stage[hold_mask] = 4

    down = valid & (event_time >= ramp + hold)
    down_fraction = (event_time - ramp - hold) / ramp
    stage[down & (down_fraction < 0.75)] = 5
    stage[down & (down_fraction >= 0.75)] = 6
    return stage, waiting_age_steps


# new change
def main() -> None:
    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils import parse_env_cfg

    import softmimic_gym.tasks  # noqa: F401

    if args.num_envs <= 0 or args.steps <= 0:
        raise ValueError("--num_envs and --steps must be positive")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    # new change
    if args.event_seed is not None:
        env_cfg.commands.upper_body_joints.command_random_seed = args.event_seed
    env_cfg.recorders = None
    env = gym.make(args.task, cfg=env_cfg)

    try:
        observations, _ = env.reset()
        unwrapped = env.unwrapped
        device = unwrapped.device
        robot = unwrapped.scene["robot"]
        command = unwrapped.command_manager.get_term("upper_body_joints")
        policy = torch.jit.load(args.jit_policy, map_location=device)
        baseline_policy = torch.jit.load(args.baseline_jit, map_location=device)
        policy.eval()
        baseline_policy.eval()

        foot_ids_list, foot_names = robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        if foot_names != ["left_ankle_roll_link", "right_ankle_roll_link"]:
            raise RuntimeError("Could not resolve G1 support feet")
        foot_ids = torch.as_tensor(foot_ids_list, device=device, dtype=torch.long)
        contact_sensor = unwrapped.scene.sensors["contact_forces"]
        support_contact_ids_list, support_contact_names = contact_sensor.find_bodies(
            ["left_knee_link", "right_knee_link"], preserve_order=True
        )
        if support_contact_names != ["left_knee_link", "right_knee_link"]:
            raise RuntimeError("Could not resolve G1 support-force channels")
        support_contact_ids = torch.as_tensor(
            support_contact_ids_list, device=device, dtype=torch.long
        )

        dtype = torch.float64
        stage_count = len(_STAGE_NAMES)
        sample_count = torch.zeros(stage_count, device=device, dtype=dtype)
        accepted_count = torch.zeros_like(sample_count)
        active_count = torch.zeros_like(sample_count)
        force_sum = torch.zeros_like(sample_count)
        force_max = torch.zeros_like(sample_count)
        tilt_sum = torch.zeros_like(sample_count)
        tilt_max = torch.zeros_like(sample_count)
        root_speed_sum = torch.zeros_like(sample_count)
        root_speed_max = torch.zeros_like(sample_count)
        com_sum = torch.zeros_like(sample_count)
        com_max = torch.zeros_like(sample_count)
        wrist_error_sum = torch.zeros_like(sample_count)
        wrist_error_max = torch.zeros_like(sample_count)
        keypoint_error_sum = torch.zeros_like(sample_count)
        # new change
        # The structured 37-D arm observations are laid out as
        # q(7), q_ref(7), q-q_ref(7), dq(7), dq_ref(7), phase(2).
        # Keep the two sides separate so a spatial wrist improvement cannot
        # hide a worse bilateral joint-angle tracking error.
        left_arm_q_error_sum = torch.zeros_like(sample_count)
        right_arm_q_error_sum = torch.zeros_like(sample_count)
        arm_target_delta_sum = torch.zeros_like(sample_count)
        arm_target_delta_max = torch.zeros_like(sample_count)
        # new change
        # Count physical per-joint target corrections near the +/-20 mrad
        # analytic-feedback limit.  The existing value is an env-wise mean and
        # cannot reveal whether a small subset of joints is saturating.
        arm_target_joint_delta_max = torch.zeros_like(sample_count)
        arm_target_saturated_count = torch.zeros_like(sample_count)
        failure_resets = torch.zeros_like(sample_count)
        bad_orientation_resets = torch.zeros_like(sample_count)
        base_contact_resets = torch.zeros_like(sample_count)
        contact_sample_count = torch.zeros(
            stage_count, 2, device=device, dtype=dtype
        )
        contact_slip_sum = torch.zeros_like(contact_sample_count)
        contact_slip_max = torch.zeros_like(contact_sample_count)
        anchor_sum = torch.zeros_like(contact_sample_count)
        anchor_max = torch.zeros_like(contact_sample_count)
        failure_by_body = torch.zeros(
            stage_count,
            len(command.cfg.force_body_names),
            device=device,
            dtype=dtype,
        )

        masses = robot.data.default_mass.to(device=device, dtype=dtype)
        total_mass = masses.sum(dim=1, keepdim=True)
        arm_action_ids = torch.as_tensor(
            [11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
            device=device,
            dtype=torch.long,
        )
        # new change
        # Head order is left 7DOF then right 7DOF, unlike the interleaved list
        # above which is retained for the existing aggregate diagnostics.
        teacher_arm_action_ids = torch.as_tensor(
            [11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
            device=device,
            dtype=torch.long,
        )
        teacher_feature_batches = []
        teacher_target_batches = []
        teacher_sample_count = 0
        previous_contact = torch.zeros(
            args.num_envs, 2, device=device, dtype=torch.bool
        )
        contact_anchor_xy = robot.data.body_com_pos_w.index_select(
            1, foot_ids
        )[:, :, :2].clone()
        previous_stage = torch.full(
            (args.num_envs,), -1, device=device, dtype=torch.long
        )
        waiting_age_steps = torch.zeros_like(previous_stage)
        first_failure_step = torch.full_like(previous_stage, -1)
        first_failure_stage = torch.full_like(previous_stage, -1)
        first_failure_force = torch.zeros(
            args.num_envs, device=device, dtype=dtype
        )
        first_failure_body = torch.full_like(previous_stage, -1)
        first_failure_bad = torch.zeros(
            args.num_envs, device=device, dtype=torch.bool
        )
        first_failure_base = torch.zeros_like(first_failure_bad)

        for step_index in range(args.steps):
            stage, waiting_age_steps = _stage_indices(
                command, previous_stage, waiting_age_steps
            )
            previous_stage = stage.clone()
            force_magnitude = torch.linalg.vector_norm(command.force_world, dim=-1)
            accepted = command.ik_feasible_scale > 0.0
            active = command.active_force_mask
            # new change
            # ManagerBasedRLEnv auto-resets terminal rows inside env.step().
            # Snapshot the event hand before stepping so failures are not
            # mislabeled with the newly sampled post-reset event.
            body_slot_before_step = command._sampled_body_slot.clone()

            policy_input = torch.cat(
                [observations["policy"]]
                + [observations[name] for name in args.jit_extra_obs_groups],
                dim=-1,
            )
            with torch.inference_mode():
                action = policy(policy_input)
                baseline_action = baseline_policy(observations["policy"])
            arm_target_joint_delta = (
                0.15
                * (action - baseline_action)
                .index_select(1, arm_action_ids)
                .abs()
            ).to(dtype)
            arm_target_delta = arm_target_joint_delta.mean(dim=1)
            # new change
            # Distil only accepted, physically synchronized force-event rows.
            # Targets are the exact analytic action increment observed in this
            # rollout, after clipping and hard gating.
            if (
                args.teacher_dataset_pt
                and step_index % max(args.teacher_sample_stride, 1) == 0
                and teacher_sample_count < args.teacher_max_samples
            ):
                teacher_mask = active & accepted
                teacher_ids = torch.where(teacher_mask)[0]
                remaining = args.teacher_max_samples - teacher_sample_count
                teacher_ids = teacher_ids[:remaining]
                if teacher_ids.numel() > 0:
                    teacher_feature_batches.append(
                        observations["online_compliance"]
                        .index_select(0, teacher_ids)
                        .detach()
                        .cpu()
                    )
                    teacher_target_batches.append(
                        (action - baseline_action)
                        .index_select(0, teacher_ids)
                        .index_select(1, teacher_arm_action_ids)
                        .detach()
                        .cpu()
                    )
                    teacher_sample_count += int(teacher_ids.numel())

            upright_cosine = torch.clamp(
                -robot.data.projected_gravity_b[:, 2], min=-1.0, max=1.0
            )
            tilt = torch.acos(upright_cosine).to(dtype)
            root_speed = torch.linalg.vector_norm(
                robot.data.root_lin_vel_w[:, :2], dim=-1
            ).to(dtype)
            body_com = robot.data.body_com_pos_w.to(dtype)
            whole_body_com = (body_com * masses.unsqueeze(-1)).sum(dim=1) / total_mass
            foot_pos = body_com.index_select(1, foot_ids)
            support_xy = foot_pos[:, :, :2].mean(dim=1)
            com_distance = torch.linalg.vector_norm(
                whole_body_com[:, :2] - support_xy, dim=-1
            )
            foot_speed = torch.linalg.vector_norm(
                robot.data.body_com_lin_vel_w.index_select(1, foot_ids)[:, :, :2],
                dim=-1,
            ).to(dtype)
            support_force = (
                contact_sensor.data.net_forces_w_history.index_select(
                    2, support_contact_ids
                )[..., 2]
                .abs()
                .max(dim=1)
                .values.to(dtype)
            )
            contact = support_force >= args.support_contact_threshold_n
            foot_xy = foot_pos[:, :, :2]
            new_contact = contact & ~previous_contact
            contact_anchor_xy = torch.where(
                new_contact.unsqueeze(-1), foot_xy, contact_anchor_xy
            )
            anchor_displacement = torch.linalg.vector_norm(
                foot_xy - contact_anchor_xy, dim=-1
            ).to(dtype)
            wrist_error = command.actual_active_link_error_m.to(dtype)
            keypoint_error = command.actual_keypoint_mean_error_m.to(dtype)
            # new change
            left_arm_q_error = (
                observations["left_arm_adapter"][:, 14:21]
                .abs()
                .mean(dim=1)
                .to(dtype)
            )
            right_arm_q_error = (
                observations["right_arm_adapter"][:, 14:21]
                .abs()
                .mean(dim=1)
                .to(dtype)
            )

            for stage_id in range(stage_count):
                mask = stage == stage_id
                if not torch.any(mask):
                    continue
                count = mask.sum().to(dtype)
                sample_count[stage_id] += count
                accepted_count[stage_id] += (accepted & mask).sum().to(dtype)
                active_count[stage_id] += (active & mask).sum().to(dtype)
                for values, sums, maxima in (
                    (force_magnitude.to(dtype), force_sum, force_max),
                    (tilt, tilt_sum, tilt_max),
                    (root_speed, root_speed_sum, root_speed_max),
                    (com_distance, com_sum, com_max),
                    (wrist_error, wrist_error_sum, wrist_error_max),
                    (arm_target_delta, arm_target_delta_sum, arm_target_delta_max),
                ):
                    sums[stage_id] += values[mask].sum()
                    maxima[stage_id] = torch.maximum(
                        maxima[stage_id], values[mask].max()
                    )
                keypoint_error_sum[stage_id] += keypoint_error[mask].sum()
                left_arm_q_error_sum[stage_id] += left_arm_q_error[mask].sum()
                right_arm_q_error_sum[stage_id] += right_arm_q_error[mask].sum()
                # new change
                selected_joint_delta = arm_target_joint_delta[mask]
                arm_target_joint_delta_max[stage_id] = torch.maximum(
                    arm_target_joint_delta_max[stage_id],
                    selected_joint_delta.max(),
                )
                arm_target_saturated_count[stage_id] += (
                    selected_joint_delta >= 0.0199
                ).sum().to(dtype)
                for side in range(2):
                    contact_mask = mask & contact[:, side]
                    if not torch.any(contact_mask):
                        continue
                    contact_count = contact_mask.sum().to(dtype)
                    contact_sample_count[stage_id, side] += contact_count
                    contact_slip_sum[stage_id, side] += foot_speed[
                        contact_mask, side
                    ].sum()
                    contact_slip_max[stage_id, side] = torch.maximum(
                        contact_slip_max[stage_id, side],
                        foot_speed[contact_mask, side].max(),
                    )
                    anchor_sum[stage_id, side] += anchor_displacement[
                        contact_mask, side
                    ].sum()
                    anchor_max[stage_id, side] = torch.maximum(
                        anchor_max[stage_id, side],
                        anchor_displacement[contact_mask, side].max(),
                    )

            observations, _, terminated, truncated, _ = env.step(action)
            bad_orientation = unwrapped.termination_manager.get_term(
                "bad_orientation"
            ).bool()
            base_contact = unwrapped.termination_manager.get_term(
                "base_contact"
            ).bool()
            failure = terminated.bool()
            for stage_id in range(stage_count):
                failure_mask = failure & (stage == stage_id)
                if not torch.any(failure_mask):
                    continue
                failure_resets[stage_id] += failure_mask.sum().to(dtype)
                bad_orientation_resets[stage_id] += (
                    failure_mask & bad_orientation
                ).sum().to(dtype)
                base_contact_resets[stage_id] += (
                    failure_mask & base_contact
                ).sum().to(dtype)
                slots = body_slot_before_step[failure_mask]
                failure_by_body[stage_id].scatter_add_(
                    0, slots, torch.ones_like(slots, dtype=dtype)
                )

            first = failure & (first_failure_step < 0)
            if torch.any(first):
                first_failure_step[first] = step_index + 1
                first_failure_stage[first] = stage[first]
                first_failure_force[first] = force_magnitude[first].to(dtype)
                first_failure_body[first] = body_slot_before_step[first]
                first_failure_bad[first] = bad_orientation[first]
                first_failure_base[first] = base_contact[first]

            done = terminated.bool() | truncated.bool()
            previous_contact = contact.clone()
            previous_contact[done] = False
            if (step_index + 1) % 250 == 0:
                print(
                    f"[FORCE_PHASE_DIAG] stepped={step_index + 1}/{args.steps}",
                    flush=True,
                )

        def safe_mean(total, count):
            return total / torch.clamp(count, min=1.0)

        stage_rows = []
        for stage_id, stage_name in enumerate(_STAGE_NAMES):
            row = {
                "stage": stage_name,
                "samples": int(sample_count[stage_id].item()),
                "sample_fraction": float(
                    sample_count[stage_id].item()
                    / max(args.num_envs * args.steps, 1)
                ),
                "ik_accepted_fraction": float(
                    safe_mean(accepted_count, sample_count)[stage_id].item()
                ),
                "force_active_fraction": float(
                    safe_mean(active_count, sample_count)[stage_id].item()
                ),
                "force_mean_n": float(
                    safe_mean(force_sum, sample_count)[stage_id].item()
                ),
                "force_max_n": float(force_max[stage_id].item()),
                "failure_resets": int(failure_resets[stage_id].item()),
                "first_failed_envs": int(
                    (first_failure_stage == stage_id).sum().item()
                ),
                "failure_resets_per_million_samples": float(
                    1.0e6
                    * failure_resets[stage_id].item()
                    / max(sample_count[stage_id].item(), 1.0)
                ),
                "bad_orientation_resets": int(
                    bad_orientation_resets[stage_id].item()
                ),
                "base_contact_resets": int(base_contact_resets[stage_id].item()),
                "root_tilt_mean_deg": math.degrees(
                    float(safe_mean(tilt_sum, sample_count)[stage_id].item())
                ),
                "root_tilt_max_deg": math.degrees(float(tilt_max[stage_id].item())),
                "root_xy_speed_mean_mps": float(
                    safe_mean(root_speed_sum, sample_count)[stage_id].item()
                ),
                "root_xy_speed_max_mps": float(root_speed_max[stage_id].item()),
                "com_support_mean_m": float(
                    safe_mean(com_sum, sample_count)[stage_id].item()
                ),
                "com_support_max_m": float(com_max[stage_id].item()),
                "active_wrist_error_mean_m": float(
                    safe_mean(wrist_error_sum, sample_count)[stage_id].item()
                ),
                "active_wrist_error_max_m": float(
                    wrist_error_max[stage_id].item()
                ),
                "all_keypoint_error_mean_m": float(
                    safe_mean(keypoint_error_sum, sample_count)[stage_id].item()
                ),
                # new change
                "left_arm_q_error_mean_rad": float(
                    safe_mean(left_arm_q_error_sum, sample_count)[stage_id].item()
                ),
                "right_arm_q_error_mean_rad": float(
                    safe_mean(right_arm_q_error_sum, sample_count)[stage_id].item()
                ),
                "arm_target_delta_mean_rad": float(
                    safe_mean(arm_target_delta_sum, sample_count)[stage_id].item()
                ),
                "arm_target_delta_max_rad": float(
                    arm_target_delta_max[stage_id].item()
                ),
                # new change
                "arm_target_joint_delta_max_rad": float(
                    arm_target_joint_delta_max[stage_id].item()
                ),
                "arm_target_saturation_fraction": float(
                    arm_target_saturated_count[stage_id].item()
                    / max(14.0 * sample_count[stage_id].item(), 1.0)
                ),
                "left_contact_fraction": float(
                    safe_mean(contact_sample_count[:, 0], sample_count)[stage_id].item()
                ),
                "right_contact_fraction": float(
                    safe_mean(contact_sample_count[:, 1], sample_count)[stage_id].item()
                ),
                "left_contact_slip_mean_mps": float(
                    safe_mean(
                        contact_slip_sum[:, 0], contact_sample_count[:, 0]
                    )[stage_id].item()
                ),
                "right_contact_slip_mean_mps": float(
                    safe_mean(
                        contact_slip_sum[:, 1], contact_sample_count[:, 1]
                    )[stage_id].item()
                ),
                "left_anchor_mean_m": float(
                    safe_mean(anchor_sum[:, 0], contact_sample_count[:, 0])[
                        stage_id
                    ].item()
                ),
                "right_anchor_mean_m": float(
                    safe_mean(anchor_sum[:, 1], contact_sample_count[:, 1])[
                        stage_id
                    ].item()
                ),
                "left_anchor_max_m": float(anchor_max[stage_id, 0].item()),
                "right_anchor_max_m": float(anchor_max[stage_id, 1].item()),
            }
            for body_id, body_name in enumerate(command.cfg.force_body_names):
                row[f"failure_resets_{body_name}"] = int(
                    failure_by_body[stage_id, body_id].item()
                )
            stage_rows.append(row)

        failed_envs = int((first_failure_step >= 0).sum().item())
        online_summary = command.diagnostic_summary()
        summary = {
            "task": args.task,
            "jit_policy": str(pathlib.Path(args.jit_policy).resolve()),
            "baseline_jit": str(pathlib.Path(args.baseline_jit).resolve()),
            "seed": args.seed,
            # new change
            "event_seed": int(command.cfg.command_random_seed),
            "num_envs": args.num_envs,
            "steps": args.steps,
            "force_range_n": list(command.cfg.force_range_n),
            "wait_time_range_s": list(command.cfg.wait_time_range_s),
            "ramp_time_range_s": list(command.cfg.ramp_time_range_s),
            "hold_time_range_s": list(command.cfg.hold_time_range_s),
            "failed_envs": failed_envs,
            "success_rate": 1.0 - failed_envs / args.num_envs,
            "failure_resets": int(failure_resets.sum().item()),
            "online_reference": online_summary,
            "stages": stage_rows,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        with (output_dir / "stage_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(stage_rows[0]))
            writer.writeheader()
            writer.writerows(stage_rows)

        failure_rows = []
        for env_id in torch.where(first_failure_step >= 0)[0].tolist():
            body_id = int(first_failure_body[env_id].item())
            failure_rows.append(
                {
                    "env_id": env_id,
                    "step": int(first_failure_step[env_id].item()),
                    "stage": _STAGE_NAMES[int(first_failure_stage[env_id].item())],
                    "force_n": float(first_failure_force[env_id].item()),
                    "force_body": command.cfg.force_body_names[body_id],
                    "bad_orientation": bool(first_failure_bad[env_id].item()),
                    "base_contact": bool(first_failure_base[env_id].item()),
                }
            )
        with (output_dir / "failure_events.csv").open(
            "w", newline="", encoding="utf-8"
        ) as output_file:
            fieldnames = (
                "env_id",
                "step",
                "stage",
                "force_n",
                "force_body",
                "bad_orientation",
                "base_contact",
            )
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(failure_rows)

        report_lines = [
            "# Force-event phase diagnosis",
            "",
            f"- Task: `{args.task}`",
            f"- Population: {args.num_envs} environments x {args.steps} steps, seed {args.seed}",
            # new change
            f"- Force-event seed: {int(command.cfg.command_random_seed)}",
            f"- Force range: {tuple(command.cfg.force_range_n)} N",
            f"- Failed environments: {failed_envs}/{args.num_envs} (success {100.0 * summary['success_rate']:.2f}%)",
            f"- Failure resets: {summary['failure_resets']}",
            f"- IK acceptance: {100.0 * online_summary['acceptance_rate']:.2f}%",
            "",
            "| Stage | samples | failures | first failed | force N | tilt deg | COM mm | root mm/s | wrist mm | arm q L/R deg | slip L/R mm/s | target residual mrad | saturation % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in stage_rows:
            report_lines.append(
                "| {stage} | {samples} | {failure_resets} | {first_failed_envs} | "
                "{force_mean_n:.2f} | {root_tilt_mean_deg:.2f} | "
                "{com:.2f} | {root:.1f} | {wrist:.2f} | {q_left:.3f}/{q_right:.3f} | {left:.1f}/{right:.1f} | "
                "{residual:.2f} | {saturation:.2f} |".format(
                    **row,
                    com=1000.0 * row["com_support_mean_m"],
                    root=1000.0 * row["root_xy_speed_mean_mps"],
                    wrist=1000.0 * row["active_wrist_error_mean_m"],
                    q_left=math.degrees(row["left_arm_q_error_mean_rad"]),
                    q_right=math.degrees(row["right_arm_q_error_mean_rad"]),
                    left=1000.0 * row["left_contact_slip_mean_mps"],
                    right=1000.0 * row["right_contact_slip_mean_mps"],
                    residual=1000.0 * row["arm_target_delta_mean_rad"],
                    saturation=100.0 * row["arm_target_saturation_fraction"],
                )
            )
        report_lines.extend(
            [
                "",
                "`waiting_early_0_25s` is the immediate post-release/initial recovery window. "
                "Ramp-up and ramp-down are split so hard-gate onset and shutoff are visible.",
                "",
            ]
        )
        (output_dir / "report.md").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )
        # new change
        if args.teacher_dataset_pt:
            if not teacher_feature_batches:
                raise RuntimeError("No accepted active-event teacher samples were collected.")
            teacher_path = pathlib.Path(args.teacher_dataset_pt)
            teacher_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "online_compliance": torch.cat(teacher_feature_batches, dim=0),
                    "arm_target_actor": torch.cat(teacher_target_batches, dim=0),
                    "feature_layout": (
                        "q_minus_reference17,dq17,reference_dq17,context20"
                    ),
                    "target_layout": "left_arm7,right_arm7",
                    "alpha": 0.50,
                    "beta_s": 0.01,
                    "target_limit_rad": 0.020,
                    "seed": args.seed,
                    # new change
                    "event_seed": int(command.cfg.command_random_seed),
                    "num_envs": args.num_envs,
                    "steps": args.steps,
                    "sample_stride": args.teacher_sample_stride,
                    "task": args.task,
                },
                teacher_path,
            )
            print(
                f"[FORCE_PHASE_DIAG] teacher_dataset={teacher_path} "
                f"samples={teacher_sample_count}",
                flush=True,
            )
        print(
            f"[FORCE_PHASE_DIAG] complete output_dir={output_dir} "
            f"failed_envs={failed_envs}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
