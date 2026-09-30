"""Collect randomized short-horizon outcomes for SupportRecoveryAwareTeacherV3."""

# new change
from __future__ import annotations

import argparse
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).absolute().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "softmimic_gym"))

from isaaclab.app import AppLauncher


# new change
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default=(
        "Isaac-G1-StandCoordination-SupportRecoveryAwareTeacherV3-"
        "100Pct-29DOF-v0"
    ),
)
parser.add_argument("--jit_policy", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--teacher_gain", type=float, default=0.10)
parser.add_argument("--teacher_limit_rad", type=float, default=0.006)
parser.add_argument("--pulse_steps", type=int, default=5)
parser.add_argument("--outcome_horizon", type=int, default=15)
parser.add_argument("--cycle_steps", type=int, default=40)
parser.add_argument("--minimum_episode_age", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# new change
SUPPORT_ACTION_IDS = (0, 1, 9, 10, 13, 14, 3, 4, 17, 18)


# new change
def _stability_cost(trend):
    """Dimensionless short-horizon cost from the eight explicit trend terms."""

    import torch

    com_distance = trend[:, 0].abs()
    com_outward = torch.relu(trend[:, 1])
    torso_tilt = trend[:, 2].abs()
    tilt_growth = torch.relu(trend[:, 3])
    root_speed = trend[:, 4].abs()
    foot_speed = trend[:, 5].abs()
    minimum_load = trend[:, 6]
    load_imbalance = trend[:, 7].abs()
    return (
        1.5 * com_distance
        + 0.5 * com_outward
        + 2.0 * torso_tilt
        + 0.5 * tilt_growth
        + root_speed
        + 0.5 * foot_speed
        + 2.0 * torch.relu(0.25 - minimum_load)
        + 0.5 * load_imbalance
    )


# new change
def main() -> None:
    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils import parse_env_cfg

    import softmimic_gym.tasks  # noqa: F401

    if args.num_envs % 16 != 0:
        raise ValueError("--num_envs must be divisible by 16 for balanced RCT assignment.")
    if not (0 < args.pulse_steps <= args.outcome_horizon < args.cycle_steps):
        raise ValueError(
            "Require 0 < pulse_steps <= outcome_horizon < cycle_steps."
        )
    if args.teacher_gain <= 0.0 or args.teacher_limit_rad <= 0.0:
        raise ValueError("Teacher gain and limit must be positive.")

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.recorders = None
    env = gym.make(args.task, cfg=env_cfg)
    try:
        observations, _ = env.reset()
        unwrapped = env.unwrapped
        device = unwrapped.device
        policy = torch.jit.load(args.jit_policy, map_location=device)
        policy.eval()
        command = unwrapped.command_manager.get_term("upper_body_joints")
        support_ids = torch.tensor(
            SUPPORT_ACTION_IDS, device=device, dtype=torch.long
        )
        environment_ids = torch.arange(args.num_envs, device=device)
        motion_ids = environment_ids % len(command.cfg.demo_recording_path)
        treatment_stratum = environment_ids // len(command.cfg.demo_recording_path)
        episode_age = torch.zeros(args.num_envs, device=device, dtype=torch.long)

        feature_batches = []
        candidate_batches = []
        treatment_batches = []
        outcome_batches = []
        failure_batches = []
        start_cost_batches = []
        motion_batches = []
        source_time_batches = []

        pending_features = None
        pending_candidate = None
        pending_treatment = None
        pending_valid = None
        pending_cost_sum = torch.zeros(args.num_envs, device=device)
        pending_failure = torch.zeros(
            args.num_envs, device=device, dtype=torch.bool
        )
        pending_start_cost = torch.zeros(args.num_envs, device=device)
        pending_source_time = torch.zeros(args.num_envs, device=device)

        for step_index in range(args.steps):
            cycle_index = step_index // args.cycle_steps
            cycle_step = step_index % args.cycle_steps
            support_features = observations["support_adapter"]
            if support_features.shape[-1] != 87:
                raise RuntimeError(
                    f"RecoveryAwareTeacherV3 expects 87D support state, got {support_features.shape[-1]}."
                )

            if cycle_step == 0:
                pending_features = support_features.detach().clone()
                support_error = pending_features[:, 20:30]
                pending_candidate = torch.clamp(
                    -args.teacher_gain * support_error,
                    min=-args.teacher_limit_rad,
                    max=args.teacher_limit_rad,
                )
                # Alternate assignment every cycle while preserving equal
                # treatment/control counts inside each of the eight motions.
                pending_treatment = (
                    (treatment_stratum + cycle_index) % 2 == 0
                )
                pending_valid = episode_age >= args.minimum_episode_age
                pending_cost_sum.zero_()
                pending_failure.zero_()
                pending_start_cost = _stability_cost(
                    pending_features[:, 79:87]
                )
                pending_source_time = (
                    command.motion_count * command.motion_dt
                ).detach().clone()

            with torch.inference_mode():
                action = policy(observations["policy"]).detach()
                if cycle_step < args.pulse_steps:
                    active = pending_treatment & pending_valid
                    action[:, support_ids] += (
                        pending_candidate
                        * active[:, None].to(pending_candidate.dtype)
                        / 0.30
                    )

            observations, _, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            episode_age += 1
            episode_age[done] = 0

            if cycle_step < args.outcome_horizon:
                pending_cost_sum += _stability_cost(
                    observations["support_adapter"][:, 79:87]
                )
                pending_failure |= terminated.bool()

            if cycle_step == args.outcome_horizon - 1:
                valid = pending_valid
                outcome = pending_cost_sum / float(args.outcome_horizon)
                # A reset makes post-failure state deceptively benign. Add a
                # dominant bounded failure cost before saving the outcome.
                outcome = outcome + 20.0 * pending_failure.to(outcome.dtype)
                feature_batches.append(pending_features[valid].cpu())
                candidate_batches.append(pending_candidate[valid].cpu())
                treatment_batches.append(pending_treatment[valid].cpu())
                outcome_batches.append(outcome[valid].cpu())
                failure_batches.append(pending_failure[valid].cpu())
                start_cost_batches.append(pending_start_cost[valid].cpu())
                motion_batches.append(motion_ids[valid].cpu())
                source_time_batches.append(pending_source_time[valid].cpu())

            if (step_index + 1) % 500 == 0:
                saved = sum(batch.shape[0] for batch in feature_batches)
                print(
                    f"[RECOVERY_V3] stepped={step_index + 1}/{args.steps} samples={saved}",
                    flush=True,
                )

        dataset = {
            "support_recovery_features": torch.cat(feature_batches, dim=0),
            "candidate_target_rad": torch.cat(candidate_batches, dim=0),
            "treatment": torch.cat(treatment_batches, dim=0),
            "future_stability_cost": torch.cat(outcome_batches, dim=0),
            "future_failure": torch.cat(failure_batches, dim=0),
            "start_stability_cost": torch.cat(start_cost_batches, dim=0),
            "motion_id": torch.cat(motion_batches, dim=0),
            "source_time_s": torch.cat(source_time_batches, dim=0),
            "support_action_ids": SUPPORT_ACTION_IDS,
            "teacher_gain": args.teacher_gain,
            "teacher_limit_rad": args.teacher_limit_rad,
            "pulse_steps": args.pulse_steps,
            "outcome_horizon": args.outcome_horizon,
            "cycle_steps": args.cycle_steps,
            "minimum_episode_age": args.minimum_episode_age,
            "seed": args.seed,
            "steps": args.steps,
            "num_envs": args.num_envs,
            "feature_order": (
                "support_v2[79],com_distance,com_outward_velocity,torso_tilt,"
                "tilt_growth,root_xy_speed,max_foot_xy_speed,min_support_load,"
                "support_load_imbalance"
            ),
        }
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, output)
        treatment = dataset["treatment"]
        print(
            {
                "output": str(output),
                "samples": int(treatment.numel()),
                "treatment_fraction": float(treatment.float().mean()),
                "control_cost_mean": float(
                    dataset["future_stability_cost"][~treatment].mean()
                ),
                "treatment_cost_mean": float(
                    dataset["future_stability_cost"][treatment].mean()
                ),
                "control_failure_fraction": float(
                    dataset["future_failure"][~treatment].float().mean()
                ),
                "treatment_failure_fraction": float(
                    dataset["future_failure"][treatment].float().mean()
                ),
            },
            flush=True,
        )
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
