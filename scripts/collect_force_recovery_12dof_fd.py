#!/usr/bin/env python3
"""Collect strict-pair +/- finite differences for force-recovery 12DOF control."""

# new change
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


# new change
PROJECT_ROOT = Path(__file__).absolute().parents[1]
TASK = (
    "Isaac-G1-CompliantStand-SmoothHalfCosine-MediumForce4to7N-"
    "SupervisedArmFeedback-RecoveryFD-100Pct-29DOF-v0"
)
DEFAULT_POLICY = PROJECT_ROOT / (
    "logs/rsl_rl/g1_stand_coordination_v1/"
    "2026-09-01_14-44-02_supervised_arm_feedback_dryrun/model_1375.jit"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "logs/stand_coordination_v1/compliant_stand_smooth_4to7_stress/"
    "recovery_12dof_fd_v1"
)
JOINT_NAMES = (
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
ACTION_IDS = (5, 8, 0, 1, 9, 10, 13, 14, 3, 4, 17, 18)
ACTION_SCALES_RAD = (0.20, 0.20) + (0.30,) * 10
EXTRA_OBS_GROUPS = (
    "left_arm_adapter",
    "right_arm_adapter",
    "online_compliance",
)


# new change
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jit_policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--event_seed", type=int, default=1001)
    parser.add_argument("--phase_offset", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--outcome_horizon_s", type=float, default=3.0)
    parser.add_argument("--pulse_steps", type=int, default=125)
    parser.add_argument("--amplitude_rad", type=float, default=0.002)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--probe_joint_names",
        nargs="*",
        default=(),
        help="Optional subset of the canonical 12 joints; empty runs all.",
    )
    parser.add_argument("--stop_after_candidates", type=int, default=None)
    return parser.parse_args()


# new change
def vector_text(values: list[float]) -> str:
    return ",".join(f"{value:.9g}" for value in values)


# new change
def run_logged(command: list[str], log_path: Path) -> None:
    print(json.dumps({"launch": command, "log": str(log_path)}), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            if any(
                token in line
                for token in ("'output':", "Traceback", "RuntimeError", "ValueError")
            ):
                print(line.rstrip(), flush=True)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"Child exited {code}; see {log_path}.")


# new change
def branch_command(
    args: argparse.Namespace,
    branch: str,
    output: Path,
    vector: list[float],
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts/collect_support_long_horizon_branch.py"),
        "--task",
        TASK,
        "--jit_policy",
        str(args.jit_policy),
        "--jit_extra_obs_groups",
        *EXTRA_OBS_GROUPS,
        "--branch",
        branch,
        "--output",
        str(output),
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--force_event_seed",
        str(args.event_seed),
        "--phase_offset",
        str(args.phase_offset),
        "--warmup_steps",
        str(args.warmup_steps),
        "--outcome_horizon_s",
        str(args.outcome_horizon_s),
        "--pulse_steps",
        str(args.pulse_steps),
        "--teacher_limit_rad",
        str(args.amplitude_rad),
        "--candidate_action_ids",
        *[str(value) for value in ACTION_IDS],
        "--candidate_action_scales_rad",
        *[str(value) for value in ACTION_SCALES_RAD],
        "--candidate_joint_names",
        *JOINT_NAMES,
        f"--candidate_vector_rad={vector_text(vector)}",
        "--require_force_context",
        "--device",
        args.device,
        "--headless",
    ]


# new change
def paired_complete(path: Path, vector: list[float]) -> bool:
    if not path.exists():
        return False
    data = torch.load(path, map_location="cpu", weights_only=False)
    target = data["candidate_target_rad"]
    expected = torch.tensor(vector, dtype=target.dtype).expand_as(target)
    # new change
    # Merge artifacts retain one replay-match flag per environment while the
    # console reports their count.  Audit the stored tensor directly instead
    # of coercing a multi-element tensor to one Python integer.
    force_replay_match = data.get("force_replay_match")
    force_candidate_match = data.get("force_candidate_match")
    force_replay_ok = bool(
        torch.is_tensor(force_replay_match)
        and force_replay_match.numel() == target.shape[0]
        and force_replay_match.bool().all()
    )
    force_candidate_ok = bool(
        torch.is_tensor(force_candidate_match)
        and force_candidate_match.numel() == target.shape[0]
        and force_candidate_match.bool().all()
    )
    return bool(
        data.get("strict_pairing", False)
        # new change
        # Force-recovery supervision is valid only when the sampled force,
        # applied wrench and IK acceptance were also merged into the audit.
        and data.get("force_conditioned_pairing", False)
        and not data.get("ppo_updates", True)
        and torch.equal(target, expected)
        and max(map(float, data["snapshot_deltas"].values()), default=0.0) == 0.0
        and max(
            map(float, data["baseline_replay_deltas"].values()), default=0.0
        )
        == 0.0
        and int(data.get("baseline_replay_failure_mismatch", -1)) == 0
        and force_replay_ok
        and force_candidate_ok
    )


# new change
def main() -> None:
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError("--num_envs must be positive.")
    if not 0.0 < args.amplitude_rad <= 0.006:
        raise ValueError("--amplitude_rad must be in (0, 0.006].")
    horizon_steps = int(round(args.outcome_horizon_s / 0.02))
    if not 1 <= args.pulse_steps <= horizon_steps:
        raise ValueError("--pulse_steps must be inside the outcome horizon.")
    unknown = sorted(set(args.probe_joint_names).difference(JOINT_NAMES))
    if unknown:
        raise ValueError(f"Unknown probe joints: {unknown}.")
    selected = set(args.probe_joint_names or JOINT_NAMES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    zero = [0.0] * len(JOINT_NAMES)
    baseline = args.output_dir / "baseline.pt"
    replay = args.output_dir / "baseline_replay.pt"
    if not baseline.exists():
        run_logged(
            branch_command(args, "baseline", baseline, zero),
            args.output_dir / "baseline.console.log",
        )
    if not replay.exists():
        run_logged(
            branch_command(args, "baseline", replay, zero),
            args.output_dir / "baseline_replay.console.log",
        )

    probes: dict[str, dict[str, object]] = {}
    completed = 0
    for joint_index, joint_name in enumerate(JOINT_NAMES):
        if joint_name not in selected:
            continue
        for sign_name, sign in (("positive", 1.0), ("negative", -1.0)):
            name = f"{joint_index:02d}_{joint_name}_{sign_name}"
            vector = [0.0] * len(JOINT_NAMES)
            vector[joint_index] = sign * args.amplitude_rad
            group = args.output_dir / name
            group.mkdir(parents=True, exist_ok=True)
            candidate = group / "candidate.pt"
            paired = group / "paired.pt"
            if not candidate.exists():
                run_logged(
                    branch_command(args, "candidate", candidate, vector),
                    group / "candidate.console.log",
                )
            if not paired_complete(paired, vector):
                run_logged(
                    [
                        sys.executable,
                        str(
                            PROJECT_ROOT
                            / "scripts/merge_support_long_horizon_branches.py"
                        ),
                        "--baseline",
                        str(baseline),
                        "--baseline_replay",
                        str(replay),
                        "--candidate",
                        str(candidate),
                        "--output",
                        str(paired),
                        # new change
                        "--require_force_context",
                    ],
                    group / "merge.console.log",
                )
            if not paired_complete(paired, vector):
                raise RuntimeError(f"Strict-pair audit failed: {paired}")
            data = torch.load(paired, map_location="cpu", weights_only=False)
            valid = data["valid"].bool()
            paired_delta = data["paired_score_delta"][valid]
            probes[name] = {
                "joint_index": joint_index,
                "joint_name": joint_name,
                "sign": sign_name,
                "paired": str(paired.resolve()),
                "valid": int(valid.sum()),
                "baseline_failures": int(
                    (data["baseline"]["failure"] & valid).sum()
                ),
                "candidate_failures": int(
                    (data["candidate"]["failure"] & valid).sum()
                ),
                "beneficial_fraction": float((paired_delta < 0.0).float().mean()),
                "paired_score_delta_mean": float(paired_delta.mean()),
            }
            completed += 1
            manifest = {
                "schema": "ForceRecovery12DOFFiniteDifferenceV1",
                "task": TASK,
                "jit_policy": str(args.jit_policy.resolve()),
                "source_checkpoint": "model_1375 supervised arm feedback",
                "num_envs": args.num_envs,
                "seed": args.seed,
                "event_seed": args.event_seed,
                "phase_offset": args.phase_offset,
                "warmup_steps": args.warmup_steps,
                "outcome_horizon_s": args.outcome_horizon_s,
                "pulse_steps": args.pulse_steps,
                "amplitude_rad": args.amplitude_rad,
                "joint_names": JOINT_NAMES,
                "action_ids": ACTION_IDS,
                "action_scales_rad": ACTION_SCALES_RAD,
                "same_physical_snapshot": True,
                "strict_baseline_replay": True,
                "strict_force_replay": True,
                "ppo_updates": False,
                "selected_joint_names": sorted(selected),
                "probes": probes,
                "completed": completed == 2 * len(selected),
            }
            (args.output_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps({"probe_complete": name, **probes[name]}), flush=True)
            if (
                args.stop_after_candidates is not None
                and completed >= args.stop_after_candidates
            ):
                return


if __name__ == "__main__":
    main()
