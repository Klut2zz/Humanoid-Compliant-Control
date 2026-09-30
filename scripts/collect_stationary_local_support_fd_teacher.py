#!/usr/bin/env python3
"""Collect one strict baseline and 20 local 10DOF support perturbations."""

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
TASK = "Isaac-G1-StandCoordination-StationaryFoundationV2-100Pct-29DOF-v0"
DEFAULT_POLICY = PROJECT_ROOT / (
    "logs/rsl_rl/g1_stand_coordination_v1/"
    "2026-08-20_18-57-37_stand_coordination_v1_full_v1/model_1375.jit"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "logs/stand_coordination_v1/stationary_foundation_v2/"
    "local_support_fd_teacher_v1"
)
# new change
DEFAULT_SNAPSHOT_ROLLOUT_POLICY = DEFAULT_OUTPUT / (
    "smooth2_raw_hardlimit/model_1375_local_fd_teacher.jit"
)
SUPPORT_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)


# new change
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jit_policy", type=Path, default=DEFAULT_POLICY)
    # new change
    parser.add_argument(
        "--snapshot_rollout_jit_policy",
        type=Path,
        default=None,
        help=(
            "Optional distilled controller used during warmup so labels are "
            "aggregated on its visited offset states."
        ),
    )
    parser.add_argument(
        "--snapshot_rollout_extra_obs_groups",
        nargs="*",
        default=("left_arm_adapter", "right_arm_adapter", "waist_adapter", "support_adapter"),
    )
    parser.add_argument("--snapshot_residual_hold_steps", type=int, default=125)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase_offset", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--outcome_horizon_s", type=float, default=2.5)
    parser.add_argument("--amplitude_rad", type=float, default=0.003)
    parser.add_argument(
        "--pulse_steps",
        type=int,
        default=125,
        help="Apply the local residual for this many 50-Hz policy steps.",
    )
    parser.add_argument("--device", default="cuda:0")
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
def branch_command(branch: str, output: Path, vector: list[float]) -> list[str]:
    args = ARGS
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/collect_support_long_horizon_branch.py"),
        "--task", TASK,
        "--jit_policy", str(args.jit_policy),
        "--branch", branch,
        "--output", str(output),
        "--num_envs", str(args.num_envs),
        "--seed", str(args.seed),
        "--phase_offset", str(args.phase_offset),
        "--warmup_steps", str(args.warmup_steps),
        "--outcome_horizon_s", str(args.outcome_horizon_s),
        "--pulse_steps", str(args.pulse_steps),
        "--teacher_limit_rad", "0.006",
        f"--candidate_vector_rad={vector_text(vector)}",
        "--device", args.device,
        "--headless",
    ]
    # new change
    if args.snapshot_rollout_jit_policy is not None:
        command.extend(
            [
                "--snapshot_rollout_jit_policy",
                str(args.snapshot_rollout_jit_policy),
                "--snapshot_residual_hold_steps",
                str(args.snapshot_residual_hold_steps),
                "--snapshot_rollout_extra_obs_groups",
                *args.snapshot_rollout_extra_obs_groups,
            ]
        )
    return command


# new change
def paired_complete(path: Path, vector: list[float]) -> bool:
    if not path.exists():
        return False
    data = torch.load(path, map_location="cpu", weights_only=False)
    target = data["candidate_target_rad"]
    expected = torch.tensor(vector, dtype=target.dtype).expand_as(target)
    return bool(
        data.get("strict_pairing", False)
        and not data.get("ppo_updates", True)
        and torch.equal(target, expected)
        and max(map(float, data["snapshot_deltas"].values()), default=0.0) == 0.0
        and max(
            map(float, data["baseline_replay_deltas"].values()), default=0.0
        ) == 0.0
        and int(data.get("baseline_replay_failure_mismatch", -1)) == 0
        and "support_feature_trajectory" in data["baseline"]
        and "support_feature_trajectory" in data["candidate"]
    )


# new change
def main() -> None:
    args = ARGS
    if args.num_envs % 8:
        raise ValueError("--num_envs must be divisible by the eight motions.")
    if not 0.0 < args.amplitude_rad <= 0.006:
        raise ValueError("--amplitude_rad must be in (0, 0.006].")
    horizon_steps = int(round(args.outcome_horizon_s / 0.02))
    if not 1 <= args.pulse_steps <= horizon_steps:
        raise ValueError("--pulse_steps must be within the outcome horizon.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zero = [0.0] * 10
    baseline = args.output_dir / "baseline.pt"
    replay = args.output_dir / "baseline_replay.pt"
    if not baseline.exists():
        run_logged(
            branch_command("baseline", baseline, zero),
            args.output_dir / "baseline.console.log",
        )
    if not replay.exists():
        run_logged(
            branch_command("baseline", replay, zero),
            args.output_dir / "baseline_replay.console.log",
        )

    probes: dict[str, dict[str, object]] = {}
    completed = 0
    for joint_index, joint_name in enumerate(SUPPORT_JOINT_NAMES):
        for sign_name, sign in (("positive", 1.0), ("negative", -1.0)):
            name = f"{joint_index:02d}_{joint_name}_{sign_name}"
            vector = [0.0] * 10
            vector[joint_index] = sign * args.amplitude_rad
            group = args.output_dir / name
            group.mkdir(parents=True, exist_ok=True)
            candidate = group / "candidate.pt"
            paired = group / "paired.pt"
            if not candidate.exists():
                run_logged(
                    branch_command("candidate", candidate, vector),
                    group / "candidate.console.log",
                )
            if not paired_complete(paired, vector):
                run_logged(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts/merge_support_long_horizon_branches.py"),
                        "--baseline", str(baseline),
                        "--baseline_replay", str(replay),
                        "--candidate", str(candidate),
                        "--output", str(paired),
                    ],
                    group / "merge.console.log",
                )
            if not paired_complete(paired, vector):
                raise RuntimeError(f"Strict-pair audit failed: {paired}")
            data = torch.load(paired, map_location="cpu", weights_only=False)
            valid = data["valid"].bool()
            probes[name] = {
                "joint_index": joint_index,
                "joint_name": joint_name,
                "sign": sign_name,
                "vector_rad": vector,
                "paired": str(paired.resolve()),
                "valid": int(valid.sum()),
                "baseline_failures": int((data["baseline"]["failure"] & valid).sum()),
                "candidate_failures": int((data["candidate"]["failure"] & valid).sum()),
            }
            completed += 1
            manifest = {
                "schema": "StationaryLocalSupportFiniteDifferenceTeacherV1",
                "task": TASK,
                "jit_policy": str(args.jit_policy.resolve()),
                "source_policy": "frozen model_1375",
                "num_envs": args.num_envs,
                "seed": args.seed,
                "phase_offset": args.phase_offset,
                "warmup_steps": args.warmup_steps,
                # new change
                "snapshot_rollout_jit_policy": (
                    None
                    if args.snapshot_rollout_jit_policy is None
                    else str(args.snapshot_rollout_jit_policy.resolve())
                ),
                "snapshot_residual_hold_steps": args.snapshot_residual_hold_steps,
                "temporal_context_dim": 98,
                "outcome_horizon_s": args.outcome_horizon_s,
                "pulse_steps": args.pulse_steps,
                "amplitude_rad": args.amplitude_rad,
                "support_joint_names": SUPPORT_JOINT_NAMES,
                "same_physical_snapshot": True,
                "strict_baseline_replay": True,
                "ppo_updates": False,
                "probes": probes,
                "completed": completed == 20,
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


# new change
ARGS = parse_args()


if __name__ == "__main__":
    main()
