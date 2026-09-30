"""Collect an audited low-force pilot for the persistent closed-loop support candidate."""

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
ROOT = Path(__file__).absolute().parents[1]
DEFAULT_OUTPUT = ROOT / (
    "logs/stand_coordination_v1/stationary_support_control_v2/"
    "continuous_candidate_paired_pilot"
)
DEFAULT_POLICY = ROOT / (
    "logs/rsl_rl/g1_stand_coordination_v1/"
    "2026-08-20_18-57-37_stand_coordination_v1_full_v1/model_1375.jit"
)
TASK = (
    "Isaac-G1-StandCoordination-FrozenNeighborGate-"
    "LowForce4to7N-100Pct-29DOF-v0"
)
MOTION_NAMES = (
    "neutral",
    "bilateral_sagittal",
    "bilateral_lateral",
    "unilateral_counterbalance",
    "axial_twist",
    "roll_coordination",
    "pitch_coordination",
    "combined_coordination",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--jit_policy", type=Path, default=DEFAULT_POLICY)
parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--event_seed", type=int, default=5051)
parser.add_argument("--warmup_steps", type=int, nargs="+", default=(25, 75, 125))
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()


# new change
def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"launch": command, "log": str(log_path)}), flush=True)
    if args.dry_run:
        return
    with log_path.open("w") as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode:
        raise RuntimeError(f"Child failed ({result.returncode}); see {log_path}.")


# new change
def _branch_command(branch: str, output: Path, warmup: int) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/collect_support_long_horizon_branch.py"),
        "--task",
        TASK,
        "--jit_policy",
        str(args.jit_policy),
        "--branch",
        branch,
        "--output",
        str(output),
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(args.event_seed),
        "--force_event_seed",
        str(args.event_seed),
        "--phase_offset",
        "0.0",
        "--warmup_steps",
        str(warmup),
        "--outcome_horizon_s",
        "5.0",
        "--teacher_gain",
        "0.10",
        "--teacher_limit_rad",
        "0.006",
        "--pulse_steps",
        "250",
        "--closed_loop_candidate",
        "--require_force_context",
        "--device",
        args.device,
        "--headless",
    ]


# new change
def _branch_complete(path: Path, branch: str, warmup: int) -> bool:
    if not path.is_file():
        return False
    data = torch.load(path, map_location="cpu", weights_only=False)
    metadata = data["snapshot_fingerprint_metadata"]
    return bool(
        data["branch"] == branch
        and data.get("candidate_mode") == "q_error_closed_loop"
        and data.get("closed_loop_candidate", False)
        and int(metadata["force_event_seed"]) == args.event_seed
        and int(metadata["warmup_steps"]) == warmup
        and int(metadata["num_envs"]) == args.num_envs
        and int(data["pulse_steps"]) == 250
        and float(data["outcome_horizon_s"]) == 5.0
    )


# new change
def _paired_summary(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    valid = data["valid"].bool()
    delta = data["paired_failure_delta"]
    score = data["paired_score_delta"]
    baseline = data["baseline"]
    candidate = data["candidate"]
    correction = candidate["candidate_correction_rad_trajectory"]
    output = {
        "artifact": str(path.resolve()),
        "valid": int(valid.sum()),
        "baseline_failures": int((baseline["failure"] & valid).sum()),
        "candidate_failures": int((candidate["failure"] & valid).sum()),
        "prevented_failures": int(((delta < 0) & valid).sum()),
        "caused_failures": int(((delta > 0) & valid).sum()),
        "beneficial_fraction": float(
            (data["candidate_beneficial"] & valid).sum() / valid.sum()
        ),
        "paired_score_delta_mean": float(score[valid].mean()),
        "paired_score_delta_median": float(score[valid].median()),
        "mean_correction_mrad": float(correction[valid].abs().mean() * 1000.0),
        "correction_saturation_fraction": float(
            (correction[valid].abs() >= 0.005999).float().mean()
        ),
        "caused_failure_motion_names": [
            MOTION_NAMES[index]
            for index in data["motion_id"][(delta > 0) & valid].tolist()
        ],
        "prevented_failure_motion_names": [
            MOTION_NAMES[index]
            for index in data["motion_id"][(delta < 0) & valid].tolist()
        ],
    }
    for steps, label in ((25, "0p5s"), (50, "1s"), (100, "2s"), (250, "5s")):
        for key, short in (
            ("com_support_xy_distance_m_trajectory", "com"),
            ("torso_tilt_rad_trajectory", "tilt"),
        ):
            improvement = (
                baseline[key][:, :steps].mean(1)
                - candidate[key][:, :steps].mean(1)
            )
            output[f"{short}_improvement_{label}_mean"] = float(
                improvement[valid].mean()
            )
    return output


# new change
def main() -> None:
    if not args.jit_policy.is_file():
        raise FileNotFoundError(args.jit_policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    for warmup in args.warmup_steps:
        group = args.output_dir / f"event{args.event_seed}_w{warmup}"
        group.mkdir(parents=True, exist_ok=True)
        for branch, filename in (
            ("baseline", "baseline.pt"),
            ("baseline", "baseline_replay.pt"),
            ("candidate", "candidate.pt"),
        ):
            output = group / filename
            if _branch_complete(output, branch, warmup):
                print(json.dumps({"skip_complete": str(output)}), flush=True)
            else:
                _run(
                    _branch_command(branch, output, warmup),
                    group / filename.replace(".pt", ".console.log"),
                )
        paired = group / "paired.pt"
        if not args.dry_run:
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts/merge_support_long_horizon_branches.py"),
                    "--baseline",
                    str(group / "baseline.pt"),
                    "--baseline_replay",
                    str(group / "baseline_replay.pt"),
                    "--candidate",
                    str(group / "candidate.pt"),
                    "--output",
                    str(paired),
                    "--require_force_context",
                ],
                group / "merge.console.log",
            )
            groups[f"event{args.event_seed}_w{warmup}"] = _paired_summary(paired)
    if args.dry_run:
        return
    manifest = {
        "schema": "ContinuousSupportCandidatePairedPilotV1",
        "controller": "frozen model_1375",
        "candidate": {
            "mode": "q_error_closed_loop",
            "gain": 0.10,
            "limit_rad": 0.006,
            "application_steps": 250,
        },
        "event_seed": args.event_seed,
        "warmup_steps": args.warmup_steps,
        "num_envs": args.num_envs,
        "ppo_updates": False,
        "groups": groups,
    }
    output = args.output_dir / f"event{args.event_seed}_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "groups": groups}, indent=2))


if __name__ == "__main__":
    main()
