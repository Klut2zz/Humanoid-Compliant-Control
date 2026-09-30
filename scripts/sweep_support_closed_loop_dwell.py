"""Sweep persistent closed-loop support-candidate dwell time on audited snapshots."""

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
DEFAULT_ROOT = ROOT / (
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
MOTIONS = (
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
parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
parser.add_argument("--jit_policy", type=Path, default=DEFAULT_POLICY)
parser.add_argument("--event_seed", type=int, default=5051)
parser.add_argument("--warmup_steps", type=int, nargs="+", default=(25, 75, 125))
parser.add_argument("--pulse_steps", type=int, nargs="+", default=(25, 50, 100, 250))
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()


# new change
def _run(command: list[str], log: Path) -> None:
    print(json.dumps({"launch": command, "log": str(log)}), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode:
        raise RuntimeError(f"Child failed ({result.returncode}); see {log}.")


# new change
def _candidate_command(output: Path, warmup: int, pulse: int) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/collect_support_long_horizon_branch.py"),
        "--task", TASK,
        "--jit_policy", str(args.jit_policy),
        "--branch", "candidate",
        "--output", str(output),
        "--num_envs", str(args.num_envs),
        "--seed", str(args.event_seed),
        "--force_event_seed", str(args.event_seed),
        "--phase_offset", "0.0",
        "--warmup_steps", str(warmup),
        "--outcome_horizon_s", "5.0",
        "--teacher_gain", "0.10",
        "--teacher_limit_rad", "0.006",
        "--pulse_steps", str(pulse),
        "--closed_loop_candidate",
        "--require_force_context",
        "--device", args.device,
        "--headless",
    ]


# new change
def _complete(path: Path, warmup: int, pulse: int) -> bool:
    if not path.is_file():
        return False
    data = torch.load(path, map_location="cpu", weights_only=False)
    metadata = data["snapshot_fingerprint_metadata"]
    return bool(
        data["branch"] == "candidate"
        and data.get("candidate_mode") == "q_error_closed_loop"
        and int(data["pulse_steps"]) == pulse
        and int(metadata["warmup_steps"]) == warmup
        and int(metadata["force_event_seed"]) == args.event_seed
        and int(metadata["num_envs"]) == args.num_envs
    )


# new change
def _summarize(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    valid = data["valid"].bool()
    delta = data["paired_failure_delta"]
    score = data["paired_score_delta"]
    baseline = data["baseline"]
    candidate = data["candidate"]
    correction = candidate["candidate_correction_rad_trajectory"]
    return {
        "artifact": str(path.resolve()),
        "valid": int(valid.sum()),
        "baseline_failures": int((baseline["failure"] & valid).sum()),
        "candidate_failures": int((candidate["failure"] & valid).sum()),
        "prevented_failures": int(((delta < 0) & valid).sum()),
        "caused_failures": int(((delta > 0) & valid).sum()),
        "beneficial_fraction": float(
            (data["candidate_beneficial"] & valid).sum() / valid.sum()
        ),
        "score_delta_mean": float(score[valid].mean()),
        "score_delta_median": float(score[valid].median()),
        "mean_correction_mrad_during_dwell": float(
            correction[valid, : data["pulse_steps"]].abs().mean() * 1000.0
        ),
        "saturation_fraction_during_dwell": float(
            (
                correction[valid, : data["pulse_steps"]].abs()
                >= 0.005999
            ).float().mean()
        ),
        "caused_motion_names": [
            MOTIONS[index]
            for index in data["motion_id"][(delta > 0) & valid].tolist()
        ],
        "prevented_motion_names": [
            MOTIONS[index]
            for index in data["motion_id"][(delta < 0) & valid].tolist()
        ],
    }


# new change
def main() -> None:
    rows = []
    for warmup in args.warmup_steps:
        source = args.root / f"event{args.event_seed}_w{warmup}"
        baseline = source / "baseline.pt"
        replay = source / "baseline_replay.pt"
        if not baseline.is_file() or not replay.is_file():
            raise FileNotFoundError(f"Missing audited baseline pair in {source}.")
        for pulse in args.pulse_steps:
            dwell = args.root / "dwell_sweep" / f"w{warmup}_p{pulse}"
            dwell.mkdir(parents=True, exist_ok=True)
            candidate = dwell / "candidate.pt"
            if pulse == 250:
                candidate = source / "candidate.pt"
            elif not _complete(candidate, warmup, pulse):
                _run(
                    _candidate_command(candidate, warmup, pulse),
                    dwell / "candidate.console.log",
                )
            paired = dwell / "paired.pt"
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts/merge_support_long_horizon_branches.py"),
                    "--baseline", str(baseline),
                    "--baseline_replay", str(replay),
                    "--candidate", str(candidate),
                    "--output", str(paired),
                    "--require_force_context",
                ],
                dwell / "merge.console.log",
            )
            row = _summarize(paired)
            row.update(
                {
                    "warmup_steps": warmup,
                    "pulse_steps": pulse,
                    "dwell_s": pulse * 0.02,
                }
            )
            rows.append(row)
    aggregates = {}
    for pulse in args.pulse_steps:
        selected = [row for row in rows if row["pulse_steps"] == pulse]
        aggregates[str(pulse)] = {
            "dwell_s": pulse * 0.02,
            "baseline_failures": sum(row["baseline_failures"] for row in selected),
            "candidate_failures": sum(row["candidate_failures"] for row in selected),
            "prevented_failures": sum(row["prevented_failures"] for row in selected),
            "caused_failures": sum(row["caused_failures"] for row in selected),
            "beneficial_fraction_mean": sum(
                row["beneficial_fraction"] for row in selected
            ) / len(selected),
            "score_delta_mean": sum(row["score_delta_mean"] for row in selected)
            / len(selected),
        }
    manifest = {
        "schema": "ClosedLoopSupportDwellSweepV1",
        "controller": "frozen model_1375",
        "event_seed": args.event_seed,
        "gain": 0.10,
        "limit_rad": 0.006,
        "ppo_updates": False,
        "rows": rows,
        "aggregates": aggregates,
    }
    output = args.root / "dwell_sweep" / "summary.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    main()
