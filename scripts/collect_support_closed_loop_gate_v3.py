"""Collect leakage-safe V3 labels for a 0.5 s closed-loop support primitive.

The script reuses only the already-audited V2 baseline/baseline-replay branches.
It reruns the candidate with the deployment-matched dynamic feedback law and
never launches PPO.  Fresh final-test event seeds remain uncollected until the
V3 model, normalizer, OOD radius, thresholds, and dwell time are frozen.
"""

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
SOURCE = ROOT / (
    "logs/stand_coordination_v1/stationary_support_control_v2/"
    "force_conditioned_grid_v2"
)
DEFAULT_OUTPUT = ROOT / (
    "logs/stand_coordination_v1/stationary_support_control_v2/"
    "force_conditioned_closed_loop_grid_v3"
)
POLICY = ROOT / (
    "logs/rsl_rl/g1_stand_coordination_v1/"
    "2026-08-20_18-57-37_stand_coordination_v1_full_v1/model_1375.jit"
)
TASK = (
    "Isaac-G1-StandCoordination-FrozenNeighborGate-"
    "LowForce4to7N-100Pct-29DOF-v0"
)
SPLITS = {
    "train": (313, 509, 887, 1231),
    "validation": (1601, 2027),
}
FRESH_TEST_EVENT_SEEDS = (5501, 6607)
STAGES = (("t050", 25), ("t150", 75), ("t250", 125))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument(
    "--splits",
    nargs="+",
    choices=tuple(SPLITS),
    default=("train", "validation"),
)
parser.add_argument("--device", default="cuda:0")
# new change
parser.add_argument("--stop_after_groups", type=int, default=None)
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
def _source_group(split: str, event_seed: int, stage: str, warmup: int) -> Path:
    return SOURCE / f"{split}_event{event_seed}_{stage}_w{warmup}"


# new change
def _candidate_command(output: Path, event_seed: int, warmup: int) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/collect_support_long_horizon_branch.py"),
        "--task", TASK,
        "--jit_policy", str(POLICY),
        "--branch", "candidate",
        "--output", str(output),
        "--num_envs", "512",
        "--seed", str(event_seed),
        "--force_event_seed", str(event_seed),
        "--phase_offset", "0.0",
        "--warmup_steps", str(warmup),
        "--outcome_horizon_s", "5.0",
        "--teacher_gain", "0.10",
        "--teacher_limit_rad", "0.006",
        "--pulse_steps", "25",
        "--closed_loop_candidate",
        "--require_force_context",
        "--device", args.device,
        "--headless",
    ]


# new change
def _candidate_complete(path: Path, event_seed: int, warmup: int) -> bool:
    if not path.is_file():
        return False
    data = torch.load(path, map_location="cpu", weights_only=False)
    metadata = data["snapshot_fingerprint_metadata"]
    return bool(
        data["branch"] == "candidate"
        and data.get("candidate_mode") == "q_error_closed_loop"
        and data.get("closed_loop_candidate", False)
        and int(data["pulse_steps"]) == 25
        and int(metadata["force_event_seed"]) == event_seed
        and int(metadata["warmup_steps"]) == warmup
        and int(metadata["num_envs"]) == 512
    )


# new change
def _paired_complete(path: Path, event_seed: int, warmup: int) -> bool:
    if not path.is_file():
        return False
    data = torch.load(path, map_location="cpu", weights_only=False)
    metadata = data["snapshot_fingerprint_metadata"]
    return bool(
        data.get("strict_pairing", False)
        and data.get("force_conditioned_pairing", False)
        and data.get("candidate_mode") == "q_error_closed_loop"
        and data.get("closed_loop_candidate", False)
        and int(data["pulse_steps"]) == 25
        and int(metadata["force_event_seed"]) == event_seed
        and int(metadata["warmup_steps"]) == warmup
        and data["force_replay_match"].all()
        and data["force_candidate_match"].all()
        and max(map(float, data["baseline_replay_deltas"].values())) == 0.0
        and int(data["baseline_replay_failure_mismatch"]) == 0
    )


# new change
def _summary(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    valid = data["valid"].bool()
    failure_delta = data["paired_failure_delta"]
    baseline = data["baseline"]
    candidate = data["candidate"]
    output = {
        "paired_artifact": str(path.resolve()),
        "valid_samples": int(valid.sum()),
        "baseline_failures": int((baseline["failure"] & valid).sum()),
        "candidate_failures": int((candidate["failure"] & valid).sum()),
        "prevented_failures": int(((failure_delta < 0) & valid).sum()),
        "caused_failures": int(((failure_delta > 0) & valid).sum()),
        "beneficial_fraction": float(
            (data["candidate_beneficial"] & valid).sum() / valid.sum()
        ),
    }
    for steps, label in ((25, "0p5s"), (50, "1s"), (100, "2s")):
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
    groups = {}
    completed_this_run = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        for event_seed in SPLITS[split]:
            for stage, warmup in STAGES:
                name = f"{split}_event{event_seed}_{stage}_w{warmup}"
                source = _source_group(split, event_seed, stage, warmup)
                baseline = source / "baseline.pt"
                replay = source / "baseline_replay.pt"
                if not baseline.is_file() or not replay.is_file():
                    raise FileNotFoundError(f"Missing V2 audited source pair: {source}")
                output_group = args.output_dir / name
                output_group.mkdir(parents=True, exist_ok=True)
                candidate = output_group / "candidate.pt"
                if _candidate_complete(candidate, event_seed, warmup):
                    print(json.dumps({"skip_complete": str(candidate)}), flush=True)
                else:
                    _run(
                        _candidate_command(candidate, event_seed, warmup),
                        output_group / "candidate.console.log",
                    )
                paired = output_group / "paired.pt"
                if not _paired_complete(paired, event_seed, warmup):
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
                        output_group / "merge.console.log",
                    )
                if not _paired_complete(paired, event_seed, warmup):
                    raise RuntimeError(f"Post-merge audit failed: {paired}")
                groups[name] = {
                    "split": split,
                    "event_seed": event_seed,
                    "warmup_steps": warmup,
                    **_summary(paired),
                }
                completed_this_run += 1
                if (
                    args.stop_after_groups is not None
                    and completed_this_run >= args.stop_after_groups
                ):
                    break
            if (
                args.stop_after_groups is not None
                and completed_this_run >= args.stop_after_groups
            ):
                break
        if (
            args.stop_after_groups is not None
            and completed_this_run >= args.stop_after_groups
        ):
            break
    manifest = {
        "schema": "ForceConditionedClosedLoopGridV3",
        "controller": "frozen model_1375 JIT permanent fallback",
        "task": TASK,
        "candidate": {
            "mode": "q_error_closed_loop",
            "gain": 0.10,
            "limit_rad": 0.006,
            "dwell_steps": 25,
            "dwell_s": 0.5,
        },
        "train_event_seeds": SPLITS["train"],
        "validation_event_seeds": SPLITS["validation"],
        "fresh_test_event_seeds_reserved_uncollected": FRESH_TEST_EVENT_SEEDS,
        "test_labels_inspected": False,
        "requested_splits": sorted(set(args.splits)),
        "collection_complete": args.stop_after_groups is None,
        "ppo_updates": False,
        "groups": groups,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "groups": len(groups)}))


if __name__ == "__main__":
    main()
