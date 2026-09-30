"""Collect the leakage-safe V2 short-horizon force-gate dataset.

Every group reconstructs one snapshot three times (baseline, baseline replay,
candidate).  This is frozen model_1375 inference only; PPO is never launched.
"""

# new change
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


# new change
PROJECT_ROOT = Path(__file__).absolute().parents[1]
DEFAULT_POLICY = PROJECT_ROOT / (
    "logs/rsl_rl/g1_stand_coordination_v1/"
    "2026-08-20_18-57-37_stand_coordination_v1_full_v1/model_1375.jit"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "logs/stand_coordination_v1/stationary_support_control_v2/"
    "force_conditioned_grid_v2"
)
TASK = (
    "Isaac-G1-StandCoordination-FrozenNeighborGate-"
    "LowForce4to7N-100Pct-29DOF-v0"
)
SPLIT_EVENT_SEEDS = {
    "train": (313, 509, 887, 1231),
    "validation": (1601, 2027),
    "test": (3001, 4093),
}
WARMUP_STAGES = (("t050", 25), ("t150", 75), ("t250", 125))


# new change
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--jit_policy", type=Path, default=DEFAULT_POLICY)
parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--outcome_horizon_s", type=float, default=5.0)
parser.add_argument("--stop_after_groups", type=int, default=None)
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()


# new change
def _write_manifest(path, payload):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


# new change
def _run_logged(command, log_path, echo_outcome=True):
    print(json.dumps({"launch": command, "log": str(log_path)}), flush=True)
    if args.dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as stream:
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
            if (
                (echo_outcome and "'output':" in line)
                or any(
                    token in line
                    for token in ("Traceback", "RuntimeError", "ValueError")
                )
            ):
                print(line.rstrip(), flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Child process failed with code {return_code}; see {log_path}."
        )


# new change
def _collector_command(branch, output, event_seed, warmup_steps):
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts/collect_support_long_horizon_branch.py"),
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
        str(event_seed),
        "--force_event_seed",
        str(event_seed),
        "--phase_offset",
        "0.0",
        "--warmup_steps",
        str(warmup_steps),
        "--outcome_horizon_s",
        str(args.outcome_horizon_s),
        "--teacher_gain",
        "0.10",
        "--teacher_limit_rad",
        "0.006",
        "--pulse_steps",
        "5",
        "--require_force_context",
        "--device",
        args.device,
        "--headless",
    ]


# new change
def _merge_command(group_dir):
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts/merge_support_long_horizon_branches.py"),
        "--baseline",
        str(group_dir / "baseline.pt"),
        "--baseline_replay",
        str(group_dir / "baseline_replay.pt"),
        "--candidate",
        str(group_dir / "candidate.pt"),
        "--output",
        str(group_dir / "paired.pt"),
        "--require_force_context",
    ]


# new change
def _branch_complete(path, branch, event_seed, warmup_steps):
    if not path.exists():
        return False
    data = torch.load(path, map_location="cpu", weights_only=False)
    metadata = data["snapshot_fingerprint_metadata"]
    result = data["result"]
    return bool(
        data["branch"] == branch
        and data.get("force_conditioned_pairing", False)
        and not data.get("ppo_updates", True)
        and not data.get("uses_v3_checkpoint", True)
        and int(metadata["force_event_seed"]) == event_seed
        and int(metadata["warmup_steps"]) == warmup_steps
        and int(metadata["num_envs"]) == args.num_envs
        and "com_support_xy_distance_m_trajectory" in result
        and "torso_tilt_rad_trajectory" in result
    )


# new change
def _paired_complete(path, event_seed, warmup_steps):
    if not path.exists():
        return False
    data = torch.load(path, map_location="cpu", weights_only=False)
    metadata = data["snapshot_fingerprint_metadata"]
    return bool(
        data.get("strict_pairing", False)
        and data.get("force_conditioned_pairing", False)
        and not data.get("ppo_updates", True)
        and not data.get("uses_v3_checkpoint", True)
        and int(metadata["force_event_seed"]) == event_seed
        and int(metadata["warmup_steps"]) == warmup_steps
        and data["force_replay_match"].all()
        and data["force_candidate_match"].all()
        and float(data["valid"].float().mean()) >= 0.95
        and "com_support_xy_distance_m_trajectory" in data["baseline"]
        and "torso_tilt_rad_trajectory" in data["baseline"]
        and "com_support_xy_distance_m_trajectory" in data["candidate"]
        and "torso_tilt_rad_trajectory" in data["candidate"]
    )


# new change
def _group_summary(path, inspect_targets=True):
    data = torch.load(path, map_location="cpu", weights_only=False)
    valid = data["valid"].bool()
    failure_delta = data["paired_failure_delta"]
    baseline = data["baseline"]
    candidate = data["candidate"]
    summary = {
        "valid": int(valid.sum()),
        "pre_snapshot_invalid": int((~valid).sum()),
    }
    # new change
    # A held-out test artifact is structurally audited but its outcome labels
    # must not be copied into the collection manifest.
    if not inspect_targets:
        summary["target_labels_inspected"] = False
        return summary
    summary["target_labels_inspected"] = True
    summary["candidate_prevented_failures"] = int(
        ((failure_delta < 0) & valid).sum()
    )
    summary["candidate_caused_failures"] = int(
        ((failure_delta > 0) & valid).sum()
    )
    for steps, name in ((25, "0p5s"), (50, "1p0s"), (100, "2p0s")):
        for metric in (
            "com_support_xy_distance_m_trajectory",
            "torso_tilt_rad_trajectory",
        ):
            delta = (
                candidate[metric][:, :steps].mean(dim=1)
                - baseline[metric][:, :steps].mean(dim=1)
            )
            summary[f"{metric}_{name}_candidate_minus_baseline_mean"] = float(
                delta[valid].mean()
            )
    return summary


# new change
def main():
    if not args.jit_policy.is_file():
        raise FileNotFoundError(args.jit_policy)
    if args.num_envs != 512 or args.outcome_horizon_s != 5.0:
        raise ValueError("V2 collection is locked to 512 envs and a 5 s horizon.")
    all_seeds = [seed for seeds in SPLIT_EVENT_SEEDS.values() for seed in seeds]
    if len(set(all_seeds)) != len(all_seeds):
        raise RuntimeError("Event-seed leakage in the V2 split definition.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "schema": "ForceConditionedShortHorizonGridV2",
        "task": TASK,
        "jit_policy": str(args.jit_policy.resolve()),
        "controller_role": "frozen model_1375 JIT permanent fallback",
        "split_event_seeds": SPLIT_EVENT_SEEDS,
        "warmup_stages": WARMUP_STAGES,
        "num_envs": args.num_envs,
        "outcome_horizon_s": args.outcome_horizon_s,
        "short_horizons_s": (0.5, 1.0, 2.0),
        "candidate": {"gain": 0.10, "limit_rad": 0.006, "pulse_steps": 5},
        "ppo_updates": False,
        "uses_v3_checkpoint": False,
        "test_labels_inspected_during_collection": False,
        "groups": {},
    }
    completed = 0
    for split_name, seeds in SPLIT_EVENT_SEEDS.items():
        for event_seed in seeds:
            for stage_name, warmup_steps in WARMUP_STAGES:
                group_name = f"{split_name}_event{event_seed}_{stage_name}_w{warmup_steps}"
                group_dir = args.output_dir / group_name
                group_dir.mkdir(parents=True, exist_ok=True)
                print(json.dumps({"group_start": group_name}), flush=True)
                for branch, filename in (
                    ("baseline", "baseline.pt"),
                    ("baseline", "baseline_replay.pt"),
                    ("candidate", "candidate.pt"),
                ):
                    output = group_dir / filename
                    if _branch_complete(output, branch, event_seed, warmup_steps):
                        print(json.dumps({"skip_complete": str(output)}), flush=True)
                    else:
                        _run_logged(
                            _collector_command(branch, output, event_seed, warmup_steps),
                            group_dir / filename.replace(".pt", ".console.log"),
                            echo_outcome=split_name != "test",
                        )
                paired = group_dir / "paired.pt"
                if not _paired_complete(paired, event_seed, warmup_steps):
                    _run_logged(
                        _merge_command(group_dir),
                        group_dir / "merge.console.log",
                        echo_outcome=split_name != "test",
                    )
                if not args.dry_run:
                    if not _paired_complete(paired, event_seed, warmup_steps):
                        raise RuntimeError(f"Strict V2 pair audit failed: {paired}")
                    manifest["groups"][group_name] = {
                        "split": split_name,
                        "force_event_seed": event_seed,
                        "warmup_steps": warmup_steps,
                        "paired_artifact": str(paired.resolve()),
                        "summary": _group_summary(
                            paired, inspect_targets=split_name != "test"
                        ),
                    }
                    _write_manifest(manifest_path, manifest)
                completed += 1
                if args.stop_after_groups is not None and completed >= args.stop_after_groups:
                    print("Reached --stop_after_groups; collection is resumable.", flush=True)
                    return
    manifest["completed"] = True
    manifest["completed_at_unix_s"] = time.time()
    manifest["group_count"] = len(manifest["groups"])
    manifest["ready_for_v2_offline_training"] = len(manifest["groups"]) == 24
    _write_manifest(manifest_path, manifest)
    print(json.dumps({"grid_complete": str(manifest_path)}), flush=True)


if __name__ == "__main__":
    main()
