"""Resumable strict baseline/replay/candidate low-force collection grid.

This orchestration script only launches frozen-policy inference collectors and
the strict merger.  It never launches train.py or performs a PPO update.
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
    "force_conditioned_grid_v1"
)
TASK = (
    "Isaac-G1-StandCoordination-FrozenNeighborGate-"
    "LowForce4to7N-100Pct-29DOF-v0"
)
FORCE_EVENT_SEEDS = (11, 29, 71, 197)
# 0.5, 1.5 and 2.5 s after reset: pre/ramp, active and recovery/new-event
# coverage is verified from recorded event_phase/IK state, not assumed.
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
def _write_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


# new change
def _run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {"launch": command, "log": str(log_path)}, ensure_ascii=False
        ),
        flush=True,
    )
    if args.dry_run:
        return
    with log_path.open("w") as log_file:
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
            log_file.write(line)
            if (
                "'output':" in line
                or "Traceback" in line
                or "RuntimeError" in line
                or "ValueError" in line
            ):
                print(line.rstrip(), flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Child process failed with code {return_code}; see {log_path}."
        )


# new change
def _branch_is_complete(path, branch, event_seed, warmup_steps):
    if not path.exists():
        return False
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        metadata = artifact["snapshot_fingerprint_metadata"]
        return bool(
            artifact["branch"] == branch
            and artifact.get("force_conditioned_pairing", False)
            and not artifact.get("ppo_updates", True)
            and not artifact.get("uses_v3_checkpoint", True)
            and int(metadata["force_event_seed"]) == event_seed
            and int(metadata["warmup_steps"]) == warmup_steps
            and int(metadata["num_envs"]) == args.num_envs
        )
    except Exception as error:
        raise RuntimeError(
            f"Existing branch artifact is incomplete/corrupt: {path}: {error}"
        ) from error


# new change
def _paired_is_complete(path, event_seed, warmup_steps):
    if not path.exists():
        return False
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        metadata = artifact["snapshot_fingerprint_metadata"]
        return bool(
            artifact.get("strict_pairing", False)
            and artifact.get("force_conditioned_pairing", False)
            and not artifact.get("ppo_updates", True)
            and not artifact.get("uses_v3_checkpoint", True)
            and int(metadata["force_event_seed"]) == event_seed
            and int(metadata["warmup_steps"]) == warmup_steps
            and artifact["force_replay_match"].all()
            and artifact["force_candidate_match"].all()
            # new change
            # Late snapshots may contain a small number of environments that
            # already failed before the paired intervention.  They are
            # explicitly excluded by ``valid``; do not confuse that with a
            # replay/context mismatch.  Refuse the group only if more than 5%
            # of snapshot starts are unusable.
            and float(artifact["valid"].float().mean()) >= 0.95
            and artifact["force_conditioned_gate_input"].shape
            == (args.num_envs, 102)
        )
    except Exception as error:
        raise RuntimeError(
            f"Existing paired artifact is incomplete/corrupt: {path}: {error}"
        ) from error


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
def _group_summary(path):
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    valid = artifact["valid"]
    force = artifact["force_gate_features"][valid]
    labels = artifact["paired_failure_delta"][valid]
    return {
        "valid": int(valid.sum()),
        # new change
        "pre_snapshot_invalid": int((~valid).sum()),
        "left_wrist": int(force[:, 4].sum()),
        "right_wrist": int(force[:, 5].sum()),
        "ik_pending": int(force[:, 10].sum()),
        "ik_accepted": int(force[:, 11].sum()),
        "ik_rejected": int(force[:, 12].sum()),
        "force_active": int(force[:, 13].sum()),
        "baseline_failures": int(
            (artifact["baseline"]["failure"] & valid).sum()
        ),
        "candidate_failures": int(
            (artifact["candidate"]["failure"] & valid).sum()
        ),
        "candidate_prevented_failures": int((labels < 0).sum()),
        "candidate_caused_failures": int((labels > 0).sum()),
        "paired_score_delta_mean": float(
            artifact["paired_score_delta"][valid].mean()
        ),
    }


# new change
def main():
    if not args.jit_policy.is_file():
        raise FileNotFoundError(args.jit_policy)
    if args.num_envs != 512:
        raise ValueError("The accepted collection grid requires 512 environments.")
    if args.outcome_horizon_s != 5.0:
        raise ValueError("The accepted collection grid requires a 5 s label window.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "task": TASK,
        "jit_policy": str(args.jit_policy.resolve()),
        "controller_role": "frozen model_1375 JIT permanent fallback",
        "force_event_seeds": FORCE_EVENT_SEEDS,
        "warmup_stages": WARMUP_STAGES,
        "num_envs": args.num_envs,
        "outcome_horizon_s": args.outcome_horizon_s,
        "candidate": {"gain": 0.10, "limit_rad": 0.006, "pulse_steps": 5},
        "ppo_updates": False,
        "groups": {},
    }
    completed_this_run = 0
    for event_seed in FORCE_EVENT_SEEDS:
        for stage_name, warmup_steps in WARMUP_STAGES:
            group_name = f"event{event_seed}_{stage_name}_w{warmup_steps}"
            group_dir = args.output_dir / group_name
            group_dir.mkdir(parents=True, exist_ok=True)
            print(
                json.dumps(
                    {
                        "group_start": group_name,
                        "force_event_seed": event_seed,
                        "warmup_steps": warmup_steps,
                    }
                ),
                flush=True,
            )
            branches = (
                ("baseline", "baseline.pt"),
                ("baseline", "baseline_replay.pt"),
                ("candidate", "candidate.pt"),
            )
            for branch, filename in branches:
                output = group_dir / filename
                if _branch_is_complete(
                    output, branch, event_seed, warmup_steps
                ):
                    print(json.dumps({"skip_complete": str(output)}), flush=True)
                    continue
                _run_logged(
                    _collector_command(
                        branch, output, event_seed, warmup_steps
                    ),
                    group_dir / filename.replace(".pt", ".console.log"),
                )
            paired = group_dir / "paired.pt"
            if not _paired_is_complete(paired, event_seed, warmup_steps):
                _run_logged(
                    _merge_command(group_dir),
                    group_dir / "merge.console.log",
                )
            if args.dry_run:
                # new change
                completed_this_run += 1
                if (
                    args.stop_after_groups is not None
                    and completed_this_run >= args.stop_after_groups
                ):
                    print(
                        "Reached --stop_after_groups during dry-run.",
                        flush=True,
                    )
                    return
                continue
            if not _paired_is_complete(paired, event_seed, warmup_steps):
                raise RuntimeError(f"Strict pair audit failed: {paired}")
            summary = _group_summary(paired)
            manifest["groups"][group_name] = {
                "force_event_seed": event_seed,
                "environment_seed": event_seed,
                "warmup_steps": warmup_steps,
                "snapshot_time_s": warmup_steps * 0.02,
                "paired_artifact": str(paired.resolve()),
                "summary": summary,
            }
            _write_manifest(manifest_path, manifest)
            completed_this_run += 1
            print(
                json.dumps(
                    {"group_complete": group_name, "summary": summary}
                ),
                flush=True,
            )
            if (
                args.stop_after_groups is not None
                and completed_this_run >= args.stop_after_groups
            ):
                print("Reached --stop_after_groups; collection is resumable.", flush=True)
                return
    # new change
    summaries = [
        group["summary"] for group in manifest["groups"].values()
    ]
    aggregate_keys = (
        "valid",
        "pre_snapshot_invalid",
        "left_wrist",
        "right_wrist",
        "ik_pending",
        "ik_accepted",
        "ik_rejected",
        "force_active",
        "baseline_failures",
        "candidate_failures",
        "candidate_prevented_failures",
        "candidate_caused_failures",
    )
    aggregate = {
        key: sum(int(summary[key]) for summary in summaries)
        for key in aggregate_keys
    }
    aggregate["group_count"] = len(summaries)
    aggregate["total_requested_starts"] = (
        len(summaries) * args.num_envs
    )
    aggregate["valid_fraction"] = (
        aggregate["valid"] / aggregate["total_requested_starts"]
    )
    aggregate["paired_score_delta_mean_over_groups"] = sum(
        float(summary["paired_score_delta_mean"])
        for summary in summaries
    ) / max(len(summaries), 1)
    quality_gate = {
        "twelve_groups_complete": len(summaries) == 12,
        "valid_fraction_at_least_95pct": aggregate["valid_fraction"] >= 0.95,
        "candidate_prevented_failure_labels_at_least_100": (
            aggregate["candidate_prevented_failures"] >= 100
        ),
        "candidate_caused_failure_labels_at_least_100": (
            aggregate["candidate_caused_failures"] >= 100
        ),
        "pending_accepted_rejected_all_covered": min(
            aggregate["ik_pending"],
            aggregate["ik_accepted"],
            aggregate["ik_rejected"],
        )
        > 0,
        "both_wrists_at_least_40pct": min(
            aggregate["left_wrist"], aggregate["right_wrist"]
        )
        / max(aggregate["valid"], 1)
        >= 0.40,
    }
    manifest["aggregate"] = aggregate
    manifest["quality_gate"] = quality_gate
    manifest["paired_dataset_index"] = [
        group["paired_artifact"] for group in manifest["groups"].values()
    ]
    manifest["ready_for_offline_gate_development"] = all(
        quality_gate.values()
    )
    manifest["gate_fitted_or_modified"] = False
    manifest["completed"] = True
    manifest["completed_at_unix_s"] = time.time()
    _write_manifest(manifest_path, manifest)
    print(json.dumps({"grid_complete": str(manifest_path)}), flush=True)


if __name__ == "__main__":
    main()
