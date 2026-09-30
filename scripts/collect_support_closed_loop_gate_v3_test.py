"""Collect fresh held-out V3 test pairs only after the threshold lock exists."""

# new change
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


# new change
ROOT = Path(__file__).absolute().parents[1]
GATE_ROOT = ROOT / (
    "logs/stand_coordination_v1/stationary_support_control_v2/"
    "multi_target_closed_loop_gate_v3"
)
DEFAULT_GATE = GATE_ROOT / "multi_target_closed_loop_safety_gate_v3.pt"
DEFAULT_LOCK = GATE_ROOT / "threshold_lock.json"
DEFAULT_OUTPUT = ROOT / (
    "logs/stand_coordination_v1/stationary_support_control_v2/"
    "force_conditioned_closed_loop_grid_v3_test"
)
POLICY = ROOT / (
    "logs/rsl_rl/g1_stand_coordination_v1/"
    "2026-08-20_18-57-37_stand_coordination_v1_full_v1/model_1375.jit"
)
TASK = (
    "Isaac-G1-StandCoordination-FrozenNeighborGate-"
    "LowForce4to7N-100Pct-29DOF-v0"
)
TEST_EVENT_SEEDS = (5501, 6607)
STAGES = (("t050", 25), ("t150", 75), ("t250", 125))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--locked_gate", type=Path, default=DEFAULT_GATE)
parser.add_argument("--threshold_lock", type=Path, default=DEFAULT_LOCK)
parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--stop_after_groups", type=int, default=None)
args = parser.parse_args()


# new change
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# new change
def _verify_lock() -> str:
    gate = torch.load(args.locked_gate, map_location="cpu", weights_only=False)
    lock = json.loads(args.threshold_lock.read_text())
    if gate.get("model_class") != "BootstrapEnsembleMultiTargetClosedLoopSafetyGateV3":
        raise RuntimeError("Unexpected V3 locked gate class.")
    if not gate.get("held_out_test_deferred", False):
        raise RuntimeError("V3 artifact was not frozen before test collection.")
    if gate.get("accepted_for_frozen_inference", True):
        raise RuntimeError("Pre-test V3 artifact must not already be accepted.")
    if gate.get("thresholds") is None or lock.get("thresholds_validation_only") is None:
        raise RuntimeError("Validation thresholds are not locked.")
    if gate["thresholds"] != lock["thresholds_validation_only"]:
        raise RuntimeError("Artifact and threshold-lock thresholds differ.")
    if lock.get("test_opened", True):
        raise RuntimeError("Threshold lock says the test was already opened.")
    if tuple(gate.get("fresh_test_event_seeds_reserved_uncollected", ())) != TEST_EVENT_SEEDS:
        raise RuntimeError("Reserved V3 test event seeds changed.")
    return _sha256(args.locked_gate)


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
def _branch_command(branch: str, output: Path, seed: int, warmup: int) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/collect_support_long_horizon_branch.py"),
        "--task", TASK,
        "--jit_policy", str(POLICY),
        "--branch", branch,
        "--output", str(output),
        "--num_envs", "512",
        "--seed", str(seed),
        "--force_event_seed", str(seed),
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
def _branch_complete(path: Path, branch: str, seed: int, warmup: int) -> bool:
    if not path.is_file():
        return False
    data = torch.load(path, map_location="cpu", weights_only=False)
    metadata = data["snapshot_fingerprint_metadata"]
    return bool(
        data["branch"] == branch
        and data.get("candidate_mode") == "q_error_closed_loop"
        and data.get("closed_loop_candidate", False)
        and int(data["pulse_steps"]) == 25
        and int(metadata["force_event_seed"]) == seed
        and int(metadata["warmup_steps"]) == warmup
        and int(metadata["num_envs"]) == 512
    )


# new change
def _paired_record(path: Path, seed: int, warmup: int) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not data.get("strict_pairing", False):
        raise RuntimeError(f"Not strict paired: {path}")
    if data.get("candidate_mode") != "q_error_closed_loop":
        raise RuntimeError(f"Wrong candidate mode: {path}")
    if max(map(float, data["baseline_replay_deltas"].values())) != 0.0:
        raise RuntimeError(f"Replay delta is nonzero: {path}")
    if int(data["baseline_replay_failure_mismatch"]) != 0:
        raise RuntimeError(f"Replay failure mismatch: {path}")
    if not data["force_replay_match"].all() or not data[
        "force_candidate_match"
    ].all():
        raise RuntimeError(f"Force/IK mismatch: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "snapshot_fingerprint": data["snapshot_fingerprint"],
        "force_event_seed": seed,
        "warmup_steps": warmup,
        "valid_samples": int(data["valid"].sum()),
    }


# new change
def main() -> None:
    locked_gate_sha256 = _verify_lock()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    completed = 0
    for seed in TEST_EVENT_SEEDS:
        for stage, warmup in STAGES:
            group = args.output_dir / f"test_event{seed}_{stage}_w{warmup}"
            group.mkdir(parents=True, exist_ok=True)
            for branch, filename in (
                ("baseline", "baseline.pt"),
                ("baseline", "baseline_replay.pt"),
                ("candidate", "candidate.pt"),
            ):
                output = group / filename
                if _branch_complete(output, branch, seed, warmup):
                    print(json.dumps({"skip_complete": str(output)}), flush=True)
                else:
                    _run(
                        _branch_command(branch, output, seed, warmup),
                        group / filename.replace(".pt", ".console.log"),
                    )
            paired = group / "paired.pt"
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts/merge_support_long_horizon_branches.py"),
                    "--baseline", str(group / "baseline.pt"),
                    "--baseline_replay", str(group / "baseline_replay.pt"),
                    "--candidate", str(group / "candidate.pt"),
                    "--output", str(paired),
                    "--require_force_context",
                ],
                group / "merge.console.log",
            )
            records.append(_paired_record(paired, seed, warmup))
            completed += 1
            if args.stop_after_groups is not None and completed >= args.stop_after_groups:
                break
        if args.stop_after_groups is not None and completed >= args.stop_after_groups:
            break
    if _sha256(args.locked_gate) != locked_gate_sha256:
        raise RuntimeError("Locked V3 gate changed during test collection.")
    manifest = {
        "schema": "StrictClosedLoopForceEventSeedTestV3",
        "locked_gate": str(args.locked_gate.resolve()),
        "locked_gate_sha256": locked_gate_sha256,
        "threshold_lock": str(args.threshold_lock.resolve()),
        "test_event_seeds": TEST_EVENT_SEEDS,
        "test_labels_inspected_during_collection": False,
        "collection_complete": args.stop_after_groups is None,
        "records": records,
        "ppo_updates": False,
    }
    output = args.output_dir / "test_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "groups": len(records)}, indent=2))


if __name__ == "__main__":
    main()
