"""Audit and lock train/validation records for the closed-loop Gate V3."""

# new change
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


# new change
ROOT = Path(__file__).absolute().parents[1]
DEFAULT_GRID = ROOT / (
    "logs/stand_coordination_v1/stationary_support_control_v2/"
    "force_conditioned_closed_loop_grid_v3/manifest.json"
)
DEFAULT_OUTPUT = ROOT / (
    "logs/stand_coordination_v1/stationary_support_control_v2/"
    "multi_target_closed_loop_gate_v3"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--grid_manifest", type=Path, default=DEFAULT_GRID)
parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
args = parser.parse_args()


# new change
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# new change
def _audit(group: dict) -> dict:
    path = Path(group["paired_artifact"])
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not data.get("strict_pairing", False):
        raise RuntimeError(f"Not strict paired data: {path}")
    if not data.get("force_conditioned_pairing", False):
        raise RuntimeError(f"Missing force conditioning: {path}")
    if data.get("candidate_mode") != "q_error_closed_loop":
        raise RuntimeError(f"Wrong candidate mode: {path}")
    if not data.get("closed_loop_candidate", False):
        raise RuntimeError(f"Closed-loop flag missing: {path}")
    if int(data["pulse_steps"]) != 25:
        raise RuntimeError(f"Wrong V3 dwell time: {path}")
    if data.get("ppo_updates", True) or data.get("uses_v3_checkpoint", True):
        raise RuntimeError(f"Disallowed checkpoint/training provenance: {path}")
    if max(map(float, data["snapshot_deltas"].values())) != 0.0:
        raise RuntimeError(f"Snapshot reconstruction mismatch: {path}")
    if max(map(float, data["baseline_replay_deltas"].values())) != 0.0:
        raise RuntimeError(f"Baseline replay mismatch: {path}")
    if int(data["baseline_replay_failure_mismatch"]) != 0:
        raise RuntimeError(f"Baseline failure replay mismatch: {path}")
    if not data["force_replay_match"].all() or not data[
        "force_candidate_match"
    ].all():
        raise RuntimeError(f"Force/IK trajectory mismatch: {path}")
    if int(data["force_conditioned_gate_input_dim"]) != 102:
        raise RuntimeError(f"Unexpected base feature dimension: {path}")
    valid = data["valid"].bool()
    if float(valid.float().mean()) < 0.99:
        raise RuntimeError(f"Too many invalid snapshots: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "snapshot_fingerprint": data["snapshot_fingerprint"],
        "force_event_seed": int(group["event_seed"]),
        "warmup_steps": int(group["warmup_steps"]),
        "valid_samples": int(valid.sum()),
    }


# new change
def main() -> None:
    grid = json.loads(args.grid_manifest.read_text())
    if grid.get("schema") != "ForceConditionedClosedLoopGridV3":
        raise RuntimeError("Unexpected V3 grid schema.")
    if not grid.get("collection_complete", False):
        raise RuntimeError("V3 train/validation collection is incomplete.")
    records = {"train": [], "validation": [], "test": []}
    for group in grid["groups"].values():
        split = group["split"]
        if split not in ("train", "validation"):
            raise RuntimeError(f"Unexpected collected split: {split}")
        records[split].append(_audit(group))
    if len(records["train"]) != 12 or len(records["validation"]) != 6:
        raise RuntimeError(
            f"Expected 12/6 train/validation groups, got "
            f"{len(records['train'])}/{len(records['validation'])}."
        )
    train_seeds = {record["force_event_seed"] for record in records["train"]}
    validation_seeds = {
        record["force_event_seed"] for record in records["validation"]
    }
    test_seeds = set(grid["fresh_test_event_seeds_reserved_uncollected"])
    if train_seeds & validation_seeds or train_seeds & test_seeds or validation_seeds & test_seeds:
        raise RuntimeError("Event-seed leakage across V3 splits.")
    split = {
        "schema": "StrictClosedLoopForceEventSeedSplitV3",
        "split_locked": True,
        "split_unit": "force_event_seed",
        "candidate": grid["candidate"],
        "records": records,
        "fresh_test_event_seeds_reserved_uncollected": sorted(test_seeds),
        "test_labels_inspected": False,
        "test_collection_deferred_until_threshold_lock": True,
        "ppo_updates": False,
        "uses_v3_checkpoint": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "split_manifest.json"
    output.write_text(json.dumps(split, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "train_groups": len(records["train"]),
                "validation_groups": len(records["validation"]),
                "test_groups": 0,
                "test_seeds_reserved": sorted(test_seeds),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
