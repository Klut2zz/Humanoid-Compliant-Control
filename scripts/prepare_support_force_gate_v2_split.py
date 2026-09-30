"""Lock the V2 event-seed split without inspecting held-out test labels."""

# new change
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


# new change
PROJECT_ROOT = Path(__file__).absolute().parents[1]
DEFAULT_GRID = PROJECT_ROOT / (
    "logs/stand_coordination_v1/stationary_support_control_v2/"
    "force_conditioned_grid_v2/manifest.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "logs/stand_coordination_v1/stationary_support_control_v2/"
    "multi_target_force_gate_v2"
)
EXPECTED_SPLIT = {
    "train": (313, 509, 887, 1231),
    "validation": (1601, 2027),
    "test": (3001, 4093),
}


# new change
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--grid_manifest", type=Path, default=DEFAULT_GRID)
parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
args = parser.parse_args()


# new change
def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# new change
def _audit_record(group_name, group, inspect_labels):
    path = Path(group["paired_artifact"])
    data = torch.load(path, map_location="cpu", weights_only=False)
    metadata = data["snapshot_fingerprint_metadata"]
    expected_seed = int(group["force_event_seed"])
    if int(metadata["force_event_seed"]) != expected_seed:
        raise RuntimeError(f"Event seed mismatch: {path}")
    if not data.get("strict_pairing", False) or not data.get(
        "force_conditioned_pairing", False
    ):
        raise RuntimeError(f"Not a strict force pair: {path}")
    if data.get("ppo_updates", True) or data.get("uses_v3_checkpoint", True):
        raise RuntimeError(f"Disallowed PPO/V3 provenance: {path}")
    if data.get("baseline_replay_failure_mismatch", -1) != 0:
        raise RuntimeError(f"Replay failure mismatch: {path}")
    if max(map(float, data["baseline_replay_deltas"].values())) != 0.0:
        raise RuntimeError(f"Non-zero deterministic replay delta: {path}")
    if not data["force_replay_match"].all() or not data[
        "force_candidate_match"
    ].all():
        raise RuntimeError(f"Force/IK pairing mismatch: {path}")
    required_metrics = (
        "com_support_xy_distance_m_trajectory",
        "torso_tilt_rad_trajectory",
    )
    for branch in ("baseline", "candidate"):
        for metric in required_metrics:
            if data[branch].get(metric) is None:
                raise RuntimeError(f"Missing {branch}/{metric}: {path}")
            if data[branch][metric].shape[1] < 100:
                raise RuntimeError(f"Short V2 metric trajectory: {path}")
    valid = data["valid"].bool()
    record = {
        "group_name": group_name,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "snapshot_fingerprint": data["snapshot_fingerprint"],
        "force_event_seed": expected_seed,
        "warmup_steps": int(metadata["warmup_steps"]),
        "valid_samples": int(valid.sum()),
    }
    # Never inspect held-out test targets while preparing the split.
    if inspect_labels:
        failure_delta = data["paired_failure_delta"][valid]
        record["candidate_prevented_failures"] = int((failure_delta < 0).sum())
        record["candidate_caused_failures"] = int((failure_delta > 0).sum())
    return record


# new change
def main():
    grid = json.loads(args.grid_manifest.read_text())
    if not grid.get("completed", False) or not grid.get(
        "ready_for_v2_offline_training", False
    ):
        raise RuntimeError("V2 collection grid is not complete and audited.")
    actual_split = {
        name: tuple(values) for name, values in grid["split_event_seeds"].items()
    }
    if actual_split != EXPECTED_SPLIT:
        raise RuntimeError(
            f"V2 split changed: actual={actual_split}, expected={EXPECTED_SPLIT}"
        )
    all_seeds = [seed for seeds in actual_split.values() for seed in seeds]
    if len(set(all_seeds)) != len(all_seeds):
        raise RuntimeError("Event-seed leakage across V2 splits.")

    records = {name: [] for name in EXPECTED_SPLIT}
    for group_name, group in grid["groups"].items():
        split_name = group["split"]
        if int(group["force_event_seed"]) not in EXPECTED_SPLIT[split_name]:
            raise RuntimeError(f"Misassigned group: {group_name}")
        records[split_name].append(
            _audit_record(
                group_name,
                group,
                inspect_labels=split_name != "test",
            )
        )
    for split_name, split_records in records.items():
        expected_groups = len(EXPECTED_SPLIT[split_name]) * 3
        if len(split_records) != expected_groups:
            raise RuntimeError(
                f"{split_name}: {len(split_records)} groups != {expected_groups}"
            )

    manifest = {
        "schema": "StrictForceEventSeedSplitV2",
        "split_locked": True,
        "split_unit": "force_event_seed",
        "split_event_seeds": EXPECTED_SPLIT,
        "records": records,
        "label_schema": {
            "candidate_caused_failure": "paired_failure_delta > 0",
            "candidate_prevented_failure": "paired_failure_delta < 0",
            "short_horizons_s": (0.5, 1.0, 2.0),
            "com_improvement_m": "baseline mean - candidate mean",
            "tilt_improvement_rad": "baseline mean - candidate mean",
            "continuous_mask": "both branches survive the complete window",
        },
        "test_labels_inspected_during_split": False,
        "test_policy": (
            "Test artifacts may be loaded by the trainer only after weights, "
            "normalizers, OOD radius and decision thresholds are frozen."
        ),
        "ppo_updates": False,
        "uses_v3_checkpoint": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "split_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "groups": {key: len(value) for key, value in records.items()},
                "test_labels_inspected": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
