"""Create an immutable event-seed split for the offline support safety gate."""

# new change
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


# new change
PROJECT_ROOT = Path(__file__).absolute().parents[1]
GRID_MANIFEST = PROJECT_ROOT / (
    "logs/stand_coordination_v1/stationary_support_control_v2/"
    "force_conditioned_grid_v1/manifest.json"
)
PRIOR_GROUPS = {
    42: PROJECT_ROOT / (
        "logs/stand_coordination_v1/stationary_support_control_v2/"
        "force_conditioned_pairs_v1/seed0_512x5s/paired.pt"
    ),
    7: PROJECT_ROOT / (
        "logs/stand_coordination_v1/stationary_support_control_v2/"
        "force_conditioned_pairs_v2/event7_seed7_512x5s/paired.pt"
    ),
    123: PROJECT_ROOT / (
        "logs/stand_coordination_v1/stationary_support_control_v2/"
        "force_conditioned_pairs_v2/event123_seed123_512x5s/paired.pt"
    ),
}
SPLIT_EVENT_SEEDS = {
    "train": (42, 11, 29, 71),
    "validation": (197,),
    "test": (7, 123),
}


# new change
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output_dir",
    type=Path,
    default=PROJECT_ROOT
    / (
        "logs/stand_coordination_v1/stationary_support_control_v2/"
        "dual_head_force_gate_v1"
    ),
)
args = parser.parse_args()


# new change
def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# new change
def _audit(path, expected_event_seed):
    data = torch.load(path, map_location="cpu", weights_only=False)
    metadata = data["snapshot_fingerprint_metadata"]
    event_seed = int(metadata.get("force_event_seed", 42))
    if event_seed != expected_event_seed:
        raise RuntimeError(
            f"Event-seed mismatch for {path}: {event_seed} != {expected_event_seed}"
        )
    if not data.get("strict_pairing", False) or not data.get(
        "force_conditioned_pairing", False
    ):
        raise RuntimeError(f"Not a strict force-conditioned pair: {path}")
    if data.get("ppo_updates", True) or data.get("uses_v3_checkpoint", True):
        raise RuntimeError(f"Disallowed PPO/V3 provenance: {path}")
    if data.get("baseline_replay_failure_mismatch", -1) != 0:
        raise RuntimeError(f"Failure replay mismatch: {path}")
    if max(map(float, data["baseline_replay_deltas"].values())) != 0.0:
        raise RuntimeError(f"Non-zero baseline replay delta: {path}")
    if not data["force_replay_match"].all() or not data[
        "force_candidate_match"
    ].all():
        raise RuntimeError(f"Force/IK context mismatch: {path}")
    if data["force_conditioned_gate_input"].shape[-1] != 102:
        raise RuntimeError(f"Unexpected gate input layout: {path}")
    valid = data["valid"].bool()
    failure = data["paired_failure_delta"][valid]
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "snapshot_fingerprint": data["snapshot_fingerprint"],
        "environment_seed": int(metadata["seed"]),
        "force_event_seed": event_seed,
        "warmup_steps": int(metadata["warmup_steps"]),
        "phase_offset": float(metadata["phase_offset"]),
        "valid_samples": int(valid.sum()),
        "candidate_prevented_failures": int((failure < 0).sum()),
        "candidate_caused_failures": int((failure > 0).sum()),
    }


# new change
def main():
    grid = json.loads(GRID_MANIFEST.read_text())
    if not grid.get("completed", False) or not grid.get(
        "ready_for_offline_gate_development", False
    ):
        raise RuntimeError("The 12-group source grid has not passed its audit.")
    records_by_event_seed = {}
    for seed, path in PRIOR_GROUPS.items():
        records_by_event_seed.setdefault(seed, []).append(_audit(path, seed))
    for group in grid["groups"].values():
        seed = int(group["force_event_seed"])
        path = Path(group["paired_artifact"])
        records_by_event_seed.setdefault(seed, []).append(_audit(path, seed))

    assigned = set()
    split_records = {}
    split_summary = {}
    for split, seeds in SPLIT_EVENT_SEEDS.items():
        overlap = assigned.intersection(seeds)
        if overlap:
            raise RuntimeError(f"Event-seed leakage across splits: {overlap}")
        assigned.update(seeds)
        records = []
        for seed in seeds:
            if seed not in records_by_event_seed:
                raise RuntimeError(f"Missing force-event seed {seed}")
            records.extend(records_by_event_seed[seed])
        split_records[split] = records
        split_summary[split] = {
            "force_event_seeds": seeds,
            "group_count": len(records),
            "valid_samples": sum(r["valid_samples"] for r in records),
            "candidate_prevented_failures": sum(
                r["candidate_prevented_failures"] for r in records
            ),
            "candidate_caused_failures": sum(
                r["candidate_caused_failures"] for r in records
            ),
        }
    if assigned != set(records_by_event_seed):
        raise RuntimeError(
            "Unassigned source event seeds: "
            f"{sorted(set(records_by_event_seed) - assigned)}"
        )
    manifest = {
        "schema": "StrictForceEventSeedSplitV1",
        "split_locked": True,
        "split_unit": "force_event_seed",
        "split_event_seeds": SPLIT_EVENT_SEEDS,
        "split_summary": split_summary,
        "records": split_records,
        "gate_input": "102D support+force context plus 8D motion one-hot",
        "test_policy": (
            "Test artifacts may be loaded only after model weights, OOD radius "
            "and selection thresholds are frozen from train/validation."
        ),
        "ppo_updates": False,
        "uses_v3_checkpoint": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "split_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), **split_summary}, indent=2))


if __name__ == "__main__":
    main()
