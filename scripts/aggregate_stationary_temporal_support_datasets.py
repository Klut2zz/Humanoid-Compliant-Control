#!/usr/bin/env python3
"""Aggregate strict paired support teachers collected on visited controller states."""

# new change
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


# new change
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


# new change
def main() -> None:
    args = parse_args()
    datasets = [torch.load(path, map_location="cpu", weights_only=False) for path in args.inputs]
    required = (
        "support_temporal_adapter",
        "support_target_rad",
        "support_target_actor",
        "accepted",
        "valid_baseline",
        "motion_id",
        "source_time_s",
        "split",
        "previous_support_residual_rad",
        "steps_since_residual_update",
    )
    for path, data in zip(args.inputs, datasets, strict=True):
        missing = [name for name in required if name not in data]
        if missing:
            raise RuntimeError(f"{path} lacks {missing}.")
        if data.get("ppo_updates", True):
            raise RuntimeError(f"{path} is not an inference-only teacher dataset.")
        if data["support_temporal_adapter"].shape[-1] != 98:
            raise RuntimeError(f"{path} does not use the 98D temporal layout.")

    concatenated = {
        name: torch.cat([data[name] for data in datasets], dim=0)
        for name in required
    }
    stage_index = torch.cat(
        [
            torch.full((data["support_temporal_adapter"].shape[0],), index, dtype=torch.long)
            for index, data in enumerate(datasets)
        ]
    )
    output = {
        "schema": "StationaryTemporalSupportDatasetAggregationV1",
        **concatenated,
        "stage_index": stage_index,
        "source_datasets": [str(path.resolve()) for path in args.inputs],
        "source_policy": "frozen model_1375",
        "feature_order": datasets[0]["feature_order"],
        "support_action_ids": tuple(datasets[0]["support_action_ids"]),
        "support_joint_names": tuple(datasets[0]["support_joint_names"]),
        "action_scale_rad": 0.30,
        "strict_pairing": True,
        "dataset_aggregation": True,
        "ppo_updates": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)

    accepted = output["accepted"].bool()
    split_names = {0: "train", 1: "validation", 2: "test"}
    report = {
        "schema": output["schema"],
        "samples": int(accepted.numel()),
        "accepted": int(accepted.sum()),
        "accepted_fraction": float(accepted.float().mean()),
        "stages": len(datasets),
        "stage_samples": [int(data["accepted"].numel()) for data in datasets],
        "stage_accepted": [int(data["accepted"].sum()) for data in datasets],
        "elapsed_steps_unique": sorted(
            set(map(int, output["steps_since_residual_update"].tolist()))
        ),
        "previous_residual_abs_mean_mrad": float(
            output["previous_support_residual_rad"].abs().mean() * 1000.0
        ),
        "target_abs_mean_mrad": float(output["support_target_rad"].abs().mean() * 1000.0),
        "target_abs_max_mrad": float(output["support_target_rad"].abs().max() * 1000.0),
        "accepted_split_counts": {
            name: int(((output["split"] == code) & accepted).sum())
            for code, name in split_names.items()
        },
        "strict_pairing": True,
        "ppo_updates": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
