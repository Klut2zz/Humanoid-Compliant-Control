#!/usr/bin/env python3
"""Infer a conservative 12DOF recovery target for an unseen strict snapshot."""

# new change
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


# new change
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # new change
    parser.add_argument(
        "--teacher",
        type=Path,
        nargs="+",
        required=True,
        help="One or more audited stage-specific oracle artifacts.",
    )
    parser.add_argument("--query_baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbor_count", type=int, default=5)
    parser.add_argument("--minimum_vote_fraction", type=float, default=0.60)
    parser.add_argument("--ood_quantile", type=float, default=0.99)
    return parser.parse_args()


# new change
def _snapshot_features(baseline: dict[str, object]) -> torch.Tensor:
    snapshot = baseline["snapshot"]
    return torch.cat(
        (
            snapshot["support_recovery_features"].float(),
            snapshot["support_temporal_features"].float(),
            snapshot["force_context_force_gate_features"].float(),
        ),
        dim=-1,
    )


# new change
def _motionwise_neighbor_distance(
    query: torch.Tensor,
    query_motion: torch.Tensor,
    reference: torch.Tensor,
    reference_motion: torch.Tensor,
    neighbor_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.full(
        (query.shape[0], neighbor_count), float("inf"), dtype=query.dtype
    )
    indices = torch.full(
        (query.shape[0], neighbor_count), -1, dtype=torch.long
    )
    divisor = math.sqrt(query.shape[1])
    for motion in query_motion.unique().tolist():
        query_ids = torch.where(query_motion == motion)[0]
        reference_ids = torch.where(reference_motion == motion)[0]
        if reference_ids.numel() == 0:
            continue
        k = min(neighbor_count, reference_ids.numel())
        local = torch.cdist(query[query_ids], reference[reference_ids]) / divisor
        values, local_ids = local.topk(k=k, largest=False)
        distances[query_ids, :k] = values
        indices[query_ids, :k] = reference_ids[local_ids]
    return distances, indices


# new change
def _training_ood_threshold(
    features: torch.Tensor,
    motion_id: torch.Tensor,
    quantile: float,
) -> float:
    nearest = torch.full((features.shape[0],), float("inf"))
    divisor = math.sqrt(features.shape[1])
    for motion in motion_id.unique().tolist():
        ids = torch.where(motion_id == motion)[0]
        if ids.numel() < 2:
            continue
        distance = torch.cdist(features[ids], features[ids]) / divisor
        distance.fill_diagonal_(float("inf"))
        nearest[ids] = distance.min(dim=1).values
    finite = nearest[torch.isfinite(nearest)]
    if finite.numel() == 0:
        raise RuntimeError("No same-motion neighbors are available for OOD fitting.")
    return float(torch.quantile(finite, quantile))


# new change
def main() -> None:
    args = parse_args()
    if args.neighbor_count < 1:
        raise ValueError("--neighbor_count must be positive.")
    if not 0.5 <= args.minimum_vote_fraction <= 1.0:
        raise ValueError("--minimum_vote_fraction must be in [0.5, 1].")
    if not 0.5 < args.ood_quantile < 1.0:
        raise ValueError("--ood_quantile must be in (0.5, 1).")

    # new change
    teachers = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.teacher
    ]
    teacher = teachers[0]
    baseline = torch.load(
        args.query_baseline, map_location="cpu", weights_only=False
    )
    for index, stage_teacher in enumerate(teachers):
        if stage_teacher.get("schema") != "ForceRecovery12DOFSingleAxisOracleV1":
            raise ValueError(f"Unexpected recovery teacher schema at index {index}.")
        if not stage_teacher.get("strict_pairing", False) or stage_teacher.get(
            "ppo_updates", True
        ):
            raise ValueError(f"Teacher {index} is not strict frozen paired data.")
        for key in (
            "support_action_ids",
            "support_joint_names",
            "selected_probe_names",
        ):
            if tuple(stage_teacher[key]) != tuple(teacher[key]):
                raise ValueError(f"Teacher {index} differs in {key}.")
        if float(stage_teacher["teacher_limit_rad"]) != float(
            teacher["teacher_limit_rad"]
        ):
            raise ValueError(f"Teacher {index} uses a different residual limit.")
    if baseline.get("branch") != "baseline":
        raise ValueError("Query artifact must be a baseline branch.")
    if not baseline.get("force_conditioned_pairing", False):
        raise ValueError("Query baseline lacks force-conditioned capture.")
    if baseline.get("ppo_updates", True):
        raise ValueError("Query baseline unexpectedly reports PPO updates.")
    if tuple(baseline["support_action_ids"]) != tuple(
        teacher["support_action_ids"]
    ):
        raise ValueError("Support action ordering differs from the teacher.")
    if tuple(baseline["candidate_joint_names"]) != tuple(
        teacher["support_joint_names"]
    ):
        raise ValueError("Support joint ordering differs from the teacher.")

    # new change
    # Concatenate only snapshots that were valid across every branch in their
    # own finite-difference matrix.  Stage identity is already represented by
    # event phase/force context and the 100D temporal state.
    teacher_valid = [
        stage_teacher.get(
            "valid",
            torch.ones(
                stage_teacher["state_features"].shape[0], dtype=torch.bool
            ),
        ).bool()
        for stage_teacher in teachers
    ]
    train_raw = torch.cat(
        [
            stage_teacher["state_features"][valid].float()
            for stage_teacher, valid in zip(teachers, teacher_valid)
        ]
    )
    query_raw = _snapshot_features(baseline)
    if train_raw.shape[1] != query_raw.shape[1]:
        raise RuntimeError(
            f"Feature dimensions differ: {train_raw.shape[1]} vs {query_raw.shape[1]}."
        )
    train_motion = torch.cat(
        [
            stage_teacher["motion_id"][valid].long()
            for stage_teacher, valid in zip(teachers, teacher_valid)
        ]
    )
    query_motion = baseline["snapshot"]["motion_id"].long()
    feature_mean = train_raw.mean(dim=0)
    feature_scale = train_raw.std(dim=0).clamp_min(0.01)
    train = (train_raw - feature_mean) / feature_scale
    query = (query_raw - feature_mean) / feature_scale

    ood_threshold = _training_ood_threshold(
        train, train_motion, args.ood_quantile
    )
    distance, neighbor_ids = _motionwise_neighbor_distance(
        query,
        query_motion,
        train,
        train_motion,
        args.neighbor_count,
    )
    selected_class = torch.cat(
        [
            stage_teacher["selected_probe_index"][valid].long() + 1
            for stage_teacher, valid in zip(teachers, teacher_valid)
        ]
    )
    train_target_rad = torch.cat(
        [
            stage_teacher["support_target_rad"][valid].float()
            for stage_teacher, valid in zip(teachers, teacher_valid)
        ]
    )
    num_classes = len(teacher["selected_probe_names"]) + 1
    weights = 1.0 / distance.clamp_min(1.0e-4)
    weights = torch.where(neighbor_ids >= 0, weights, torch.zeros_like(weights))
    neighbor_class = torch.zeros_like(neighbor_ids)
    valid_neighbor = neighbor_ids >= 0
    neighbor_class[valid_neighbor] = selected_class[neighbor_ids[valid_neighbor]]
    votes = torch.zeros((query.shape[0], num_classes), dtype=torch.float32)
    votes.scatter_add_(1, neighbor_class, weights)
    winning_weight, winning_class = votes.max(dim=1)
    vote_fraction = winning_weight / weights.sum(dim=1).clamp_min(1.0e-8)
    nearest_distance = distance[:, 0]
    in_distribution = nearest_distance <= ood_threshold
    agreed = vote_fraction >= args.minimum_vote_fraction
    apply = in_distribution & agreed & (winning_class > 0)

    prototypes = torch.zeros(
        (num_classes, train_target_rad.shape[1]),
        dtype=torch.float32,
    )
    for class_id in range(1, num_classes):
        rows = torch.where(selected_class == class_id)[0]
        if rows.numel() > 0:
            prototypes[class_id] = train_target_rad[rows[0]]
    support_target_rad = torch.zeros(
        (query.shape[0], prototypes.shape[1]), dtype=torch.float32
    )
    support_target_rad[apply] = prototypes[winning_class[apply]]
    if float(support_target_rad.abs().max()) > float(teacher["teacher_limit_rad"]) + 1e-9:
        raise RuntimeError("Inferred target exceeds teacher limit.")

    output = {
        "schema": "ForceRecovery12DOFNeighborInferenceV1",
        "support_target_rad": support_target_rad,
        "support_joint_names": teacher["support_joint_names"],
        "support_action_ids": teacher["support_action_ids"],
        "winning_class": winning_class,
        "vote_fraction": vote_fraction,
        "nearest_distance": nearest_distance,
        "in_distribution": in_distribution,
        "agreed": agreed,
        "applied": apply,
        "query_motion_id": query_motion,
        "query_snapshot_fingerprint": baseline["snapshot_fingerprint"],
        # new change
        "teacher_paths": tuple(str(path.resolve()) for path in args.teacher),
        "query_baseline_path": str(args.query_baseline.resolve()),
        "neighbor_count": args.neighbor_count,
        "minimum_vote_fraction": args.minimum_vote_fraction,
        "ood_quantile": args.ood_quantile,
        "ood_threshold": ood_threshold,
        "strict_pairing": True,
        "ppo_updates": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    summary = {
        "schema": output["schema"],
        "output": str(args.output.resolve()),
        "query_snapshot_fingerprint": baseline["snapshot_fingerprint"],
        "num_envs": int(query.shape[0]),
        "teacher_groups": len(teachers),
        "teacher_samples": int(train_raw.shape[0]),
        "state_feature_dim": int(query.shape[1]),
        "neighbor_count": args.neighbor_count,
        "ood_threshold": ood_threshold,
        "in_distribution": int(in_distribution.sum()),
        "agreed": int(agreed.sum()),
        "applied": int(apply.sum()),
        "applied_fraction": float(apply.float().mean()),
        "mean_vote_fraction": float(vote_fraction.mean()),
        "max_target_rad": float(support_target_rad.abs().max()),
        "ppo_updates": False,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
