"""Fit a conservative force-conditioned support gate from strict pairs.

This is an offline non-parametric fit.  It never launches IsaacLab, loads an
RSL-RL checkpoint, or performs a PPO update.
"""

# new change
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


# new change
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--datasets", nargs="+", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--neighbor_count", type=int, default=12)
parser.add_argument("--confidence_sem_multiplier", type=float, default=1.0)
parser.add_argument("--score_deadband", type=float, default=0.005)
parser.add_argument("--score_clip", type=float, default=0.5)
parser.add_argument("--state_ood_quantile", type=float, default=0.995)
args = parser.parse_args()


# new change
SUPPORT_FEATURE_DIM = 87
FORCE_FEATURE_DIM = 15
MOTION_PHASE_COLUMNS = (50, 51)
STATE_SAFETY_COLUMNS = tuple(range(20, 40)) + tuple(range(52, 87))
# Force magnitude/direction, signed event phase, phase sin/cos, envelope and
# feasible scale.  Wrist and IK state are exact categorical strata below.
FORCE_DISTANCE_COLUMNS = (87, 88, 89, 90, 93, 94, 95, 96, 101)


# new change
def _load_groups(paths):
    groups = []
    feature_names = None
    action_ids = None
    controller_metadata = None
    fingerprints = set()
    for text in paths:
        path = Path(text)
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not data.get("strict_pairing", False) or not data.get(
            "force_conditioned_pairing", False
        ):
            raise ValueError(f"Not a strict force-conditioned pair: {path}")
        if data.get("ppo_updates", True) or data.get("uses_v3_checkpoint", True):
            raise ValueError(f"Disallowed PPO/V3 provenance in {path}")
        if data.get("baseline_replay_failure_mismatch", -1) != 0:
            raise ValueError(f"Baseline failure replay mismatch in {path}")
        if not torch.all(data["force_replay_match"]):
            raise ValueError(f"Force replay mismatch in {path}")
        fingerprint = str(data["snapshot_fingerprint"])
        if fingerprint in fingerprints:
            raise ValueError(f"Duplicate reconstructed group: {path}")
        fingerprints.add(fingerprint)
        current_names = tuple(data["force_gate_feature_names"])
        current_action_ids = tuple(data["support_action_ids"])
        current_metadata = {
            "teacher_gain": float(data["teacher_gain"]),
            "teacher_limit_rad": float(data["teacher_limit_rad"]),
            "pulse_steps": int(data["pulse_steps"]),
            "outcome_horizon_steps": int(data["outcome_horizon_steps"]),
            "outcome_horizon_s": float(data["outcome_horizon_s"]),
            "action_scale_rad": 0.30,
        }
        if feature_names is None:
            feature_names = current_names
            action_ids = current_action_ids
            controller_metadata = current_metadata
        elif feature_names != current_names:
            raise ValueError("Force feature layouts differ across groups.")
        elif action_ids != current_action_ids:
            raise ValueError("Support action layouts differ across groups.")
        elif controller_metadata != current_metadata:
            raise ValueError("Candidate controller metadata differs across groups.")
        valid = data["valid"].bool() & data["force_candidate_match"].bool()
        features = data["force_conditioned_gate_input"][valid].float()
        if features.shape[-1] != SUPPORT_FEATURE_DIM + FORCE_FEATURE_DIM:
            raise ValueError(f"Expected 102D gate input, got {features.shape[-1]}.")
        groups.append(
            {
                "path": str(path.resolve()),
                "seed": int(data["snapshot_fingerprint_metadata"]["seed"]),
                # new change
                "force_event_seed": int(
                    data["snapshot_fingerprint_metadata"].get(
                        "force_event_seed", 42
                    )
                ),
                "fingerprint": fingerprint,
                "features": features,
                "score": data["paired_score_delta"][valid].float(),
                "failure": data["paired_failure_delta"][valid].to(torch.int8),
                "motion": data["motion_id"][valid].long(),
            }
        )
    if len({group["seed"] for group in groups}) != len(groups):
        raise ValueError("Each held-out group must use a distinct seed.")
    # new change
    if len({group["force_event_seed"] for group in groups}) != len(groups):
        raise ValueError(
            "Each held-out group must use a distinct force_event_seed."
        )
    return groups, feature_names, action_ids, controller_metadata


# new change
def _wrist_slot(features):
    return torch.argmax(features[:, 91:93], dim=-1)


# new change
def _ik_status(features):
    # pending, accepted, rejected; inactive accepted events remain accepted.
    return torch.argmax(features[:, 97:100], dim=-1)


# new change
def _continuous_columns():
    return torch.tensor(MOTION_PHASE_COLUMNS + FORCE_DISTANCE_COLUMNS)


# new change
def _state_scale(features):
    columns = torch.tensor(STATE_SAFETY_COLUMNS)
    return features[:, columns].std(dim=0).clamp_min(0.01)


# new change
def _distance_scale(features):
    columns = _continuous_columns()
    # Preserve directional geometry while preventing one scalar from
    # dominating nearest-neighbor selection.
    return features[:, columns].std(dim=0).clamp_min(0.05)


# new change
def _find_neighbors(train, query, count, distance_scale):
    train_features = train["features"]
    query_features = query["features"]
    columns = _continuous_columns()
    neighbors = []
    available = torch.zeros(query_features.shape[0], dtype=torch.bool)
    query_wrist = _wrist_slot(query_features)
    query_status = _ik_status(query_features)
    train_wrist = _wrist_slot(train_features)
    train_status = _ik_status(train_features)
    for query_index in range(query_features.shape[0]):
        mask = (
            (train["motion"] == query["motion"][query_index])
            & (train_wrist == query_wrist[query_index])
            & (train_status == query_status[query_index])
        )
        indices = torch.where(mask)[0]
        if indices.numel() == 0:
            neighbors.append(torch.empty(0, dtype=torch.long))
            continue
        delta = (
            train_features[indices][:, columns]
            - query_features[query_index, columns]
        ) / distance_scale
        distance = torch.sqrt(torch.mean(delta.square(), dim=-1))
        k = min(count, indices.numel())
        neighbors.append(indices[distance.topk(k, largest=False).indices])
        available[query_index] = True
    return neighbors, available


# new change
def _fit_state_ood_threshold(train, distance_scale, state_scale):
    """Fit the OOD radius from leave-one-out training states only."""
    state_columns = torch.tensor(STATE_SAFETY_COLUMNS)
    neighbors, available = _find_neighbors(
        train, train, args.neighbor_count + 1, distance_scale
    )
    distances = []
    for index, neighbor_ids in enumerate(neighbors):
        neighbor_ids = neighbor_ids[neighbor_ids != index][
            : args.neighbor_count
        ]
        if neighbor_ids.numel() == 0:
            continue
        local_state = train["features"][neighbor_ids][
            :, state_columns
        ].mean(dim=0)
        distances.append(
            torch.sqrt(
                torch.mean(
                    (
                        (
                            train["features"][index, state_columns]
                            - local_state
                        )
                        / state_scale
                    ).square()
                )
            )
        )
    if not distances or not torch.all(available):
        raise RuntimeError("Unable to fit state OOD threshold in every stratum.")
    return torch.quantile(
        torch.stack(distances), args.state_ood_quantile
    )


# new change
def _evaluate(train_groups, validation):
    train = {
        key: torch.cat([group[key] for group in train_groups])
        for key in ("features", "score", "failure", "motion")
    }
    distance_scale = _distance_scale(train["features"])
    state_scale = _state_scale(train["features"])
    ood_threshold = _fit_state_ood_threshold(
        train, distance_scale, state_scale
    )
    state_columns = torch.tensor(STATE_SAFETY_COLUMNS)
    neighbors, available = _find_neighbors(
        train, validation, args.neighbor_count, distance_scale
    )
    predicted_upper = torch.full_like(validation["score"], float("inf"))
    failure_veto = torch.ones(validation["score"].shape[0], dtype=torch.bool)
    state_distance = torch.full_like(validation["score"], float("inf"))
    for index, neighbor_ids in enumerate(neighbors):
        if neighbor_ids.numel() == 0:
            continue
        local_score = train["score"][neighbor_ids].clamp(
            -args.score_clip, args.score_clip
        )
        sem = (
            local_score.std(unbiased=False) / math.sqrt(neighbor_ids.numel())
        )
        predicted_upper[index] = local_score.mean() + (
            args.confidence_sem_multiplier * sem
        )
        failure_veto[index] = torch.any(train["failure"][neighbor_ids] > 0)
        local_state = train["features"][neighbor_ids][:, state_columns].mean(dim=0)
        state_distance[index] = torch.sqrt(
            torch.mean(
                (
                    (
                        validation["features"][index, state_columns]
                        - local_state
                    )
                    / state_scale
                ).square()
            )
        )
    state_ood = (~available) | (state_distance > ood_threshold)
    selected = (
        (predicted_upper < -args.score_deadband)
        & ~failure_veto
        & ~state_ood
    )
    gated_delta = torch.where(
        selected, validation["score"], torch.zeros_like(validation["score"])
    )
    count = int(selected.sum())
    beneficial = (validation["failure"] < 0) | (
        (validation["failure"] == 0)
        & (validation["score"] < -args.score_deadband)
    )
    return {
        "held_out_seed": validation["seed"],
        "held_out_force_event_seed": validation["force_event_seed"],
        "samples": int(selected.numel()),
        "neighbor_available": int(available.sum()),
        "selected": count,
        "selected_fraction": float(selected.float().mean()),
        "selected_precision": float(
            (selected & beneficial).sum().float() / max(count, 1)
        ),
        "selected_candidate_caused_failures": int(
            (selected & (validation["failure"] > 0)).sum()
        ),
        "gated_policy_score_delta_mean": float(gated_delta.mean()),
        "always_on_score_delta_mean": float(validation["score"].mean()),
        "state_ood_count": int(state_ood.sum()),
        "state_ood_threshold": float(ood_threshold),
    }


# new change
def main():
    groups, feature_names, action_ids, controller_metadata = _load_groups(
        args.datasets
    )
    if len(groups) < 3:
        raise ValueError("At least three distinct seeds are required.")
    reports = []
    for held_out, validation in enumerate(groups):
        train = [group for index, group in enumerate(groups) if index != held_out]
        report = _evaluate(train, validation)
        reports.append(report)
        print(report, flush=True)

    all_features = torch.cat([group["features"] for group in groups])
    all_score = torch.cat([group["score"] for group in groups])
    all_failure = torch.cat([group["failure"] for group in groups])
    all_motion = torch.cat([group["motion"] for group in groups])
    # new change
    final_distance_scale = _distance_scale(all_features)
    final_state_scale = _state_scale(all_features)
    final_train = {
        "features": all_features,
        "score": all_score,
        "failure": all_failure,
        "motion": all_motion,
    }
    final_state_ood_threshold = _fit_state_ood_threshold(
        final_train, final_distance_scale, final_state_scale
    )
    accepted = all(
        report["selected_candidate_caused_failures"] == 0
        and report["gated_policy_score_delta_mean"] <= 0.0
        for report in reports
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "force_conditioned_neighbor_gate.pt"
    torch.save(
        {
            "model_class": "ForceConditionedPairedNeighborGate",
            "features": all_features,
            "paired_score_delta": all_score,
            "paired_failure_delta": all_failure,
            "motion_id": all_motion,
            "force_gate_feature_names": feature_names,
            "motion_phase_columns": MOTION_PHASE_COLUMNS,
            "state_safety_columns": STATE_SAFETY_COLUMNS,
            "force_distance_columns": FORCE_DISTANCE_COLUMNS,
            "force_distance_scale": final_distance_scale,
            "state_scale": final_state_scale,
            "state_ood_threshold": final_state_ood_threshold,
            "neighbor_count": args.neighbor_count,
            "confidence_sem_multiplier": args.confidence_sem_multiplier,
            "score_deadband": args.score_deadband,
            "score_clip": args.score_clip,
            "support_action_ids": action_ids,
            "controller_metadata": controller_metadata,
            "held_out_seed_reports": reports,
            "accepted_for_frozen_inference": accepted,
            "strict_force_pairing": True,
            "ppo_updates": False,
            "uses_v3_checkpoint": False,
        },
        artifact_path,
    )
    summary = {
        "artifact": str(artifact_path.resolve()),
        "sample_count": int(all_features.shape[0]),
        "gate_input_dim": int(all_features.shape[1]),
        "seeds": [group["seed"] for group in groups],
        "force_event_seeds": [
            group["force_event_seed"] for group in groups
        ],
        "candidate_caused_failure_labels": int((all_failure > 0).sum()),
        "candidate_prevented_failure_labels": int((all_failure < 0).sum()),
        "accepted_for_frozen_inference": accepted,
        "ppo_updates": False,
        "held_out_seed_reports": reports,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
