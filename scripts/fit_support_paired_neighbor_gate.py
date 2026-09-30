"""Fit and validate a conservative non-parametric support state gate.

The gate uses same-motion phase neighbors for causal benefit estimation and
the remaining support state for an out-of-distribution safety veto.  This is
offline fitting only: no IsaacLab simulation and no PPO update is performed.
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


# new change: q-q_ref, dq, COM/torso/root/feet/contact/failure-trend state.
STATE_SAFETY_COLUMNS = tuple(range(20, 40)) + tuple(range(52, 87))


# new change
def load_groups(paths):
    groups = []
    offsets = set()
    fingerprints = set()
    common_action_ids = None
    common_controller_metadata = None
    for path_text in paths:
        path = Path(path_text)
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not bool(data.get("strict_pairing", False)):
            raise ValueError(f"Not strictly paired: {path}")
        if not bool(data.get("deterministic_process_reconstruction", False)):
            raise ValueError(f"Not deterministically reconstructed: {path}")
        if data.get("ppo_updates", True) or data.get("uses_v3_checkpoint", True):
            raise ValueError(f"Disallowed PPO/V3 provenance in {path}")
        if data.get("baseline_replay_failure_mismatch", -1) != 0:
            raise ValueError(f"Baseline failure mismatch in {path}")
        if max(map(float, data["baseline_replay_deltas"].values())) > 1.0e-7:
            raise ValueError(f"Baseline replay mismatch in {path}")
        if max(map(float, data["snapshot_deltas"].values())) > 1.0e-7:
            raise ValueError(f"Snapshot mismatch in {path}")
        offset = float(data["phase_offset"])
        fingerprint = str(data["snapshot_fingerprint"])
        if offset in offsets or fingerprint in fingerprints:
            raise ValueError(f"Duplicate phase group: {path}")
        offsets.add(offset)
        fingerprints.add(fingerprint)
        valid = data["valid"].bool()
        action_ids = tuple(data["support_action_ids"])
        controller_metadata = {
            "teacher_gain": float(data["teacher_gain"]),
            "teacher_limit_rad": float(data["teacher_limit_rad"]),
            "pulse_steps": int(data["pulse_steps"]),
            "outcome_horizon_steps": int(data["outcome_horizon_steps"]),
            "action_scale_rad": 0.30,
        }
        if common_action_ids is None:
            common_action_ids = action_ids
            common_controller_metadata = controller_metadata
        elif common_action_ids != action_ids:
            raise ValueError("Support action ordering mismatch.")
        elif common_controller_metadata != controller_metadata:
            raise ValueError("Support candidate metadata differs between groups.")
        groups.append(
            {
                "path": str(path.resolve()),
                "offset": offset,
                "fingerprint": fingerprint,
                "features": data["support_recovery_features"][valid].float(),
                "score": data["paired_score_delta"][valid].float(),
                "failure": data["paired_failure_delta"][valid].to(torch.int8),
                "motion": data["motion_id"][valid].long(),
            }
        )
    return (
        sorted(groups, key=lambda item: item["offset"]),
        common_action_ids,
        common_controller_metadata,
    )


# new change
def fit_state_statistics(features, motion, neighbor_count):
    columns = torch.tensor(STATE_SAFETY_COLUMNS)
    scale = features[:, columns].std(dim=0).clamp_min(0.01)
    leave_one_out_distance = []
    for motion_id in motion.unique().tolist():
        indices = torch.where(motion == motion_id)[0]
        phase = features[indices, 50:52]
        distance = torch.cdist(phase, phase)
        distance.fill_diagonal_(float("inf"))
        k = min(neighbor_count, indices.numel() - 1)
        neighbors = distance.topk(k, largest=False).indices
        local_mean = features[indices][neighbors][:, :, columns].mean(dim=1)
        normalized_distance = torch.sqrt(
            (((features[indices][:, columns] - local_mean) / scale) ** 2).mean(dim=1)
        )
        leave_one_out_distance.append(normalized_distance)
    threshold = torch.quantile(
        torch.cat(leave_one_out_distance), args.state_ood_quantile
    )
    return scale, threshold


# new change
def evaluate(train_groups, validation):
    features = torch.cat([group["features"] for group in train_groups])
    score = torch.cat([group["score"] for group in train_groups])
    failure = torch.cat([group["failure"] for group in train_groups])
    motion = torch.cat([group["motion"] for group in train_groups])
    scale, ood_threshold = fit_state_statistics(
        features, motion, args.neighbor_count
    )
    columns = torch.tensor(STATE_SAFETY_COLUMNS)
    query = validation["features"]
    predicted_upper = torch.empty_like(validation["score"])
    failure_veto = torch.zeros_like(validation["failure"], dtype=torch.bool)
    state_ood = torch.zeros_like(validation["failure"], dtype=torch.bool)
    for motion_id in validation["motion"].unique().tolist():
        train_indices = torch.where(motion == motion_id)[0]
        query_indices = torch.where(validation["motion"] == motion_id)[0]
        phase_distance = torch.cdist(
            query[query_indices, 50:52], features[train_indices, 50:52]
        )
        k = min(args.neighbor_count, train_indices.numel())
        neighbors = phase_distance.topk(k, largest=False).indices
        local_score = score[train_indices][neighbors].clamp(
            -args.score_clip, args.score_clip
        )
        predicted_upper[query_indices] = local_score.mean(dim=1) + (
            args.confidence_sem_multiplier
            * local_score.std(dim=1)
            / math.sqrt(k)
        )
        failure_veto[query_indices] = (
            failure[train_indices][neighbors] > 0
        ).any(dim=1)
        local_state_mean = features[train_indices][neighbors][
            :, :, columns
        ].mean(dim=1)
        state_distance = torch.sqrt(
            (
                (
                    (query[query_indices][:, columns] - local_state_mean)
                    / scale
                )
                ** 2
            ).mean(dim=1)
        )
        state_ood[query_indices] = state_distance > ood_threshold
    selected = (
        (predicted_upper < -args.score_deadband)
        & ~failure_veto
        & ~state_ood
    )
    policy_delta = torch.where(
        selected, validation["score"], torch.zeros_like(validation["score"])
    )
    beneficial = (validation["failure"] < 0) | (
        (validation["failure"] == 0)
        & (validation["score"] < -args.score_deadband)
    )
    count = int(selected.sum())
    return {
        "held_out_phase_offset": validation["offset"],
        "samples": int(selected.numel()),
        "selected": count,
        "selected_fraction": float(selected.float().mean()),
        "selected_precision": float(
            (selected & beneficial).sum().float() / max(count, 1)
        ),
        "gated_policy_score_delta_mean": float(policy_delta.mean()),
        "always_on_score_delta_mean": float(validation["score"].mean()),
        "selected_candidate_caused_failures": int(
            (selected & (validation["failure"] > 0)).sum()
        ),
        "available_candidate_caused_failures": int(
            (validation["failure"] > 0).sum()
        ),
        "state_ood_count": int(state_ood.sum()),
        "training_state_ood_threshold": float(ood_threshold),
    }


# new change
def main():
    groups, support_action_ids, controller_metadata = load_groups(args.datasets)
    if len(groups) < 4:
        raise ValueError("At least four phase groups are required.")
    reports = []
    for held_out in range(len(groups)):
        train = [group for index, group in enumerate(groups) if index != held_out]
        report = evaluate(train, groups[held_out])
        reports.append(report)
        print(report, flush=True)

    all_features = torch.cat([group["features"] for group in groups])
    all_score = torch.cat([group["score"] for group in groups])
    all_failure = torch.cat([group["failure"] for group in groups])
    all_motion = torch.cat([group["motion"] for group in groups])
    state_scale, state_ood_threshold = fit_state_statistics(
        all_features, all_motion, args.neighbor_count
    )
    fold_safe = all(
        report["gated_policy_score_delta_mean"] <= 0.0
        and report["selected_candidate_caused_failures"] == 0
        for report in reports
    )
    failure_count = int((all_failure != 0).sum())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "paired_phase_state_neighbor_gate.pt"
    torch.save(
        {
            "model_class": "PairedPhaseNeighborGateWithStateOODVeto",
            "features": all_features,
            "paired_score_delta": all_score,
            "paired_failure_delta": all_failure,
            "motion_id": all_motion,
            "state_safety_columns": STATE_SAFETY_COLUMNS,
            "state_scale": state_scale,
            "state_ood_threshold": state_ood_threshold,
            "neighbor_count": args.neighbor_count,
            "confidence_sem_multiplier": args.confidence_sem_multiplier,
            "score_deadband": args.score_deadband,
            "score_clip": args.score_clip,
            "support_action_ids": support_action_ids,
            "controller_metadata": controller_metadata,
            "phase_offsets": [group["offset"] for group in groups],
            "snapshot_fingerprints": [group["fingerprint"] for group in groups],
            "strict_pairing": True,
            "ppo_updates": False,
            "uses_v3_checkpoint": False,
        },
        artifact,
    )
    summary = {
        "artifact": str(artifact.resolve()),
        "sample_count": int(all_features.shape[0]),
        "phase_group_count": len(groups),
        "phase_offsets": [group["offset"] for group in groups],
        "failure_labeled_samples": failure_count,
        "failure_supervision_sufficient": failure_count >= 10,
        "all_held_out_offsets_safe": fold_safe,
        "phase_manifold_gate_accepted": fold_safe,
        "off_manifold_failure_gate_accepted": fold_safe and failure_count >= 10,
        "fold_reports": reports,
        "ppo_updates": False,
    }
    (output_dir / "fit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# Paired Phase/State Neighbor Gate",
        "",
        "Strictly paired offline fit. No PPO and no V3 checkpoint were used.",
        "",
        f"- Samples: {all_features.shape[0]}",
        f"- Phase groups: {len(groups)}",
        f"- Failure labels: {failure_count}",
        f"- All held-out offsets safe: {fold_safe}",
        f"- Phase-manifold gate accepted: {fold_safe}",
        f"- Off-manifold failure gate accepted: {fold_safe and failure_count >= 10}",
        "",
    ]
    for report in reports:
        lines.append(
            "- offset={:.9f}: selected={}/{} ({:.3f}), precision={:.3f}, "
            "gated_delta={:.6f}, failure_harm={}, OOD={}".format(
                report["held_out_phase_offset"],
                report["selected"],
                report["samples"],
                report["selected_fraction"],
                report["selected_precision"],
                report["gated_policy_score_delta_mean"],
                report["selected_candidate_caused_failures"],
                report["state_ood_count"],
            )
        )
    (output_dir / "GATE_ACCEPTANCE.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
