"""Train a conservative state gate from strictly paired long-horizon outcomes.

This is offline supervised learning only.  It neither creates an IsaacLab
environment nor performs PPO updates.
"""

# new change
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# new change
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--datasets", nargs="+", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=3.0e-4)
parser.add_argument("--score_deadband", type=float, default=0.005)
parser.add_argument("--full_benefit", type=float, default=0.050)
parser.add_argument("--score_clip", type=float, default=0.50)
parser.add_argument("--deployment_threshold", type=float, default=0.70)
parser.add_argument("--failure_weight", type=float, default=40.0)
parser.add_argument("--neighbor_count", type=int, default=12)
parser.add_argument("--confidence_sem_multiplier", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()


# new change
class PairedStateGate(nn.Module):
    """Predict candidate usefulness and paired long-horizon score delta."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(87, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        self.gate_logit = nn.Linear(64, 1)
        self.score_delta = nn.Linear(64, 1)
        # Conservative prior: do not apply the candidate before evidence.
        nn.init.zeros_(self.gate_logit.weight)
        nn.init.constant_(self.gate_logit.bias, -2.0)

    def forward(self, features: torch.Tensor):
        latent = self.encoder(features)
        return (
            self.gate_logit(latent).squeeze(-1),
            self.score_delta(latent).squeeze(-1),
        )


# new change
def _safe_float(value) -> float:
    return float(value.detach().cpu() if torch.is_tensor(value) else value)


# new change
def _load_groups(paths: list[str]):
    groups = []
    fingerprints = set()
    offsets = set()
    common_ids = None
    for path_text in paths:
        path = Path(path_text)
        data = torch.load(path, map_location="cpu", weights_only=False)
        required_true = (
            "strict_pairing",
            "deterministic_process_reconstruction",
        )
        if not all(bool(data.get(key, False)) for key in required_true):
            raise ValueError(f"{path} is not a strictly reconstructed pair.")
        if bool(data.get("ppo_updates", True)):
            raise ValueError(f"{path} reports PPO updates; refusing offline gate fit.")
        if bool(data.get("uses_v3_checkpoint", True)):
            raise ValueError(f"{path} uses a V3 checkpoint; model_1375 is required.")
        if int(data.get("baseline_replay_failure_mismatch", -1)) != 0:
            raise ValueError(f"{path} has baseline replay failure mismatch.")
        replay_max = max(map(float, data["baseline_replay_deltas"].values()))
        snapshot_max = max(map(float, data["snapshot_deltas"].values()))
        if replay_max > 1.0e-7 or snapshot_max > 1.0e-7:
            raise ValueError(f"{path} exceeds strict replay tolerance.")
        offset = float(data["phase_offset"])
        fingerprint = str(data["snapshot_fingerprint"])
        if offset in offsets or fingerprint in fingerprints:
            raise ValueError(f"Duplicate phase group or fingerprint in {path}.")
        offsets.add(offset)
        fingerprints.add(fingerprint)
        valid = data["valid"].bool()
        features = data["support_recovery_features"][valid].float()
        score_delta = data["paired_score_delta"][valid].float()
        failure_delta = data["paired_failure_delta"][valid].to(torch.int8)
        motion_id = data["motion_id"][valid].long()
        if features.ndim != 2 or features.shape[1] != 87:
            raise ValueError(f"Expected [N,87] state in {path}, got {features.shape}.")
        action_ids = tuple(data["support_action_ids"])
        if common_ids is None:
            common_ids = action_ids
        elif action_ids != common_ids:
            raise ValueError("Support action ordering differs between groups.")
        groups.append(
            {
                "path": str(path.resolve()),
                "phase_offset": offset,
                "fingerprint": fingerprint,
                "features": features,
                "score_delta": score_delta,
                "failure_delta": failure_delta,
                "motion_id": motion_id,
            }
        )
    if len(groups) < 3:
        raise ValueError("At least three independent phase-offset groups are required.")
    return sorted(groups, key=lambda item: item["phase_offset"]), common_ids


# new change
def _targets(score_delta: torch.Tensor, failure_delta: torch.Tensor):
    denominator = args.full_benefit - args.score_deadband
    if denominator <= 0.0:
        raise ValueError("--full_benefit must exceed --score_deadband.")
    target = torch.clamp(
        (-score_delta - args.score_deadband) / denominator,
        min=0.0,
        max=1.0,
    )
    target = torch.where(failure_delta < 0, torch.ones_like(target), target)
    target = torch.where(failure_delta > 0, torch.zeros_like(target), target)
    hard_beneficial = (failure_delta < 0) | (
        (failure_delta == 0) & (score_delta < -args.score_deadband)
    )
    return target, hard_beneficial


# new change
def _neighbor_teacher(features, score_delta, failure_delta, motion_id):
    """Build a robust target using training-fold neighbors only.

    The upper confidence estimate deliberately asks for repeatable local
    benefit, rather than fitting the noisy sign of one five-second rollout.
    """
    phase = features[:, 50:52]
    robust_score = torch.empty_like(score_delta)
    neighboring_failure = torch.zeros_like(failure_delta)
    for motion in motion_id.unique().tolist():
        indices = torch.where(motion_id == motion)[0]
        motion_phase = phase[indices]
        distance = torch.cdist(motion_phase, motion_phase)
        distance.fill_diagonal_(float("inf"))
        k = min(args.neighbor_count, max(indices.numel() - 1, 1))
        neighbors = distance.topk(k=k, largest=False).indices
        local_score = score_delta[indices][neighbors].clamp(
            -args.score_clip, args.score_clip
        )
        upper_confidence = local_score.mean(dim=1) + (
            args.confidence_sem_multiplier
            * local_score.std(dim=1)
            / math.sqrt(k)
        )
        local_failure = failure_delta[indices][neighbors]
        robust_score[indices] = upper_confidence
        neighboring_failure[indices] = torch.where(
            (local_failure > 0).any(dim=1),
            torch.ones(indices.numel(), dtype=torch.int8),
            torch.zeros(indices.numel(), dtype=torch.int8),
        )
    return robust_score, neighboring_failure


# new change
def _roc_auc(probability: torch.Tensor, label: torch.Tensor) -> float:
    positive = label.bool()
    negative = ~positive
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(probability)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, order.numel() + 1, dtype=torch.float64)
    rank_sum = ranks[positive].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# new change
def _evaluate(probability, predicted_delta, score_delta, failure_delta, motion_id):
    _, hard_beneficial = _targets(score_delta, failure_delta)
    selected = probability >= args.deployment_threshold
    selected_count = int(selected.sum())
    true_selected = int((selected & hard_beneficial).sum())
    candidate_caused_failure = failure_delta > 0
    prevented_failure = failure_delta < 0
    policy_delta = torch.where(selected, score_delta, torch.zeros_like(score_delta))
    report = {
        "samples": int(score_delta.numel()),
        "gate_probability_mean": _safe_float(probability.mean()),
        "gate_probability_max": _safe_float(probability.max()),
        "selected_count": selected_count,
        "selected_fraction": selected_count / max(score_delta.numel(), 1),
        "selected_precision": true_selected / max(selected_count, 1),
        "selected_score_delta_mean": (
            _safe_float(score_delta[selected].mean()) if selected_count else None
        ),
        "gated_policy_score_delta_mean": _safe_float(policy_delta.mean()),
        "candidate_always_on_score_delta_mean": _safe_float(score_delta.mean()),
        "selected_candidate_caused_failures": int(
            (selected & candidate_caused_failure).sum()
        ),
        "available_candidate_caused_failures": int(candidate_caused_failure.sum()),
        "selected_prevented_failures": int((selected & prevented_failure).sum()),
        "hard_beneficial_fraction": _safe_float(hard_beneficial.float().mean()),
        "hard_label_auc": _roc_auc(probability, hard_beneficial),
        "score_delta_mae": _safe_float((predicted_delta - score_delta).abs().mean()),
        "per_motion": {},
    }
    for motion in sorted(motion_id.unique().tolist()):
        mask = motion_id == motion
        chosen = selected & mask
        report["per_motion"][str(motion)] = {
            "samples": int(mask.sum()),
            "selected": int(chosen.sum()),
            "selected_fraction": _safe_float(chosen.float().sum() / mask.sum()),
            "gated_policy_score_delta_mean": _safe_float(policy_delta[mask].mean()),
            "always_on_score_delta_mean": _safe_float(score_delta[mask].mean()),
            "selected_failure_harm": int((chosen & candidate_caused_failure).sum()),
        }
    return report


# new change
def _fit_fold(groups, held_out_index, device):
    train_groups = [group for index, group in enumerate(groups) if index != held_out_index]
    validation = groups[held_out_index]
    train_features = torch.cat([group["features"] for group in train_groups])
    train_score = torch.cat([group["score_delta"] for group in train_groups])
    train_failure = torch.cat([group["failure_delta"] for group in train_groups])
    train_motion = torch.cat([group["motion_id"] for group in train_groups])
    teacher_score, teacher_failure = _neighbor_teacher(
        train_features, train_score, train_failure, train_motion
    )
    target_gate, _ = _targets(teacher_score, teacher_failure)
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0).clamp_min(1.0e-4)
    normalized_train = (train_features - feature_mean) / feature_std
    normalized_validation = (
        validation["features"] - feature_mean
    ) / feature_std

    model = PairedStateGate().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4
    )
    generator = torch.Generator().manual_seed(args.seed + held_out_index)
    for epoch in range(args.epochs):
        order = torch.randperm(train_features.shape[0], generator=generator)
        model.train()
        for start in range(0, order.numel(), args.batch_size):
            indices = order[start : start + args.batch_size]
            x = normalized_train[indices].to(device)
            target = target_gate[indices].to(device)
            delta = teacher_score[indices].clamp(
                -args.score_clip, args.score_clip
            ).to(device)
            failure = teacher_failure[indices].to(device)
            logit, predicted_delta = model(x)
            weights = 1.0 + 2.0 * target + args.failure_weight * (failure != 0)
            gate_loss = (
                F.binary_cross_entropy_with_logits(logit, target, reduction="none")
                * weights
            ).mean()
            delta_loss = F.smooth_l1_loss(
                predicted_delta, delta, beta=0.02
            )
            # A failure-causing candidate is never allowed a high gate score.
            failure_barrier = (
                F.softplus(logit[failure > 0]).mean()
                if bool((failure > 0).any())
                else logit.sum() * 0.0
            )
            loss = gate_loss + 0.25 * delta_loss + 2.0 * failure_barrier
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

    # new change: use a fixed epoch budget.  The held-out phase group is never
    # used for model/epoch selection, so the reported fold remains genuinely
    # out of distribution with respect to phase_offset.
    model.eval()
    with torch.inference_mode():
        logit, predicted_delta = model(normalized_validation.to(device))
        validation_target, _ = _targets(
            validation["score_delta"], validation["failure_delta"]
        )
        validation_loss = F.binary_cross_entropy_with_logits(
            logit, validation_target.to(device)
        ) + 0.25 * F.smooth_l1_loss(
            predicted_delta,
            validation["score_delta"].clamp(
                -args.score_clip, args.score_clip
            ).to(device),
            beta=0.02,
        )
    report = _evaluate(
        torch.sigmoid(logit).cpu(),
        predicted_delta.cpu(),
        validation["score_delta"],
        validation["failure_delta"],
        validation["motion_id"],
    )
    report.update(
        {
            "held_out_phase_offset": validation["phase_offset"],
            "held_out_fingerprint": validation["fingerprint"],
            "fixed_training_epochs": args.epochs,
            "validation_loss": float(validation_loss),
        }
    )
    return {
        "state_dict": copy.deepcopy(model).cpu().state_dict(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "report": report,
    }


# new change
def main() -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    groups, support_action_ids = _load_groups(args.datasets)
    folds = []
    for held_out_index in range(len(groups)):
        fold = _fit_fold(groups, held_out_index, device)
        folds.append(fold)
        print({"fold": fold["report"]}, flush=True)

    # Each sample is unseen by exactly one fold.  This produces a genuine
    # leave-one-phase-offset-out estimate for the conservative ensemble.
    oof_reports = [fold["report"] for fold in folds]
    all_features = torch.cat([group["features"] for group in groups])
    all_score = torch.cat([group["score_delta"] for group in groups])
    all_failure = torch.cat([group["failure_delta"] for group in groups])
    all_motion = torch.cat([group["motion_id"] for group in groups])
    ensemble_probabilities = []
    ensemble_deltas = []
    with torch.inference_mode():
        for fold in folds:
            model = PairedStateGate()
            model.load_state_dict(fold["state_dict"])
            normalized = (all_features - fold["feature_mean"]) / fold["feature_std"]
            logit, predicted_delta = model(normalized)
            ensemble_probabilities.append(torch.sigmoid(logit))
            ensemble_deltas.append(predicted_delta)
    conservative_probability = torch.stack(ensemble_probabilities).amin(dim=0)
    mean_predicted_delta = torch.stack(ensemble_deltas).mean(dim=0)
    ensemble_report = _evaluate(
        conservative_probability,
        mean_predicted_delta,
        all_score,
        all_failure,
        all_motion,
    )

    failure_count = int((all_failure != 0).sum())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "paired_state_gate_ensemble.pt"
    torch.save(
        {
            "model_class": "PairedStateGate87DScalar",
            "folds": [
                {
                    "state_dict": fold["state_dict"],
                    "feature_mean": fold["feature_mean"],
                    "feature_std": fold["feature_std"],
                    "held_out_phase_offset": fold["report"][
                        "held_out_phase_offset"
                    ],
                }
                for fold in folds
            ],
            "aggregation": "minimum_probability",
            "deployment_threshold": args.deployment_threshold,
            "score_deadband": args.score_deadband,
            "full_benefit": args.full_benefit,
            "teacher": "same_motion_phase_neighbor_upper_confidence",
            "neighbor_count": args.neighbor_count,
            "confidence_sem_multiplier": args.confidence_sem_multiplier,
            "support_action_ids": support_action_ids,
            "phase_offsets": [group["phase_offset"] for group in groups],
            "source_fingerprints": [group["fingerprint"] for group in groups],
            "ppo_updates": False,
            "uses_v3_checkpoint": False,
        },
        artifact_path,
    )
    summary = {
        "artifact": str(artifact_path.resolve()),
        "sample_count": int(all_features.shape[0]),
        "phase_offsets": [group["phase_offset"] for group in groups],
        "strict_pairing_all_groups": True,
        "failure_labeled_samples": failure_count,
        "failure_supervision_sufficient": failure_count >= 10,
        "deployment_aggregation": "minimum_probability",
        "deployment_threshold": args.deployment_threshold,
        "teacher": "same_motion_phase_neighbor_upper_confidence",
        "neighbor_count": args.neighbor_count,
        "confidence_sem_multiplier": args.confidence_sem_multiplier,
        "fold_reports": oof_reports,
        "conservative_ensemble_on_all_pairs": ensemble_report,
        "ppo_updates": False,
        "uses_v3_checkpoint": False,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    lines = [
        "# Paired State Gate V1",
        "",
        "Offline supervised gate; no PPO and no V3 checkpoint were used.",
        "",
        f"- Strictly paired samples: {all_features.shape[0]}",
        f"- Phase offsets: {[group['phase_offset'] for group in groups]}",
        f"- Failure-labeled samples: {failure_count}",
        f"- Failure supervision sufficient (>=10): {failure_count >= 10}",
        f"- Deployment threshold: {args.deployment_threshold}",
        f"- Conservative selected fraction: {ensemble_report['selected_fraction']:.4f}",
        f"- Conservative selected precision: {ensemble_report['selected_precision']:.4f}",
        f"- Gated policy score delta mean: {ensemble_report['gated_policy_score_delta_mean']:.6f}",
        f"- Always-on candidate score delta mean: {ensemble_report['candidate_always_on_score_delta_mean']:.6f}",
        f"- Selected candidate-caused failures: {ensemble_report['selected_candidate_caused_failures']}",
        "",
        "## Leave-one-phase-offset-out folds",
        "",
    ]
    for report in oof_reports:
        lines.append(
            "- offset={:.8f}: selected={:.3f}, precision={:.3f}, "
            "gated_delta={:.6f}, failure_harm={}".format(
                report["held_out_phase_offset"],
                report["selected_fraction"],
                report["selected_precision"],
                report["gated_policy_score_delta_mean"],
                report["selected_candidate_caused_failures"],
            )
        )
    (output_dir / "GATE_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
