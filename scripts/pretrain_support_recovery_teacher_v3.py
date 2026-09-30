"""Causal outcome filtering and offline initialization for support TeacherV3."""

# new change
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# new change
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", required=True)
parser.add_argument("--source_checkpoint", required=True)
parser.add_argument("--output_checkpoint", required=True)
parser.add_argument("--outcome_epochs", type=int, default=150)
parser.add_argument("--adapter_epochs", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--outcome_learning_rate", type=float, default=5.0e-4)
parser.add_argument("--adapter_learning_rate", type=float, default=1.0e-3)
parser.add_argument("--minimum_relative_benefit", type=float, default=0.01)
parser.add_argument("--full_gate_relative_benefit", type=float, default=0.06)
parser.add_argument("--maximum_predicted_degradation", type=float, default=0.05)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


# new change
SUPPORT_ACTION_IDS = (0, 1, 9, 10, 13, 14, 3, 4, 17, 18)
SUPPORT_LIMIT_RAD = 0.006


# new change
class PotentialOutcomeModel(nn.Module):
    """Shared state encoder with separate randomized control/treatment heads."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(87, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.control = nn.Linear(64, 1)
        self.treatment = nn.Linear(64, 1)

    def potential_outcomes(self, x: torch.Tensor):
        latent = self.backbone(x)
        # Future cost is non-negative. Softplus avoids an unphysical negative
        # treatment effect created only by output extrapolation.
        return (
            F.softplus(self.control(latent)).squeeze(-1),
            F.softplus(self.treatment(latent)).squeeze(-1),
        )


# new change
def make_support_adapter() -> nn.ModuleDict:
    def head(hidden: int, output: int) -> nn.Sequential:
        module = nn.Sequential(
            nn.Linear(87, hidden),
            nn.ELU(),
            nn.Linear(hidden, output),
        )
        nn.init.zeros_(module[-1].weight)
        nn.init.constant_(module[-1].bias, -4.0)
        return module

    return nn.ModuleDict(
        {
            "sagittal": head(64, 6),
            "lateral": head(48, 4),
        }
    )


# new change
def adapter_gate_logits(adapter: nn.ModuleDict, features: torch.Tensor):
    raw = torch.cat(
        (adapter["sagittal"](features), adapter["lateral"](features)),
        dim=-1,
    )
    return raw


# new change
def adapter_output_rad(adapter: nn.ModuleDict, features: torch.Tensor):
    gate = torch.sigmoid(adapter_gate_logits(adapter, features))
    candidate = torch.clamp(
        -0.10 * features[:, 20:30],
        min=-SUPPORT_LIMIT_RAD,
        max=SUPPORT_LIMIT_RAD,
    )
    return candidate * gate


# new change
def main() -> None:
    if args.full_gate_relative_benefit <= args.minimum_relative_benefit:
        raise ValueError("full gate threshold must exceed minimum benefit.")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    features = data["support_recovery_features"].float()
    candidate = data["candidate_target_rad"].float()
    treatment = data["treatment"].bool()
    outcome = torch.clamp(data["future_stability_cost"].float(), max=25.0)
    start_cost = data["start_stability_cost"].float()
    if features.ndim != 2 or features.shape[1] != 87:
        raise ValueError(f"Expected [N,87] recovery state, got {features.shape}.")
    if candidate.shape != (features.shape[0], 10):
        raise ValueError(f"Expected [N,10] candidates, got {candidate.shape}.")
    if tuple(data.get("support_action_ids", ())) != SUPPORT_ACTION_IDS:
        raise ValueError("Support action ordering mismatch.")
    if abs(float(data.get("teacher_gain", -1.0)) - 0.10) > 1.0e-8:
        raise ValueError("RecoveryAwareTeacherV3 requires gain=0.10 probes.")
    if abs(float(data.get("teacher_limit_rad", -1.0)) - 0.006) > 1.0e-8:
        raise ValueError("RecoveryAwareTeacherV3 requires limit=0.006 rad probes.")

    permutation = torch.randperm(features.shape[0])
    split = int(0.8 * features.shape[0])
    train_indices = permutation[:split]
    validation_indices = permutation[split:]
    feature_mean = features[train_indices].mean(dim=0)
    feature_std = features[train_indices].std(dim=0).clamp_min(1.0e-4)
    normalized = (features - feature_mean) / feature_std

    outcome_model = PotentialOutcomeModel().to(device)
    outcome_optimizer = torch.optim.AdamW(
        outcome_model.parameters(),
        lr=args.outcome_learning_rate,
        weight_decay=1.0e-5,
    )
    outcome_reports = []
    for epoch in range(args.outcome_epochs):
        order = train_indices[torch.randperm(train_indices.numel())]
        loss_sum = 0.0
        count = 0
        for start in range(0, order.numel(), args.batch_size):
            indices = order[start : start + args.batch_size]
            x = normalized[indices].to(device)
            y = outcome[indices].to(device)
            t = treatment[indices].to(device)
            y0, y1 = outcome_model.potential_outcomes(x)
            prediction = torch.where(t, y1, y0)
            loss = F.smooth_l1_loss(prediction, y, beta=0.5)
            outcome_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(outcome_model.parameters(), 2.0)
            outcome_optimizer.step()
            loss_sum += float(loss.item()) * indices.numel()
            count += indices.numel()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.outcome_epochs:
            with torch.inference_mode():
                vi = validation_indices.to(device)
                y0, y1 = outcome_model.potential_outcomes(normalized[vi.cpu()].to(device))
                observed = torch.where(treatment[vi.cpu()].to(device), y1, y0)
                error = observed - outcome[vi.cpu()].to(device)
                report = {
                    "epoch": epoch + 1,
                    "train_huber": loss_sum / max(count, 1),
                    "validation_mae": float(error.abs().mean()),
                    "validation_rmse": float(torch.sqrt(error.square().mean())),
                }
                outcome_reports.append(report)
                print({"outcome": report}, flush=True)

    with torch.inference_mode():
        all_y0 = []
        all_y1 = []
        for start in range(0, features.shape[0], args.batch_size):
            x = normalized[start : start + args.batch_size].to(device)
            y0, y1 = outcome_model.potential_outcomes(x)
            all_y0.append(y0.cpu())
            all_y1.append(y1.cpu())
        predicted_control = torch.cat(all_y0)
        predicted_treatment = torch.cat(all_y1)

    relative_benefit = (
        predicted_control - predicted_treatment
    ) / torch.clamp(predicted_control.abs(), min=0.5)
    benefit_gate = torch.clamp(
        (relative_benefit - args.minimum_relative_benefit)
        / (
            args.full_gate_relative_benefit
            - args.minimum_relative_benefit
        ),
        min=0.0,
        max=1.0,
    )
    degradation_gate = torch.clamp(
        (
            start_cost
            + args.maximum_predicted_degradation
            - predicted_treatment
        )
        / max(args.maximum_predicted_degradation, 1.0e-4),
        min=0.0,
        max=1.0,
    )
    gate = benefit_gate * degradation_gate
    supervised_target = candidate * gate[:, None]

    adapter = make_support_adapter().to(device)
    adapter_optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.adapter_learning_rate,
        weight_decay=1.0e-6,
    )
    adapter_reports = []
    for epoch in range(args.adapter_epochs):
        order = train_indices[torch.randperm(train_indices.numel())]
        loss_sum = 0.0
        count = 0
        for start in range(0, order.numel(), args.batch_size):
            indices = order[start : start + args.batch_size]
            x = features[indices].to(device)
            target_gate = gate[indices, None].expand(-1, 10).to(device)
            logits = adapter_gate_logits(adapter, x)
            # False positives remain costly while the sparse beneficial states
            # receive enough weight not to collapse to an all-zero gate.
            element_loss = F.binary_cross_entropy_with_logits(
                logits, target_gate, reduction="none"
            )
            element_weight = 1.0 + 4.0 * target_gate
            loss = (element_loss * element_weight).mean()
            adapter_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            adapter_optimizer.step()
            loss_sum += float(loss.item()) * indices.numel()
            count += indices.numel()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.adapter_epochs:
            with torch.inference_mode():
                vi = validation_indices.to(device)
                validation_features = features[vi.cpu()].to(device)
                prediction_gate = torch.sigmoid(
                    adapter_gate_logits(adapter, validation_features)
                )
                prediction = adapter_output_rad(adapter, validation_features)
                error = prediction - supervised_target[vi.cpu()].to(device)
                target_gate = gate[vi.cpu()].to(device)
                report = {
                    "epoch": epoch + 1,
                    "train_mse_rad": loss_sum / max(count, 1),
                    "validation_mae_mrad": float(error.abs().mean() * 1000.0),
                    "validation_rmse_mrad": float(
                        torch.sqrt(error.square().mean()) * 1000.0
                    ),
                    "validation_max_mrad": float(error.abs().max() * 1000.0),
                    "prediction_gate_mean": float(prediction_gate.mean()),
                    "zero_target_false_gate_fraction": float(
                        (
                            prediction_gate[target_gate <= 0.0]
                            > 0.10
                        ).float().mean()
                    ),
                }
                adapter_reports.append(report)
                print({"adapter": report}, flush=True)

    checkpoint = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    actor_state = checkpoint["actor_state_dict"]
    for key, value in copy.deepcopy(adapter).cpu().state_dict().items():
        actor_state[f"support_residual_adapter.{key}"] = value
    checkpoint["optimizer_state_dict"] = None
    checkpoint["infos"] = {
        **(checkpoint.get("infos") or {}),
        "support_recovery_aware_teacher_v3": True,
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "dataset": str(Path(args.dataset).resolve()),
        "support_action_ids": SUPPORT_ACTION_IDS,
        "support_target_limit_rad": SUPPORT_LIMIT_RAD,
        "sample_count": int(features.shape[0]),
        "gate_mean": float(gate.mean()),
        "gate_nonzero_fraction": float((gate > 0.0).float().mean()),
        "gate_full_fraction": float((gate >= 0.999).float().mean()),
        "target_mean_abs_mrad": float(supervised_target.abs().mean() * 1000.0),
        "outcome_reports": outcome_reports,
        "adapter_reports": adapter_reports,
    }
    output = Path(args.output_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)

    outcome_artifact = output.with_name("outcome_model.pt")
    torch.save(
        {
            "model_state_dict": copy.deepcopy(outcome_model).cpu().state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "minimum_relative_benefit": args.minimum_relative_benefit,
            "full_gate_relative_benefit": args.full_gate_relative_benefit,
            "maximum_predicted_degradation": args.maximum_predicted_degradation,
        },
        outcome_artifact,
    )
    summary = {
        "output_checkpoint": str(output),
        "outcome_artifact": str(outcome_artifact),
        "samples": int(features.shape[0]),
        "treatment_fraction": float(treatment.float().mean()),
        "gate_mean": float(gate.mean()),
        "gate_nonzero_fraction": float((gate > 0.0).float().mean()),
        "gate_full_fraction": float((gate >= 0.999).float().mean()),
        "target_mean_abs_mrad": float(supervised_target.abs().mean() * 1000.0),
        "observed_control_cost": float(outcome[~treatment].mean()),
        "observed_treatment_cost": float(outcome[treatment].mean()),
        "final_outcome": outcome_reports[-1],
        "final_adapter": adapter_reports[-1],
    }
    output.with_name("training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
