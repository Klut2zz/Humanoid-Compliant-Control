"""Train and strictly evaluate an offline dual-head force safety gate."""

# new change
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F


# new change
PROJECT_ROOT = Path(__file__).absolute().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--split_manifest",
    type=Path,
    default=PROJECT_ROOT
    / (
        "logs/stand_coordination_v1/stationary_support_control_v2/"
        "dual_head_force_gate_v1/split_manifest.json"
    ),
)
parser.add_argument(
    "--output_dir",
    type=Path,
    default=PROJECT_ROOT
    / (
        "logs/stand_coordination_v1/stationary_support_control_v2/"
        "dual_head_force_gate_v1"
    ),
)
parser.add_argument("--ensemble_size", type=int, default=5)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=3.0e-4)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()


# new change
class DualHeadSafetyGate(nn.Module):
    def __init__(self, input_dim=110):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
        )
        self.risk_head = nn.Linear(64, 1)
        self.benefit_head = nn.Linear(64, 1)

    def forward(self, features):
        latent = self.trunk(features)
        return (
            self.risk_head(latent).squeeze(-1),
            self.benefit_head(latent).squeeze(-1),
        )


# new change
def _load_records(records, device):
    fields = {
        key: []
        for key in (
            "features",
            "risk",
            "score",
            "failure_delta",
            "baseline_failure",
            "motion_id",
        )
    }
    for record in records:
        data = torch.load(
            record["path"], map_location="cpu", weights_only=False
        )
        if data["snapshot_fingerprint"] != record["snapshot_fingerprint"]:
            raise RuntimeError(f"Fingerprint changed: {record['path']}")
        valid = data["valid"].bool()
        features = data["force_conditioned_gate_input"][valid].float()
        motion_id = data["motion_id"][valid].long()
        motion_onehot = F.one_hot(motion_id, num_classes=8).float()
        failure_delta = data["paired_failure_delta"][valid].to(torch.int8)
        fields["features"].append(torch.cat((features, motion_onehot), dim=-1))
        fields["risk"].append((failure_delta > 0).float())
        fields["score"].append(data["paired_score_delta"][valid].float())
        fields["failure_delta"].append(failure_delta)
        fields["baseline_failure"].append(
            data["baseline"]["failure"][valid].bool()
        )
        fields["motion_id"].append(motion_id)
    return {
        key: torch.cat(values).to(device) for key, values in fields.items()
    }


# new change
def _loss(model, features, risk, clipped_score, pos_weight):
    risk_logit, benefit = model(features)
    risk_loss = F.binary_cross_entropy_with_logits(
        risk_logit, risk, pos_weight=pos_weight
    )
    benefit_loss = F.smooth_l1_loss(benefit, clipped_score, beta=0.05)
    return risk_loss + benefit_loss, risk_loss, benefit_loss


# new change
def _train_member(index, train, validation, mean, scale, pos_weight, device):
    seed = args.seed + 1009 * index
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = DualHeadSafetyGate().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    bootstrap = torch.randint(
        0,
        train["features"].shape[0],
        (train["features"].shape[0],),
        device=device,
        generator=generator,
    )
    normalized_train = (train["features"] - mean) / scale
    normalized_validation = (validation["features"] - mean) / scale
    train_target = train["score"].clamp(-0.5, 0.5)
    validation_target = validation["score"].clamp(-0.5, 0.5)
    best_state = None
    best_validation = float("inf")
    best_epoch = 0
    patience = 30
    stale = 0
    for epoch in range(args.epochs):
        order = bootstrap[
            torch.randperm(
                bootstrap.numel(), device=device, generator=generator
            )
        ]
        model.train()
        for start in range(0, order.numel(), args.batch_size):
            ids = order[start : start + args.batch_size]
            total, _, _ = _loss(
                model,
                normalized_train[ids],
                train["risk"][ids],
                train_target[ids],
                pos_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss, _, _ = _loss(
                model,
                normalized_validation,
                validation["risk"],
                validation_target,
                pos_weight,
            )
        value = float(validation_loss)
        if value < best_validation - 1.0e-5:
            best_validation = value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("No valid model state was produced.")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "member": index,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_loss": best_validation,
    }


# new change
def _ensemble_predict(models, features, mean, scale):
    normalized = (features - mean) / scale
    probabilities = []
    benefits = []
    with torch.inference_mode():
        for model in models:
            risk_logit, benefit = model(normalized)
            probabilities.append(torch.sigmoid(risk_logit))
            benefits.append(benefit)
    probability = torch.stack(probabilities)
    benefit = torch.stack(benefits)
    return {
        "risk_mean": probability.mean(dim=0),
        "risk_std": probability.std(dim=0, unbiased=False),
        "benefit_mean": benefit.mean(dim=0),
        "benefit_std": benefit.std(dim=0, unbiased=False),
    }


# new change
def _nearest_distance(query, reference, exclude_self=False):
    output = []
    divisor = math.sqrt(reference.shape[1])
    for start in range(0, query.shape[0], 256):
        end = min(start + 256, query.shape[0])
        distance = torch.cdist(query[start:end], reference) / divisor
        if exclude_self:
            rows = torch.arange(end - start, device=query.device)
            cols = torch.arange(start, end, device=query.device)
            distance[rows, cols] = float("inf")
        output.append(distance.min(dim=1).values)
    return torch.cat(output)


# new change
def _evaluate_selection(dataset, selected):
    failure_delta = dataset["failure_delta"]
    score = dataset["score"]
    beneficial = (failure_delta < 0) | (
        (failure_delta == 0) & (score < -0.005)
    )
    baseline_failures = int(dataset["baseline_failure"].sum())
    gated_failure_delta = int(failure_delta[selected].sum())
    selected_count = int(selected.sum())
    return {
        "samples": int(selected.numel()),
        "selected": selected_count,
        "selected_fraction": float(selected.float().mean()),
        "selected_beneficial": int((selected & beneficial).sum()),
        "selected_candidate_caused_failures": int(
            (selected & (failure_delta > 0)).sum()
        ),
        "selected_candidate_prevented_failures": int(
            (selected & (failure_delta < 0)).sum()
        ),
        "baseline_failures": baseline_failures,
        "gated_failures": baseline_failures + gated_failure_delta,
        "gated_score_delta_mean": float(
            torch.where(selected, score, torch.zeros_like(score)).mean()
        ),
    }


# new change
def _select_thresholds(validation, predictions, ood_distance, ood_threshold):
    candidates = []
    quantiles = torch.linspace(0.01, 0.75, 40, device=ood_distance.device)
    for uncertainty in (1.0, 2.0, 3.0):
        risk_upper = predictions["risk_mean"] + (
            uncertainty * predictions["risk_std"]
        )
        benefit_upper = predictions["benefit_mean"] + (
            uncertainty * predictions["benefit_std"]
        )
        risk_thresholds = torch.unique(
            torch.quantile(risk_upper, quantiles)
        ).tolist()
        for risk_threshold in risk_thresholds:
            for deadband in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10):
                for ood_scale in (0.75, 0.90, 1.0):
                    selected = (
                        (risk_upper <= risk_threshold)
                        & (benefit_upper < -deadband)
                        & (ood_distance <= ood_threshold * ood_scale)
                    )
                    metrics = _evaluate_selection(validation, selected)
                    if (
                        metrics["selected"] >= 16
                        and metrics["selected_candidate_caused_failures"] == 0
                        and metrics["gated_failures"]
                        <= metrics["baseline_failures"]
                        and metrics["gated_score_delta_mean"] <= 0.0
                    ):
                        candidates.append(
                            (
                                metrics["selected_candidate_prevented_failures"],
                                metrics["selected_beneficial"],
                                metrics["selected"],
                                -metrics["gated_score_delta_mean"],
                                {
                                    "uncertainty_multiplier": uncertainty,
                                    "risk_threshold": float(risk_threshold),
                                    "benefit_deadband": deadband,
                                    "ood_scale": ood_scale,
                                },
                                metrics,
                            )
                        )
    if not candidates:
        return None, None
    winner = max(candidates, key=lambda item: item[:4])
    return winner[4], winner[5]


# new change
def _apply_thresholds(predictions, distance, ood_threshold, thresholds):
    uncertainty = thresholds["uncertainty_multiplier"]
    risk_upper = predictions["risk_mean"] + uncertainty * predictions["risk_std"]
    benefit_upper = (
        predictions["benefit_mean"]
        + uncertainty * predictions["benefit_std"]
    )
    return (
        (risk_upper <= thresholds["risk_threshold"])
        & (benefit_upper < -thresholds["benefit_deadband"])
        & (distance <= ood_threshold * thresholds["ood_scale"])
    )


# new change
def main():
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    split = json.loads(args.split_manifest.read_text())
    if not split.get("split_locked", False):
        raise RuntimeError("Split manifest is not locked.")
    if split.get("ppo_updates", True) or split.get("uses_v3_checkpoint", True):
        raise RuntimeError("Disallowed split provenance.")

    # Test records are deliberately not loaded here.
    train = _load_records(split["records"]["train"], device)
    validation = _load_records(split["records"]["validation"], device)
    mean = train["features"].mean(dim=0)
    scale = train["features"].std(dim=0).clamp_min(0.01)
    normalized_train = (train["features"] - mean) / scale
    train_ood = _nearest_distance(
        normalized_train, normalized_train, exclude_self=True
    )
    ood_threshold = float(torch.quantile(train_ood, 0.995))
    validation_ood = _nearest_distance(
        (validation["features"] - mean) / scale, normalized_train
    )

    positives = train["risk"].sum()
    negatives = train["risk"].numel() - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(max=50.0)
    models = []
    member_reports = []
    for index in range(args.ensemble_size):
        model, report = _train_member(
            index, train, validation, mean, scale, pos_weight, device
        )
        models.append(model)
        member_reports.append(report)
        print(report, flush=True)

    validation_predictions = _ensemble_predict(
        models, validation["features"], mean, scale
    )
    thresholds, validation_metrics = _select_thresholds(
        validation,
        validation_predictions,
        validation_ood,
        ood_threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_lock_path = args.output_dir / "threshold_lock.json"
    threshold_lock = {
        "weights_frozen": True,
        "normalizer_frozen": True,
        "ood_threshold_train_only": ood_threshold,
        "thresholds_validation_only": thresholds,
        "validation_metrics": validation_metrics,
        "test_opened": False,
        "ppo_updates": False,
    }
    threshold_lock_path.write_text(json.dumps(threshold_lock, indent=2) + "\n")

    test_metrics = None
    accepted = False
    if thresholds is not None:
        # The threshold lock is on disk before the first test artifact load.
        test = _load_records(split["records"]["test"], device)
        test_predictions = _ensemble_predict(models, test["features"], mean, scale)
        test_ood = _nearest_distance(
            (test["features"] - mean) / scale, normalized_train
        )
        test_selected = _apply_thresholds(
            test_predictions, test_ood, ood_threshold, thresholds
        )
        test_metrics = _evaluate_selection(test, test_selected)
        accepted = bool(
            test_metrics["selected"] >= 10
            and test_metrics["selected_candidate_caused_failures"] == 0
            and test_metrics["gated_failures"] <= test_metrics["baseline_failures"]
            and test_metrics["gated_score_delta_mean"] <= 0.0
        )
        threshold_lock["test_opened"] = True
        threshold_lock["test_metrics"] = test_metrics
        threshold_lock_path.write_text(
            json.dumps(threshold_lock, indent=2) + "\n"
        )

    artifact_path = args.output_dir / "dual_head_force_safety_gate.pt"
    torch.save(
        {
            "model_class": "BootstrapEnsembleDualHeadForceSafetyGate",
            "architecture": {"input_dim": 110, "hidden_dims": (128, 64)},
            "ensemble_state_dicts": [
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
                for model in models
            ],
            "feature_mean": mean.detach().cpu(),
            "feature_scale": scale.detach().cpu(),
            "train_features_normalized": normalized_train.detach().cpu(),
            "ood_threshold": ood_threshold,
            "thresholds": thresholds,
            "split_manifest": str(args.split_manifest.resolve()),
            "source_snapshot_fingerprints": {
                split_name: [r["snapshot_fingerprint"] for r in records]
                for split_name, records in split["records"].items()
            },
            "accepted_for_frozen_inference": accepted,
            "test_metrics": test_metrics,
            "ppo_updates": False,
            "uses_v3_checkpoint": False,
        },
        artifact_path,
    )
    summary = {
        "artifact": str(artifact_path.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "device": str(device),
        "ensemble_size": args.ensemble_size,
        "train_samples": int(train["risk"].numel()),
        "validation_samples": int(validation["risk"].numel()),
        "train_candidate_caused_failures": int(train["risk"].sum()),
        "positive_class_weight": float(pos_weight),
        "member_reports": member_reports,
        "thresholds": thresholds,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "accepted_for_frozen_inference": accepted,
        "ppo_updates": False,
    }
    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
