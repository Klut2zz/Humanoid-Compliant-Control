"""Train V2 safety gate with decomposed short-horizon and recovery targets."""

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
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "logs/stand_coordination_v1/stationary_support_control_v2/"
    "multi_target_force_gate_v2"
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--split_manifest",
    type=Path,
    default=DEFAULT_OUTPUT / "split_manifest.json",
)
parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--ensemble_size", type=int, default=5)
parser.add_argument("--epochs", type=int, default=160)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=3.0e-4)
parser.add_argument("--seed", type=int, default=2027)
parser.add_argument("--device", default="cuda:0")
# new change
parser.add_argument(
    "--defer_test",
    action="store_true",
    help=(
        "Freeze weights/normalizers/OOD/validation thresholds without opening "
        "a held-out test split."
    ),
)
# new change
parser.add_argument(
    "--artifact_name",
    default="multi_target_force_safety_gate_v2.pt",
)
# new change
parser.add_argument(
    "--artifact_model_class",
    default="BootstrapEnsembleMultiTargetForceSafetyGateV2",
)
args = parser.parse_args()

HORIZON_STEPS = (25, 50, 100)
HORIZON_NAMES = ("0p5s", "1p0s", "2p0s")


# new change
class MultiTargetSafetyGateV2(nn.Module):
    def __init__(self, input_dim=110):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 160),
            nn.SiLU(),
            nn.LayerNorm(160),
            nn.Linear(160, 96),
            nn.SiLU(),
            nn.LayerNorm(96),
        )
        self.risk_head = nn.Linear(96, 1)
        self.prevented_failure_head = nn.Linear(96, 1)
        self.com_improvement_head = nn.Linear(96, 3)
        self.tilt_improvement_head = nn.Linear(96, 3)

    def forward(self, features):
        latent = self.trunk(features)
        return {
            "risk_logit": self.risk_head(latent).squeeze(-1),
            "prevented_logit": self.prevented_failure_head(latent).squeeze(-1),
            "com": self.com_improvement_head(latent),
            "tilt": self.tilt_improvement_head(latent),
        }


# new change
def _window_improvements(data, valid):
    baseline = data["baseline"]
    candidate = data["candidate"]
    com = []
    tilt = []
    mask = []
    for steps in HORIZON_STEPS:
        com.append(
            baseline["com_support_xy_distance_m_trajectory"][:, :steps].mean(1)
            - candidate["com_support_xy_distance_m_trajectory"][:, :steps].mean(1)
        )
        tilt.append(
            baseline["torso_tilt_rad_trajectory"][:, :steps].mean(1)
            - candidate["torso_tilt_rad_trajectory"][:, :steps].mean(1)
        )
        mask.append(
            (baseline["first_failure_step"] > steps)
            & (candidate["first_failure_step"] > steps)
        )
    return (
        torch.stack(com, dim=-1)[valid].float(),
        torch.stack(tilt, dim=-1)[valid].float(),
        torch.stack(mask, dim=-1)[valid].bool(),
    )


# new change
def _load_records(records, device):
    fields = {
        key: []
        for key in (
            "features",
            "risk",
            "prevented",
            "failure_delta",
            "baseline_failure",
            "com",
            "tilt",
            "continuous_mask",
        )
    }
    for record in records:
        data = torch.load(record["path"], map_location="cpu", weights_only=False)
        if data["snapshot_fingerprint"] != record["snapshot_fingerprint"]:
            raise RuntimeError(f"Fingerprint changed: {record['path']}")
        valid = data["valid"].bool()
        features = data["force_conditioned_gate_input"][valid].float()
        motion = F.one_hot(data["motion_id"][valid].long(), num_classes=8).float()
        failure_delta = data["paired_failure_delta"][valid].to(torch.int8)
        com, tilt, continuous_mask = _window_improvements(data, valid)
        fields["features"].append(torch.cat((features, motion), dim=-1))
        fields["risk"].append((failure_delta > 0).float())
        fields["prevented"].append((failure_delta < 0).float())
        fields["failure_delta"].append(failure_delta)
        fields["baseline_failure"].append(data["baseline"]["failure"][valid].bool())
        fields["com"].append(com)
        fields["tilt"].append(tilt)
        fields["continuous_mask"].append(continuous_mask)
    return {key: torch.cat(values).to(device) for key, values in fields.items()}


# new change
def _masked_smooth_l1(prediction, target, mask):
    loss = F.smooth_l1_loss(prediction, target, beta=0.20, reduction="none")
    weights = mask.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


# new change
def _loss(model, features, dataset, ids, risk_weight, prevented_weight, scales):
    output = model(features)
    risk_loss = F.binary_cross_entropy_with_logits(
        output["risk_logit"], dataset["risk"][ids], pos_weight=risk_weight
    )
    prevented_loss = F.binary_cross_entropy_with_logits(
        output["prevented_logit"],
        dataset["prevented"][ids],
        pos_weight=prevented_weight,
    )
    mask = dataset["continuous_mask"][ids]
    com_target = (dataset["com"][ids] / scales["com"]).clamp(-5.0, 5.0)
    tilt_target = (dataset["tilt"][ids] / scales["tilt"]).clamp(-5.0, 5.0)
    com_loss = _masked_smooth_l1(output["com"], com_target, mask)
    tilt_loss = _masked_smooth_l1(output["tilt"], tilt_target, mask)
    total = risk_loss + 0.75 * prevented_loss + 0.50 * com_loss + 0.50 * tilt_loss
    return total


# new change
def _train_member(index, train, validation, mean, scale, class_weights, metric_scales, device):
    seed = args.seed + 1009 * index
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = MultiTargetSafetyGateV2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed)
    bootstrap = torch.randint(
        train["features"].shape[0],
        (train["features"].shape[0],),
        generator=generator,
        device=device,
    )
    normalized_train = (train["features"] - mean) / scale
    normalized_validation = (validation["features"] - mean) / scale
    validation_ids = torch.arange(validation["features"].shape[0], device=device)
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(args.epochs):
        order = bootstrap[torch.randperm(bootstrap.numel(), generator=generator, device=device)]
        model.train()
        for start in range(0, order.numel(), args.batch_size):
            ids = order[start : start + args.batch_size]
            total = _loss(
                model,
                normalized_train[ids],
                train,
                ids,
                class_weights["risk"],
                class_weights["prevented"],
                metric_scales,
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            value = float(
                _loss(
                    model,
                    normalized_validation,
                    validation,
                    validation_ids,
                    class_weights["risk"],
                    class_weights["prevented"],
                    metric_scales,
                )
            )
        if value < best_loss - 1e-5:
            best_loss = value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 25:
                break
    if best_state is None:
        raise RuntimeError("No valid V2 model state was produced.")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "member": index,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
    }


# new change
def _ensemble_predict(models, features, mean, scale):
    normalized = (features - mean) / scale
    collected = {key: [] for key in ("risk", "prevented", "com", "tilt")}
    with torch.inference_mode():
        for model in models:
            output = model(normalized)
            collected["risk"].append(torch.sigmoid(output["risk_logit"]))
            collected["prevented"].append(torch.sigmoid(output["prevented_logit"]))
            collected["com"].append(output["com"])
            collected["tilt"].append(output["tilt"])
    result = {}
    for name, values in collected.items():
        stacked = torch.stack(values)
        result[f"{name}_mean"] = stacked.mean(0)
        result[f"{name}_std"] = stacked.std(0, unbiased=False)
    return result


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
        output.append(distance.min(1).values)
    return torch.cat(output)


# new change
def _evaluate(dataset, selected, metric_scales):
    failure_delta = dataset["failure_delta"]
    baseline_failures = int(dataset["baseline_failure"].sum())
    selected_count = int(selected.sum())
    mask = dataset["continuous_mask"] & selected[:, None]
    normalized = torch.cat(
        (dataset["com"] / metric_scales["com"], dataset["tilt"] / metric_scales["tilt"]),
        dim=-1,
    )
    all_windows = dataset["continuous_mask"].all(dim=-1)
    actual_short_score = normalized.mean(dim=-1)
    metrics = {
        "samples": int(selected.numel()),
        "selected": selected_count,
        "selected_fraction": float(selected.float().mean()),
        "selected_candidate_caused_failures": int((selected & (failure_delta > 0)).sum()),
        "selected_candidate_prevented_failures": int((selected & (failure_delta < 0)).sum()),
        "selected_short_beneficial": int((selected & all_windows & (actual_short_score > 0)).sum()),
        "baseline_failures": baseline_failures,
        "gated_failures": baseline_failures + int(failure_delta[selected].sum()),
    }
    aggregate_values = []
    for metric_name in ("com", "tilt"):
        for index, horizon_name in enumerate(HORIZON_NAMES):
            values = dataset[metric_name][:, index]
            selected_mask = mask[:, index]
            mean = float(values[selected_mask].mean()) if selected_mask.any() else float("nan")
            metrics[f"{metric_name}_improvement_{horizon_name}_mean"] = mean
            if selected_mask.any():
                aggregate_values.append(
                    values[selected_mask].mean() / metric_scales[metric_name][index]
                )
    metrics["normalized_short_improvement_mean"] = (
        float(torch.stack(aggregate_values).mean()) if aggregate_values else float("nan")
    )
    return metrics


# new change
def _selection(prediction, distance, ood_threshold, thresholds):
    uncertainty = thresholds["uncertainty_multiplier"]
    risk_upper = prediction["risk_mean"] + uncertainty * prediction["risk_std"]
    prevented_lower = prediction["prevented_mean"] - uncertainty * prediction["prevented_std"]
    short_lower = torch.cat(
        (
            prediction["com_mean"] - uncertainty * prediction["com_std"],
            prediction["tilt_mean"] - uncertainty * prediction["tilt_std"],
        ),
        dim=-1,
    ).mean(dim=-1)
    benefit = (prevented_lower >= thresholds["prevented_probability_threshold"]) | (
        short_lower >= thresholds["short_improvement_threshold"]
    )
    return (
        (risk_upper <= thresholds["risk_threshold"])
        & benefit
        & (distance <= ood_threshold * thresholds["ood_scale"])
    )


# new change
def _validation_safe(metrics):
    return bool(
        metrics["selected"] >= 32
        and metrics["selected_candidate_caused_failures"] == 0
        and metrics["gated_failures"] <= metrics["baseline_failures"]
        and metrics["normalized_short_improvement_mean"] >= 0.0
        and metrics["com_improvement_1p0s_mean"] >= -2e-4
        and metrics["com_improvement_2p0s_mean"] >= -2e-4
        and metrics["tilt_improvement_1p0s_mean"] >= -5e-4
        and metrics["tilt_improvement_2p0s_mean"] >= -5e-4
    )


# new change
def _select_thresholds(validation, prediction, distance, ood_threshold, metric_scales):
    candidates = []
    quantiles = torch.linspace(0.01, 0.65, 40, device=distance.device)
    for uncertainty in (1.0, 2.0, 3.0):
        risk_upper = prediction["risk_mean"] + uncertainty * prediction["risk_std"]
        for risk_threshold in torch.unique(torch.quantile(risk_upper, quantiles)).tolist():
            for prevented_threshold in (0.35, 0.50, 0.65, 0.80):
                for short_threshold in (-0.10, 0.0, 0.10, 0.25, 0.50):
                    for ood_scale in (0.75, 0.90, 1.0):
                        thresholds = {
                            "uncertainty_multiplier": uncertainty,
                            "risk_threshold": float(risk_threshold),
                            "prevented_probability_threshold": prevented_threshold,
                            "short_improvement_threshold": short_threshold,
                            "ood_scale": ood_scale,
                        }
                        selected = _selection(prediction, distance, ood_threshold, thresholds)
                        metrics = _evaluate(validation, selected, metric_scales)
                        if _validation_safe(metrics):
                            candidates.append(
                                (
                                    metrics["selected_candidate_prevented_failures"],
                                    metrics["selected_short_beneficial"],
                                    metrics["selected"],
                                    metrics["normalized_short_improvement_mean"],
                                    thresholds,
                                    metrics,
                                )
                            )
    if not candidates:
        return None, None
    winner = max(candidates, key=lambda item: item[:4])
    return winner[4], winner[5]


# new change
def main():
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    split = json.loads(args.split_manifest.read_text())
    if not split.get("split_locked", False):
        raise RuntimeError("V2 split is not locked.")
    if split.get("ppo_updates", True) or split.get("uses_v3_checkpoint", True):
        raise RuntimeError("Disallowed V2 provenance.")

    # Held-out test records are deliberately not loaded before threshold_lock.
    train = _load_records(split["records"]["train"], device)
    validation = _load_records(split["records"]["validation"], device)
    mean = train["features"].mean(0)
    scale = train["features"].std(0).clamp_min(0.01)
    metric_scales = {}
    for metric_name, floor in (("com", 2e-4), ("tilt", 5e-4)):
        values = train[metric_name]
        mask = train["continuous_mask"]
        scales = []
        for index in range(3):
            scales.append(values[:, index][mask[:, index]].std().clamp_min(floor))
        metric_scales[metric_name] = torch.stack(scales)

    normalized_train = (train["features"] - mean) / scale
    train_ood = _nearest_distance(normalized_train, normalized_train, exclude_self=True)
    ood_threshold = float(torch.quantile(train_ood, 0.995))
    validation_ood = _nearest_distance((validation["features"] - mean) / scale, normalized_train)

    def positive_weight(target):
        positives = target.sum()
        return ((target.numel() - positives) / positives.clamp_min(1.0)).clamp(max=50.0)

    class_weights = {
        "risk": positive_weight(train["risk"]),
        "prevented": positive_weight(train["prevented"]),
    }
    models = []
    member_reports = []
    for index in range(args.ensemble_size):
        model, report = _train_member(
            index, train, validation, mean, scale, class_weights, metric_scales, device
        )
        models.append(model)
        member_reports.append(report)
        print(report, flush=True)

    validation_prediction = _ensemble_predict(models, validation["features"], mean, scale)
    thresholds, validation_metrics = _select_thresholds(
        validation, validation_prediction, validation_ood, ood_threshold, metric_scales
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_lock_path = args.output_dir / "threshold_lock.json"
    lock = {
        "schema": "MultiTargetSafetyGateV2ThresholdLock",
        "weights_frozen": True,
        "normalizers_frozen": True,
        "metric_scales_frozen": True,
        "ood_threshold_train_only": ood_threshold,
        "thresholds_validation_only": thresholds,
        "validation_metrics": validation_metrics,
        "test_opened": False,
        "ppo_updates": False,
    }
    threshold_lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    test_metrics = None
    accepted = False
    if thresholds is not None and not args.defer_test:
        test = _load_records(split["records"]["test"], device)
        test_prediction = _ensemble_predict(models, test["features"], mean, scale)
        test_ood = _nearest_distance((test["features"] - mean) / scale, normalized_train)
        selected = _selection(test_prediction, test_ood, ood_threshold, thresholds)
        test_metrics = _evaluate(test, selected, metric_scales)
        accepted = bool(
            test_metrics["selected"] >= 24
            and test_metrics["selected_candidate_caused_failures"] == 0
            and test_metrics["gated_failures"] <= test_metrics["baseline_failures"]
            and test_metrics["normalized_short_improvement_mean"] >= 0.0
            and test_metrics["com_improvement_1p0s_mean"] >= -2e-4
            and test_metrics["com_improvement_2p0s_mean"] >= -2e-4
            and test_metrics["tilt_improvement_1p0s_mean"] >= -5e-4
            and test_metrics["tilt_improvement_2p0s_mean"] >= -5e-4
        )
        lock["test_opened"] = True
        lock["test_metrics"] = test_metrics
        threshold_lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    artifact_path = args.output_dir / args.artifact_name
    artifact = {
        "model_class": args.artifact_model_class,
        "architecture": {"input_dim": 110, "hidden_dims": (160, 96)},
        "ensemble_state_dicts": [
            {key: value.detach().cpu() for key, value in model.state_dict().items()}
            for model in models
        ],
        "feature_mean": mean.detach().cpu(),
        "feature_scale": scale.detach().cpu(),
        "metric_scales": {key: value.detach().cpu() for key, value in metric_scales.items()},
        "train_features_normalized": normalized_train.detach().cpu(),
        "ood_threshold": ood_threshold,
        "thresholds": thresholds,
        "split_manifest": str(args.split_manifest.resolve()),
        "accepted_for_frozen_inference": accepted,
        # new change
        "held_out_test_deferred": bool(args.defer_test),
        "candidate_semantics": split.get("candidate"),
        "fresh_test_event_seeds_reserved_uncollected": split.get(
            "fresh_test_event_seeds_reserved_uncollected", ()
        ),
        "test_metrics": test_metrics,
        "ppo_updates": False,
        "uses_v3_checkpoint": False,
    }
    torch.save(artifact, artifact_path)
    summary = {
        "artifact": str(artifact_path.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "device": str(device),
        "ensemble_size": args.ensemble_size,
        "train_samples": int(train["risk"].numel()),
        "validation_samples": int(validation["risk"].numel()),
        "train_caused_failures": int(train["risk"].sum()),
        "train_prevented_failures": int(train["prevented"].sum()),
        "class_weights": {key: float(value) for key, value in class_weights.items()},
        "metric_scales": {key: value.tolist() for key, value in metric_scales.items()},
        "member_reports": member_reports,
        "thresholds": thresholds,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "accepted_for_frozen_inference": accepted,
        # new change
        "held_out_test_deferred": bool(args.defer_test),
        "ppo_updates": False,
    }
    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
