"""Open the held-out V3 test once and issue an immutable acceptance decision."""

# new change
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


# new change
ROOT = Path(__file__).absolute().parents[1]
GATE_ROOT = ROOT / (
    "logs/stand_coordination_v1/stationary_support_control_v2/"
    "multi_target_closed_loop_gate_v3"
)
DEFAULT_GATE = GATE_ROOT / "multi_target_closed_loop_safety_gate_v3.pt"
DEFAULT_TEST = ROOT / (
    "logs/stand_coordination_v1/stationary_support_control_v2/"
    "force_conditioned_closed_loop_grid_v3_test/test_manifest.json"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--locked_gate", type=Path, default=DEFAULT_GATE)
parser.add_argument("--test_manifest", type=Path, default=DEFAULT_TEST)
parser.add_argument("--output_dir", type=Path, default=GATE_ROOT)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()

HORIZON_STEPS = (25, 50, 100)
HORIZON_NAMES = ("0p5s", "1p0s", "2p0s")


# new change
class Member(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(110, 160),
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
            "risk": torch.sigmoid(self.risk_head(latent).squeeze(-1)),
            "prevented": torch.sigmoid(
                self.prevented_failure_head(latent).squeeze(-1)
            ),
            "com": self.com_improvement_head(latent),
            "tilt": self.tilt_improvement_head(latent),
        }


# new change
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# new change
def _load_records(records: list[dict], device: torch.device) -> dict:
    fields = {
        key: []
        for key in (
            "features",
            "failure_delta",
            "baseline_failure",
            "com",
            "tilt",
            "continuous_mask",
        )
    }
    for record in records:
        path = Path(record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"Held-out test hash changed: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data["snapshot_fingerprint"] != record["snapshot_fingerprint"]:
            raise RuntimeError(f"Held-out fingerprint changed: {path}")
        valid = data["valid"].bool()
        features = data["force_conditioned_gate_input"][valid].float()
        motion = F.one_hot(data["motion_id"][valid].long(), num_classes=8).float()
        baseline = data["baseline"]
        candidate = data["candidate"]
        fields["features"].append(torch.cat((features, motion), dim=-1))
        fields["failure_delta"].append(
            data["paired_failure_delta"][valid].to(torch.int8)
        )
        fields["baseline_failure"].append(baseline["failure"][valid].bool())
        com = []
        tilt = []
        masks = []
        for steps in HORIZON_STEPS:
            com.append(
                baseline["com_support_xy_distance_m_trajectory"][:, :steps]
                .mean(1)[valid]
                - candidate["com_support_xy_distance_m_trajectory"][:, :steps]
                .mean(1)[valid]
            )
            tilt.append(
                baseline["torso_tilt_rad_trajectory"][:, :steps].mean(1)[valid]
                - candidate["torso_tilt_rad_trajectory"][:, :steps].mean(1)[valid]
            )
            masks.append(
                (
                    (baseline["first_failure_step"] > steps)
                    & (candidate["first_failure_step"] > steps)
                )[valid]
            )
        fields["com"].append(torch.stack(com, dim=-1))
        fields["tilt"].append(torch.stack(tilt, dim=-1))
        fields["continuous_mask"].append(torch.stack(masks, dim=-1))
    return {key: torch.cat(values).to(device) for key, values in fields.items()}


# new change
def _predict(models, normalized):
    collected = {key: [] for key in ("risk", "prevented", "com", "tilt")}
    with torch.inference_mode():
        for model in models:
            output = model(normalized)
            for key in collected:
                collected[key].append(output[key])
    result = {}
    for key, values in collected.items():
        stacked = torch.stack(values)
        result[f"{key}_mean"] = stacked.mean(0)
        result[f"{key}_std"] = stacked.std(0, unbiased=False)
    return result


# new change
def _nearest(query, train):
    values = []
    divisor = math.sqrt(query.shape[1])
    for start in range(0, query.shape[0], 128):
        distance = torch.cdist(query[start : start + 128], train)
        values.append(distance.min(1).values / divisor)
    return torch.cat(values)


# new change
def _select(prediction, distance, ood_threshold, thresholds, return_details=False):
    uncertainty = thresholds["uncertainty_multiplier"]
    risk_upper = prediction["risk_mean"] + uncertainty * prediction["risk_std"]
    prevented_lower = (
        prediction["prevented_mean"]
        - uncertainty * prediction["prevented_std"]
    )
    short_lower = torch.cat(
        (
            prediction["com_mean"] - uncertainty * prediction["com_std"],
            prediction["tilt_mean"] - uncertainty * prediction["tilt_std"],
        ),
        dim=-1,
    ).mean(-1)
    benefit = (
        prevented_lower >= thresholds["prevented_probability_threshold"]
    ) | (short_lower >= thresholds["short_improvement_threshold"])
    selected = (
        (risk_upper <= thresholds["risk_threshold"])
        & benefit
        & (distance <= ood_threshold * thresholds["ood_scale"])
    )
    if return_details:
        return {
            "selected": selected,
            "risk_upper": risk_upper,
            "prevented_lower": prevented_lower,
            "short_lower": short_lower,
            "ood_distance": distance,
            "benefit": benefit,
        }
    return selected


# new change
def _evaluate(dataset, selected, metric_scales):
    failure_delta = dataset["failure_delta"]
    baseline_failures = int(dataset["baseline_failure"].sum())
    mask = dataset["continuous_mask"] & selected[:, None]
    normalized_score = torch.cat(
        (
            dataset["com"] / metric_scales["com"],
            dataset["tilt"] / metric_scales["tilt"],
        ),
        dim=-1,
    ).mean(-1)
    output = {
        "samples": int(selected.numel()),
        "selected": int(selected.sum()),
        "selected_fraction": float(selected.float().mean()),
        "selected_candidate_caused_failures": int(
            (selected & (failure_delta > 0)).sum()
        ),
        "selected_candidate_prevented_failures": int(
            (selected & (failure_delta < 0)).sum()
        ),
        "selected_short_beneficial": int(
            (selected & dataset["continuous_mask"].all(-1) & (normalized_score > 0)).sum()
        ),
        "baseline_failures": baseline_failures,
        "gated_failures": baseline_failures + int(failure_delta[selected].sum()),
    }
    normalized_means = []
    for metric in ("com", "tilt"):
        for index, name in enumerate(HORIZON_NAMES):
            selected_mask = mask[:, index]
            value = dataset[metric][:, index]
            mean = float(value[selected_mask].mean()) if selected_mask.any() else float("nan")
            output[f"{metric}_improvement_{name}_mean"] = mean
            if selected_mask.any():
                normalized_means.append(
                    value[selected_mask].mean() / metric_scales[metric][index]
                )
    output["normalized_short_improvement_mean"] = (
        float(torch.stack(normalized_means).mean())
        if normalized_means
        else float("nan")
    )
    return output


# new change
def _accepted(metrics):
    return bool(
        metrics["selected"] >= 24
        and metrics["selected_candidate_caused_failures"] == 0
        and metrics["gated_failures"] <= metrics["baseline_failures"]
        and metrics["normalized_short_improvement_mean"] >= 0.0
        and metrics["com_improvement_1p0s_mean"] >= -2.0e-4
        and metrics["com_improvement_2p0s_mean"] >= -2.0e-4
        and metrics["tilt_improvement_1p0s_mean"] >= -5.0e-4
        and metrics["tilt_improvement_2p0s_mean"] >= -5.0e-4
    )


# new change
def main() -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    locked_sha256 = _sha256(args.locked_gate)
    artifact = torch.load(args.locked_gate, map_location="cpu", weights_only=False)
    test_manifest = json.loads(args.test_manifest.read_text())
    if test_manifest.get("schema") != "StrictClosedLoopForceEventSeedTestV3":
        raise RuntimeError("Unexpected held-out V3 test schema.")
    if not test_manifest.get("collection_complete", False):
        raise RuntimeError("Held-out V3 test collection is incomplete.")
    if test_manifest["locked_gate_sha256"] != locked_sha256:
        raise RuntimeError("Held-out test was collected against a different gate hash.")
    if len(test_manifest["records"]) != 6:
        raise RuntimeError("Expected exactly six V3 held-out groups.")
    if artifact.get("model_class") != "BootstrapEnsembleMultiTargetClosedLoopSafetyGateV3":
        raise RuntimeError("Unexpected locked V3 model class.")
    if not artifact.get("held_out_test_deferred", False):
        raise RuntimeError("The source artifact is not a pre-test lock.")
    models = []
    for state in artifact["ensemble_state_dicts"]:
        model = Member().to(device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    mean = artifact["feature_mean"].to(device)
    scale = artifact["feature_scale"].to(device)
    train = artifact["train_features_normalized"].to(device)
    metric_scales = {
        key: value.to(device) for key, value in artifact["metric_scales"].items()
    }
    dataset = _load_records(test_manifest["records"], device)
    normalized = (dataset["features"] - mean) / scale
    prediction = _predict(models, normalized)
    distance = _nearest(normalized, train)
    decision = _select(
        prediction,
        distance,
        float(artifact["ood_threshold"]),
        artifact["thresholds"],
        return_details=True,
    )
    selected = decision["selected"]
    metrics = _evaluate(dataset, selected, metric_scales)
    accepted = _accepted(metrics)
    per_event = {}
    for seed in (5501, 6607):
        records = [
            record
            for record in test_manifest["records"]
            if int(record["force_event_seed"]) == seed
        ]
        event_dataset = _load_records(records, device)
        event_normalized = (event_dataset["features"] - mean) / scale
        event_prediction = _predict(models, event_normalized)
        event_selected = _select(
            event_prediction,
            _nearest(event_normalized, train),
            float(artifact["ood_threshold"]),
            artifact["thresholds"],
        )
        per_event[str(seed)] = _evaluate(
            event_dataset, event_selected, metric_scales
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted_artifact = copy.deepcopy(artifact)
    accepted_artifact["accepted_for_frozen_inference"] = accepted
    accepted_artifact["held_out_test_deferred"] = False
    accepted_artifact["held_out_test_opened_once"] = True
    accepted_artifact["source_locked_artifact"] = str(args.locked_gate.resolve())
    accepted_artifact["source_locked_artifact_sha256"] = locked_sha256
    accepted_artifact["test_manifest"] = str(args.test_manifest.resolve())
    accepted_artifact["test_metrics"] = metrics
    accepted_artifact["test_metrics_by_event_seed"] = per_event
    output_artifact = args.output_dir / (
        "multi_target_closed_loop_safety_gate_v3_accepted.pt"
        if accepted
        else "multi_target_closed_loop_safety_gate_v3_rejected_test_sample.pt"
    )
    torch.save(accepted_artifact, output_artifact)
    # new change
    decision_output = args.output_dir / "final_test_decisions.pt"
    torch.save(
        {
            "source_locked_artifact_sha256": locked_sha256,
            "test_manifest": str(args.test_manifest.resolve()),
            "thresholds": artifact["thresholds"],
            "selected": selected.detach().cpu(),
            "risk_upper": decision["risk_upper"].detach().cpu(),
            "prevented_lower": decision["prevented_lower"].detach().cpu(),
            "short_lower": decision["short_lower"].detach().cpu(),
            "ood_distance": decision["ood_distance"].detach().cpu(),
            "benefit": decision["benefit"].detach().cpu(),
            "failure_delta": dataset["failure_delta"].detach().cpu(),
            "baseline_failure": dataset["baseline_failure"].detach().cpu(),
            "features": dataset["features"].detach().cpu(),
            "com_improvement": dataset["com"].detach().cpu(),
            "tilt_improvement": dataset["tilt"].detach().cpu(),
            "continuous_mask": dataset["continuous_mask"].detach().cpu(),
            "record_valid_counts": [
                int(record["valid_samples"])
                for record in test_manifest["records"]
            ],
            "records": test_manifest["records"],
        },
        decision_output,
    )
    report = {
        "schema": "ClosedLoopSafetyGateV3FinalAcceptance",
        "source_locked_artifact": str(args.locked_gate.resolve()),
        "source_locked_artifact_sha256": locked_sha256,
        "accepted_artifact": str(output_artifact.resolve()),
        "decision_artifact": str(decision_output.resolve()),
        "test_manifest": str(args.test_manifest.resolve()),
        "thresholds_unchanged": artifact["thresholds"],
        "test_metrics": metrics,
        "test_metrics_by_event_seed": per_event,
        "accepted_for_frozen_inference": accepted,
        "ppo_updates": False,
    }
    report_path = args.output_dir / "final_test_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
