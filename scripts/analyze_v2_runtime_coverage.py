#!/usr/bin/env python3
"""Analyze why the V2 gate rejects actual long-rollout decision states."""

# new change
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


FEATURE_GROUPS = (
    ("joint_q", 10),
    ("reference_q", 10),
    ("q_minus_reference", 10),
    ("joint_dq", 10),
    ("reference_dq", 10),
    ("phase_sincos", 2),
    ("com_support_offset", 2),
    ("torso_projected_gravity", 3),
    ("base_ang_vel", 3),
    ("root_xy_position_error", 2),
    ("base_lin_vel", 3),
    ("foot_velocities", 6),
    ("resolved_support_forces", 2),
    ("resolved_support_contacts", 2),
    ("fixed_foot_xy_error", 4),
    ("failure_trend", 8),
    ("force_context", 15),
    ("motion_onehot", 8),
)


def count(mask: torch.Tensor) -> int:
    return int(mask.sum())


def q(values: torch.Tensor) -> dict:
    return {
        name: float(torch.quantile(values.float(), value))
        for name, value in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99), ("max", 1.0))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    data = torch.load(args.trace, map_location="cpu", weights_only=False)
    if data.get("schema") != "V2FrozenInferenceDecisionTraceV1":
        raise ValueError("Unexpected decision trace schema.")
    if data.get("branch") not in ("gate_disabled", "gated"):
        raise ValueError("Coverage diagnosis requires Gate predictions.")
    if data.get("ppo_updates", True):
        raise ValueError("Trace provenance unexpectedly contains PPO updates.")

    thresholds = data["thresholds"]
    risk_veto = data["risk_upper"] > thresholds["risk_threshold"]
    benefit_veto = ~data["benefit"].bool()
    ood_veto = data["ood_distance"] > data["ood_threshold_runtime"]
    selected = data["selected"].bool()
    future = data["future_failure"]
    fail5 = future["5p0s"].bool()
    active_force = data["features"][:, 100] > 0.5
    accepted_ik = data["features"][:, 98] > 0.5

    samples = data["features"].shape[0]
    veto = {
        "samples": samples,
        "selected": count(selected),
        "risk_veto": count(risk_veto),
        "benefit_veto": count(benefit_veto),
        "ood_veto": count(ood_veto),
        "all_three_veto": count(risk_veto & benefit_veto & ood_veto),
        "risk_only": count(risk_veto & ~benefit_veto & ~ood_veto),
        "benefit_only": count(~risk_veto & benefit_veto & ~ood_veto),
        "ood_only": count(~risk_veto & ~benefit_veto & ood_veto),
        "active_force_samples": count(active_force),
        "accepted_ik_samples": count(accepted_ik),
    }
    future_report = {}
    for name, labels in future.items():
        labels = labels.bool()
        future_report[name] = {
            "future_failures": count(labels),
            "selected": count(labels & selected),
            "risk_veto": count(labels & risk_veto),
            "benefit_veto": count(labels & benefit_veto),
            "ood_veto": count(labels & ood_veto),
            "active_force": count(labels & active_force),
        }

    per_step = []
    for step in torch.unique(data["step_index"]).tolist():
        mask = data["step_index"] == step
        per_step.append(
            {
                "step": int(step),
                "samples": count(mask),
                "future_failure_5s": count(mask & fail5),
                "risk_veto": count(mask & risk_veto),
                "benefit_veto": count(mask & benefit_veto),
                "ood_veto": count(mask & ood_veto),
                "selected": count(mask & selected),
                "active_force": count(mask & active_force),
            }
        )

    normalized = data["normalized_features"].float()
    group_shift = []
    start = 0
    for name, width in FEATURE_GROUPS:
        end = start + width
        values = normalized[:, start:end].abs()
        group_shift.append(
            {
                "name": name,
                "start": start,
                "end": end,
                "mean_abs_z": float(values.mean()),
                "p95_abs_z": float(torch.quantile(values, 0.95)),
                "fraction_abs_z_gt_3": float((values > 3.0).float().mean()),
            }
        )
        start = end
    if start != 110:
        raise RuntimeError(f"Feature layout totals {start}, expected 110.")
    group_shift.sort(key=lambda row: row["mean_abs_z"], reverse=True)

    per_dimension = normalized.abs().mean(0)
    top_dimensions = []
    for index in torch.argsort(per_dimension, descending=True)[:20].tolist():
        group = next(
            name
            for name, begin, end in (
                (row["name"], row["start"], row["end"])
                for row in group_shift
            )
            if begin <= index < end
        )
        top_dimensions.append(
            {
                "index": index,
                "group": group,
                "mean_abs_z": float(per_dimension[index]),
                "p95_abs_z": float(torch.quantile(normalized[:, index].abs(), 0.95)),
            }
        )

    report = {
        "schema": "V2RuntimeCoverageDiagnosisV1",
        "trace": str(args.trace.resolve()),
        "policy_frozen": data["policy_frozen"],
        "gate_frozen": data["gate_frozen"],
        "ppo_updates": data["ppo_updates"],
        "veto": veto,
        "future_failure": future_report,
        "per_decision_step": per_step,
        "prediction_quantiles": {
            "risk_upper": q(data["risk_upper"]),
            "prevented_lower": q(data["prevented_lower"]),
            "short_lower": q(data["short_lower"]),
            "ood_distance": q(data["ood_distance"]),
            "runtime_ood_threshold": float(data["ood_threshold_runtime"]),
        },
        "feature_group_shift": group_shift,
        "top_shifted_dimensions": top_dimensions,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "coverage_diagnosis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# V2 runtime coverage diagnosis",
        "",
        f"- Samples: {samples}",
        f"- Selected: {veto['selected']}",
        f"- Risk veto: {veto['risk_veto']}",
        f"- Benefit veto: {veto['benefit_veto']}",
        f"- OOD veto: {veto['ood_veto']}",
        f"- Future failure within 5 s: {future_report['5p0s']['future_failures']}",
        "",
        "## Failure-preceding coverage",
        "",
        f"Of the {future_report['5p0s']['future_failures']} decision states followed by a failure within 5 s, "
        f"risk veto rejected {future_report['5p0s']['risk_veto']}, benefit veto rejected {future_report['5p0s']['benefit_veto']}, "
        f"and OOD veto rejected {future_report['5p0s']['ood_veto']}.",
        "",
        "## Largest feature-group shifts",
        "",
        "| group | mean |z| | p95 |z| | fraction |z|>3 |",
        "|---|---:|---:|---:|",
    ]
    for row in group_shift[:10]:
        lines.append(
            f"| {row['name']} | {row['mean_abs_z']:.3f} | {row['p95_abs_z']:.3f} | {row['fraction_abs_z_gt_3']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Collection decision",
            "",
            "Collect strict paired data at the actual decision schedule (warmup 100, 350, 600, 850, 1100, 1350 steps), prioritizing states followed by failure within 5 s and the shifted feature groups above. Keep event-seed splits disjoint. Do not relax the current Gate thresholds before the new validation split is locked.",
        ]
    )
    (args.output_dir / "COVERAGE_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
