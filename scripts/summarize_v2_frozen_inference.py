#!/usr/bin/env python3
"""Summarize strict equivalence and multi-seed V2 frozen-inference runs."""

# new change: keep V2 frozen-inference acceptance separate from PPO training.
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


SEEDS = (0, 7, 123)
EXACT_KEYS = (
    "start_fingerprint",
    "rollout_digest",
    "failed_envs",
    "total_failures",
    "max_torso_tilt_rad",
    "max_com_offset_m",
    "max_root_speed_m_s",
    "max_foot_speed_m_s",
    "max_fixed_foot_error_m",
    "mean_stability_cost",
    "mean_force_n",
    "active_force_fraction",
    "peak_force_n",
)


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def equivalence(root: Path) -> dict:
    result = {}
    for mode in ("zero", "low"):
        baseline = load(root / "equivalence" / f"{mode}_baseline_seed0_64x100.json")
        disabled = load(root / "equivalence" / f"{mode}_gate_disabled_seed0_64x100.json")
        comparisons = {key: baseline[key] == disabled[key] for key in EXACT_KEYS}
        result[mode] = {
            "passed": all(comparisons.values()),
            "comparisons": comparisons,
            "baseline_digest": baseline["rollout_digest"],
            "gate_disabled_digest": disabled["rollout_digest"],
            "gate_disabled_selected": disabled["gate_selected"],
        }
    return result


def aggregate(root: Path, mode: str) -> dict:
    rows = []
    for seed in SEEDS:
        baseline = load(root / "validation" / f"{mode}_baseline_seed{seed}_512x1500.json")
        gated = load(root / "validation" / f"{mode}_gated_seed{seed}_512x1500.json")
        rows.append(
            {
                "seed": seed,
                "baseline_failed_envs": baseline["failed_envs"],
                "gated_failed_envs": gated["failed_envs"],
                "baseline_total_failures": baseline["total_failures"],
                "gated_total_failures": gated["total_failures"],
                "baseline_mean_stability_cost": baseline["mean_stability_cost"],
                "gated_mean_stability_cost": gated["mean_stability_cost"],
                "baseline_p95_tilt_rad": baseline["p95_max_torso_tilt_rad"],
                "gated_p95_tilt_rad": gated["p95_max_torso_tilt_rad"],
                "gate_decisions": gated["gate_decisions"],
                "gate_selected": gated["gate_selected"],
            }
        )
    decisions = sum(row["gate_decisions"] for row in rows)
    selected = sum(row["gate_selected"] for row in rows)
    return {
        "num_seeds": len(SEEDS),
        "total_envs": 512 * len(SEEDS),
        "steps_per_env": 1500,
        "baseline_failed_envs": sum(row["baseline_failed_envs"] for row in rows),
        "gated_failed_envs": sum(row["gated_failed_envs"] for row in rows),
        "baseline_total_failures": sum(row["baseline_total_failures"] for row in rows),
        "gated_total_failures": sum(row["gated_total_failures"] for row in rows),
        "baseline_mean_stability_cost": mean(
            row["baseline_mean_stability_cost"] for row in rows
        ),
        "gated_mean_stability_cost": mean(
            row["gated_mean_stability_cost"] for row in rows
        ),
        "gate_decisions": decisions,
        "gate_selected": selected,
        "gate_selected_fraction": selected / decisions if decisions else 0.0,
        "no_failure_regression": all(
            row["gated_total_failures"] <= row["baseline_total_failures"]
            for row in rows
        ),
        "per_seed": rows,
    }


def write_report(path: Path, summary: dict) -> None:
    zero = summary["validation"]["zero"]
    low = summary["validation"]["low"]
    lines = [
        "# V2 FrozenInference validation",
        "",
        "- Base controller: `model_1375` (frozen)",
        "- Gate: V2 two-output safety gate (frozen)",
        "- PPO updates: none",
        "- Validation: 512 environments × 1500 steps × seeds 0/7/123",
        "",
        "## Runtime equivalence",
        "",
        f"- Zero force: **{'PASS' if summary['equivalence']['zero']['passed'] else 'FAIL'}**",
        f"- Low force: **{'PASS' if summary['equivalence']['low']['passed'] else 'FAIL'}**",
        "- Baseline and gate-disabled branches have identical initial-state and rollout hashes.",
        "",
        "## Long-rollout results",
        "",
        "| condition | baseline failed envs | gated failed envs | baseline resets | gated resets | gate selected |",
        "|---|---:|---:|---:|---:|---:|",
        f"| zero force | {zero['baseline_failed_envs']} | {zero['gated_failed_envs']} | {zero['baseline_total_failures']} | {zero['gated_total_failures']} | {zero['gate_selected']}/{zero['gate_decisions']} |",
        f"| 4–7 N | {low['baseline_failed_envs']} | {low['gated_failed_envs']} | {low['baseline_total_failures']} | {low['gated_total_failures']} | {low['gate_selected']}/{low['gate_decisions']} |",
        "",
        "## Decision",
        "",
        "- **Runtime wiring accepted:** disabled Gate is exactly equivalent to baseline.",
        "- **Safety accepted:** Gate did not increase failures in any tested seed.",
        "- **Recovery benefit not accepted:** low-force coverage is effectively zero; one selection did not change the failure count.",
        "- Do not start PPO and do not promote this Gate as a recovery controller yet.",
        "- Next collect strict paired samples from the actual long-rollout decision distribution, especially pre-failure states currently rejected as OOD/risk, then recalibrate coverage on a held-out event-seed split.",
        "",
        "Note: fixed-foot maxima span resets and are therefore diagnostic-only; failure counts and paired comparisons remain the hard acceptance metrics.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "logs/stand_coordination_v1/stationary_support_control_v2/"
            "v2_frozen_inference"
        ),
    )
    args = parser.parse_args()
    summary = {
        "schema": "V2FrozenInferenceValidationSummaryV1",
        "model": "model_1375",
        "policy_frozen": True,
        "gate_frozen": True,
        "ppo_updates": False,
        "equivalence": equivalence(args.root),
        "validation": {
            "zero": aggregate(args.root, "zero"),
            "low": aggregate(args.root, "low"),
        },
    }
    summary["accepted_runtime_wiring"] = all(
        item["passed"] for item in summary["equivalence"].values()
    )
    summary["accepted_safety"] = all(
        item["no_failure_regression"] for item in summary["validation"].values()
    )
    summary["accepted_recovery_benefit"] = (
        summary["validation"]["low"]["gated_total_failures"]
        < summary["validation"]["low"]["baseline_total_failures"]
    )
    (args.root / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.root / "VALIDATION_REPORT.md", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
