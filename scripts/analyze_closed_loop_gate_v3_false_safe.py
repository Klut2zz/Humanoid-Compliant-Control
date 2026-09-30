"""Explain every false-safe selected by the rejected held-out Gate V3 test."""

# new change
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


# new change
ROOT = Path(__file__).absolute().parents[1]
DEFAULT_DECISIONS = ROOT / (
    "logs/stand_coordination_v1/stationary_support_control_v2/"
    "multi_target_closed_loop_gate_v3/final_test_decisions.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "logs/stand_coordination_v1/stationary_support_control_v2/"
    "multi_target_closed_loop_gate_v3/false_safe_analysis"
)
MOTIONS = (
    "neutral",
    "bilateral_sagittal",
    "bilateral_lateral",
    "unilateral_counterbalance",
    "axial_twist",
    "roll_coordination",
    "pitch_coordination",
    "combined_coordination",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
args = parser.parse_args()


# new change
def _mean(values, mask):
    return float(values[mask].mean()) if mask.any() else None


# new change
def main() -> None:
    decisions = torch.load(args.decisions, map_location="cpu", weights_only=False)
    selected = decisions["selected"].bool()
    failure_delta = decisions["failure_delta"]
    caused = selected & (failure_delta > 0)
    prevented = selected & (failure_delta < 0)
    safe = selected & (failure_delta <= 0)
    if int(caused.sum()) != 5:
        raise RuntimeError(f"Expected five V3 false-safe states, got {int(caused.sum())}.")

    counts = decisions["record_valid_counts"]
    boundaries = [0]
    for count in counts:
        boundaries.append(boundaries[-1] + count)
    cases = []
    for global_index in torch.where(caused)[0].tolist():
        record_index = next(
            index
            for index in range(len(counts))
            if boundaries[index] <= global_index < boundaries[index + 1]
        )
        record = decisions["records"][record_index]
        paired = torch.load(record["path"], map_location="cpu", weights_only=False)
        valid_env_ids = torch.where(paired["valid"].bool())[0]
        local_valid_index = global_index - boundaries[record_index]
        env_id = int(valid_env_ids[local_valid_index])
        features = decisions["features"][global_index]
        support = features[:87]
        force = features[87:102]
        motion_id = int(torch.argmax(features[102:110]))
        trend = support[79:87]
        correction = paired["candidate"][
            "candidate_correction_rad_trajectory"
        ][env_id, :25]
        fixed_foot = support[75:79].reshape(2, 2) / 10.0
        case = {
            "global_index": global_index,
            "event_seed": int(record["force_event_seed"]),
            "warmup_steps": int(record["warmup_steps"]),
            "env_id": env_id,
            "motion_id": motion_id,
            "motion_name": MOTIONS[motion_id],
            "risk_upper": float(decisions["risk_upper"][global_index]),
            "prevented_lower": float(decisions["prevented_lower"][global_index]),
            "short_lower": float(decisions["short_lower"][global_index]),
            "ood_distance": float(decisions["ood_distance"][global_index]),
            "baseline_failure": bool(
                decisions["baseline_failure"][global_index]
            ),
            "baseline_first_failure_step": int(
                paired["baseline"]["first_failure_step"][env_id]
            ),
            "candidate_first_failure_step": int(
                paired["candidate"]["first_failure_step"][env_id]
            ),
            "q_error_max_rad": float(support[20:30].abs().max()),
            "correction_mean_abs_mrad": float(correction.abs().mean() * 1000.0),
            "correction_max_abs_mrad": float(correction.abs().max() * 1000.0),
            "correction_saturation_fraction": float(
                (correction.abs() >= 0.005999).float().mean()
            ),
            "com_distance_m": float(trend[0] * 0.05),
            "com_outward_velocity_m_s": float(trend[1] * 0.20),
            "torso_gravity_xy_norm": float(trend[2] * 0.35),
            "torso_tilt_growth_proxy": float(trend[3]),
            "root_xy_speed_m_s": float(trend[4] * 0.30),
            "max_foot_xy_speed_m_s": float(trend[5] * 0.30),
            "minimum_support_load_n": float(trend[6] * 300.0),
            "support_load_imbalance_n": float(trend[7] * 300.0),
            "root_xy_error_m": support[60:62].tolist(),
            "fixed_foot_error_m": fixed_foot.tolist(),
            "sampled_force_n": float(force[0] * 7.0),
            "sampled_force_direction_w": force[1:4].tolist(),
            "force_wrist": "left" if force[4] > force[5] else "right",
            "signed_event_phase": float(force[6]),
            "force_envelope": float(force[9]),
            "ik_pending": bool(force[10] > 0.5),
            "ik_accepted": bool(force[11] > 0.5),
            "ik_rejected": bool(force[12] > 0.5),
            "force_active": bool(force[13] > 0.5),
            "ik_feasible_scale": float(force[14]),
            "com_improvement_m": decisions["com_improvement"][
                global_index
            ].tolist(),
            "tilt_improvement_rad": decisions["tilt_improvement"][
                global_index
            ].tolist(),
        }
        cases.append(case)

    summary = {
        "selected": int(selected.sum()),
        "selected_caused": int(caused.sum()),
        "selected_prevented": int(prevented.sum()),
        "selected_safe_or_prevented": int(safe.sum()),
        "selected_caused_motion_counts": {
            name: sum(case["motion_name"] == name for case in cases)
            for name in MOTIONS
            if any(case["motion_name"] == name for case in cases)
        },
        "selected_caused_force_active": sum(case["force_active"] for case in cases),
        "selected_caused_ik_accepted": sum(case["ik_accepted"] for case in cases),
        "score_comparison": {
            "risk_upper_caused_mean": _mean(decisions["risk_upper"], caused),
            "risk_upper_safe_mean": _mean(decisions["risk_upper"], safe),
            "ood_caused_mean": _mean(decisions["ood_distance"], caused),
            "ood_safe_mean": _mean(decisions["ood_distance"], safe),
            "short_lower_caused_mean": _mean(decisions["short_lower"], caused),
            "short_lower_safe_mean": _mean(decisions["short_lower"], safe),
        },
        "cases": cases,
        "conclusion": (
            "Gate V3 is rejected. These test labels may be used only as V4 "
            "development/hard-negative data; V4 requires fresh final test seeds."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "false_safe_report.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Gate V3 false-safe analysis",
        "",
        f"- Selected: {summary['selected']}",
        f"- Caused failures: {summary['selected_caused']}",
        f"- Prevented failures: {summary['selected_prevented']}",
        f"- Caused motion counts: {summary['selected_caused_motion_counts']}",
        f"- Force active among caused: {summary['selected_caused_force_active']}/5",
        f"- IK accepted among caused: {summary['selected_caused_ik_accepted']}/5",
        "",
        "| seed | warmup | motion | wrist | force N | risk upper | OOD | candidate failure step |",
        "|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case['event_seed']} | {case['warmup_steps']} | "
            f"{case['motion_name']} | {case['force_wrist']} | "
            f"{case['sampled_force_n']:.2f} | {case['risk_upper']:.3f} | "
            f"{case['ood_distance']:.3f} | {case['candidate_first_failure_step']} |"
        )
    lines.extend(
        (
            "",
            "V3 remains rejected; no threshold was retuned on this test set.",
        )
    )
    md_path = args.output_dir / "FALSE_SAFE_REPORT.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
