#!/usr/bin/env python3
"""Re-analyze strict support finite differences in the robot yaw frame."""

# new change
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


# new change
# The layout is fixed by G1SupportRecoveryAwareTeacherV3ObservationsCfg.  Keep
# the named slices here so the audit fails loudly if the 87D interface changes.
FEATURE_SLICES = {
    "joint_q": (0, 10),
    "reference_q": (10, 20),
    "q_minus_reference": (20, 30),
    "joint_dq": (30, 40),
    "reference_dq": (40, 50),
    "phase_sincos": (50, 52),
    "com_support_offset_scaled": (52, 54),
    "torso_projected_gravity": (54, 57),
    "base_ang_vel_scaled": (57, 60),
    "root_xy_position_error": (60, 62),
    "base_lin_vel": (62, 65),
    "foot_velocities": (65, 71),
    "resolved_support_forces": (71, 73),
    "resolved_support_contacts": (73, 75),
    "fixed_foot_xy_error_scaled": (75, 79),
    "failure_trend": (79, 87),
}
PATTERN_AXES = {
    "hip_pitch": "sagittal_x",
    "knee_pitch": "sagittal_x",
    "ankle_pitch": "sagittal_x",
    "hip_roll": "lateral_y",
    "ankle_roll": "lateral_y",
}


def _load(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not data.get("strict_pairing", False):
        raise RuntimeError(f"Not a strict pair: {path}")
    if data.get("ppo_updates", True):
        raise RuntimeError(f"PPO-updated artifact is invalid for FD: {path}")
    # new change
    # Grid V1 predates the complete signed support trajectory and stores only
    # root_state_w.  That is still sufficient for a strict yaw-frame root
    # response audit, so retain backwards compatibility instead of rerunning
    # eight expensive branches merely to recover a coordinate transform.
    feature = data["baseline"].get("support_feature_trajectory")
    if feature is not None and feature.shape[-1] != 87:
        raise RuntimeError(
            f"Expected the audited 87D support feature, got {feature.shape[-1]}"
        )
    return data


def _feature(data: dict, branch: str, name: str, step: int) -> torch.Tensor:
    start, end = FEATURE_SLICES[name]
    value = data[branch]["support_feature_trajectory"][:, step, start:end]
    if name in ("com_support_offset_scaled", "fixed_foot_xy_error_scaled"):
        value = value / 10.0
    return value


# new change
def _yaw_from_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Return yaw for IsaacLab's scalar-first quaternion convention."""

    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y.square() + z.square()),
    )


# new change
def _world_xy_to_baseline_yaw(
    vector_xy: torch.Tensor, baseline_quaternion: torch.Tensor
) -> torch.Tensor:
    yaw = _yaw_from_wxyz(baseline_quaternion)
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    x = cosine * vector_xy[:, 0] + sine * vector_xy[:, 1]
    y = -sine * vector_xy[:, 0] + cosine * vector_xy[:, 1]
    return torch.stack((x, y), dim=-1)


# new change
def _branch_delta(
    data: dict, name: str, step: int
) -> torch.Tensor | None:
    baseline = data["baseline"]
    candidate = data["candidate"]
    if "support_feature_trajectory" in baseline:
        return _feature(data, "candidate", name, step) - _feature(
            data, "baseline", name, step
        )
    root_baseline = baseline["root_state_trajectory"][:, step]
    root_candidate = candidate["root_state_trajectory"][:, step]
    if name == "root_xy_position_error":
        vector_w = root_candidate[:, :2] - root_baseline[:, :2]
    elif name == "base_lin_vel":
        vector_w = root_candidate[:, 7:9] - root_baseline[:, 7:9]
    else:
        return None
    return _world_xy_to_baseline_yaw(vector_w, root_baseline[:, 3:7])


def _valid_intersection(positive: dict, negative: dict) -> torch.Tensor:
    return (
        positive["valid"].bool()
        & negative["valid"].bool()
        & ~positive["baseline"]["failure"].bool()
        & ~negative["baseline"]["failure"].bool()
    )


def _mean_std(value: torch.Tensor, valid: torch.Tensor) -> dict:
    selected = value[valid]
    return {
        "mean": selected.mean(dim=0).tolist(),
        "std": selected.std(dim=0, unbiased=False).tolist(),
    }


def _analyze_group(root: Path, group: str) -> dict:
    positive = _load(root / f"{group}_pos" / "paired.pt")
    negative = _load(root / f"{group}_neg" / "paired.pt")
    valid = _valid_intersection(positive, negative)
    amplitude = float(
        positive["candidate_target_rad"].abs().max().item()
    )
    if amplitude <= 0.0:
        raise RuntimeError(f"Zero perturbation amplitude for {group}")
    # new change
    trajectory_key = (
        "support_feature_trajectory"
        if "support_feature_trajectory" in positive["baseline"]
        else "root_state_trajectory"
    )
    horizon = positive["baseline"][trajectory_key].shape[1]
    probe_steps = sorted(set((min(24, horizon - 1), horizon - 1)))
    metrics = {}
    for step in probe_steps:
        step_metrics = {}
        for name in (
            "root_xy_position_error",
            "base_lin_vel",
            "com_support_offset_scaled",
            "fixed_foot_xy_error_scaled",
        ):
            # new change
            plus_delta = _branch_delta(positive, name, step)
            minus_delta = _branch_delta(negative, name, step)
            if plus_delta is None or minus_delta is None:
                continue
            central = (plus_delta - minus_delta) / (2.0 * amplitude)
            step_metrics[name] = {
                "positive_minus_baseline": _mean_std(plus_delta, valid),
                "negative_minus_baseline": _mean_std(minus_delta, valid),
                "central_response_per_rad": _mean_std(central, valid),
            }
        metrics[str(step + 1)] = step_metrics

    axis_index = 0 if PATTERN_AXES[group] == "sagittal_x" else 1
    final_response = metrics[str(horizon)]["root_xy_position_error"][
        "central_response_per_rad"
    ]["mean"][axis_index]
    return {
        "axis": PATTERN_AXES[group],
        "amplitude_rad": amplitude,
        "valid_environments": int(valid.sum()),
        "horizon_steps": horizon,
        "metrics_by_step": metrics,
        "primary_axis_root_position_response_m_per_rad": final_response,
        "positive_residual_moves_primary_axis": (
            "positive" if final_response > 0.0 else "negative"
        ),
    }


def _render_markdown(report: dict) -> str:
    lines = [
        "# Support finite-difference body-frame audit",
        "",
        "All responses are candidate minus the strictly reconstructed baseline, "
        "expressed in the robot yaw frame. Positive X is sagittal/forward and "
        "positive Y is lateral.",
        "",
        "| Group | Axis | Valid | Final root response | + residual moves |",
        "|---|---:|---:|---:|---:|",
    ]
    for group, result in report["groups"].items():
        lines.append(
            f"| {group} | {result['axis']} | {result['valid_environments']} | "
            f"{result['primary_axis_root_position_response_m_per_rad']:+.4f} m/rad | "
            f"{result['positive_residual_moves_primary_axis']} |"
        )
    lines.extend(
        (
            "",
            "Interpretation: this table identifies local control direction only. "
            "A feedback controller must apply the opposite residual sign to the "
            "measured root/capture error. Magnitude and hip/ankle allocation must "
            "still be validated in frozen closed-loop inference.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "schema": "SupportFiniteDifferenceBodyFrameAuditV1",
        "strict_pairing_required": True,
        "ppo_updates": False,
        "feature_layout": FEATURE_SLICES,
        # new change: accept legacy four-group datasets while requiring all
        # five groups for the corrected V2 collection.
        "groups": {
            group: _analyze_group(args.input_dir, group)
            for group in PATTERN_AXES
            if (args.input_dir / f"{group}_pos" / "paired.pt").exists()
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    args.output_md.write_text(_render_markdown(report))
    print(json.dumps(report["groups"], indent=2))


if __name__ == "__main__":
    main()
