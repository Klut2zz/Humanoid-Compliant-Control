#!/usr/bin/env python3
"""Solve bounded 10DOF support residuals from strict paired FD responses."""

# new change
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


# new change
ACTION_SCALE_RAD = 0.30
TARGET_LIMIT_RAD = 0.006
METRIC_NAMES = (
    "root_displacement_x_m",
    "root_displacement_y_m",
    "root_velocity_x_mps",
    "root_velocity_y_mps",
    "left_foot_drift_x_m",
    "left_foot_drift_y_m",
    "right_foot_drift_x_m",
    "right_foot_drift_y_m",
    "com_support_offset_x_m",
    "com_support_offset_y_m",
    "torso_projected_gravity_x",
    "torso_projected_gravity_y",
    "torso_angular_velocity_x_radps",
    "torso_angular_velocity_y_radps",
)
METRIC_SCALES = torch.tensor(
    [0.03, 0.03, 0.20, 0.20, 0.02, 0.02, 0.02, 0.02,
     0.04, 0.04, 0.10, 0.10, 0.50, 0.50],
    dtype=torch.float32,
)


# new change
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dataset", type=Path, required=True)
    parser.add_argument("--output_report", type=Path, required=True)
    parser.add_argument("--window_s", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, default=0.25)
    parser.add_argument("--target_limit_rad", type=float, default=TARGET_LIMIT_RAD)
    parser.add_argument("--max_residual_norm_rad", type=float, default=0.012)
    parser.add_argument("--min_predicted_improvement", type=float, default=0.01)
    # new change
    parser.add_argument(
        "--phase_smoothing_radius",
        type=int,
        default=0,
        help="Symmetric per-motion phase-neighbor radius for distillable labels.",
    )
    return parser.parse_args()


# new change
def term_slice(data: dict, term: str) -> slice:
    schema = data["short_horizon_metric_schema"]["support_metric_slices"]
    start, end = schema[term]
    return slice(int(start), int(end))


# new change
def physical_metric(data: dict, result: dict, window_steps: int) -> torch.Tensor:
    """Return signed last-window physical errors, one row per environment."""
    root = result["root_state_trajectory"]
    feet = result["support_foot_position_w_trajectory"]
    support = result["support_feature_trajectory"]
    window = slice(max(root.shape[1] - window_steps, 0), root.shape[1])
    root_displacement = root[:, window, :2].mean(1) - root[:, 0, :2]
    root_velocity = root[:, window, 7:9].mean(1)
    foot_drift = (
        feet[:, window, :, :2].mean(1) - feet[:, 0, :, :2]
    ).reshape(feet.shape[0], 4)
    com_offset = support[:, window, term_slice(data, "com_support_offset")].mean(1) / 10.0
    gravity = support[:, window, term_slice(data, "torso_projected_gravity")].mean(1)
    angular_velocity = support[:, window, term_slice(data, "base_ang_vel")].mean(1) / 0.25
    return torch.cat(
        (
            root_displacement,
            root_velocity,
            foot_drift,
            com_offset[:, :2],
            gravity[:, :2],
            angular_velocity[:, :2],
        ),
        dim=1,
    )


# new change
def com_margin_m(result: dict, window_steps: int) -> torch.Tensor:
    """Approximate COM margin inside a two-foot rectangular support hull."""
    feet = result["support_foot_position_w_trajectory"]
    com = result["whole_body_com_w_trajectory"]
    window = slice(max(feet.shape[1] - window_steps, 0), feet.shape[1])
    feet_xy = feet[:, window, :, :2].mean(1)
    com_xy = com[:, window, :2].mean(1)
    # G1 foot-center hull expanded by conservative half sole dimensions.
    lower = feet_xy.min(dim=1).values - torch.tensor([0.10, 0.045])
    upper = feet_xy.max(dim=1).values + torch.tensor([0.10, 0.045])
    margins = torch.cat((com_xy - lower, upper - com_xy), dim=1)
    return margins.min(dim=1).values


# new change
def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest.get("completed", False) or len(manifest["probes"]) != 20:
        raise RuntimeError("Finite-difference manifest is not complete.")
    if manifest.get("ppo_updates", True):
        raise RuntimeError("Teacher input must be inference-only data.")
    amplitude = float(manifest["amplitude_rad"])
    if not 0.0 < args.target_limit_rad <= 0.006:
        raise ValueError("Target limit must remain inside the accepted 6-mrad envelope.")
    window_steps = max(1, int(round(args.window_s / 0.02)))

    by_joint: dict[int, dict[str, dict]] = {index: {} for index in range(10)}
    for probe in manifest["probes"].values():
        paired = torch.load(probe["paired"], map_location="cpu", weights_only=False)
        by_joint[int(probe["joint_index"])][str(probe["sign"])] = paired
    template = by_joint[0]["positive"]
    baseline = template["baseline"]
    features = template["support_recovery_features"].float()
    # new change
    temporal_features = template.get("support_temporal_features")
    if temporal_features is None:
        raise RuntimeError(
            "DAgger teacher requires support_temporal_features: "
            "87D state + previous 10DOF residual + update clock."
        )
    temporal_features = temporal_features.float()
    if temporal_features.shape != (features.shape[0], 98):
        raise RuntimeError(
            f"Expected [N,98] temporal features, got {tuple(temporal_features.shape)}."
        )
    baseline_metric = physical_metric(template, baseline, window_steps)
    normalized_baseline = baseline_metric / METRIC_SCALES
    num_envs = features.shape[0]
    response = torch.zeros(num_envs, len(METRIC_NAMES), 10)
    lower = torch.full((num_envs, 10), -args.target_limit_rad / amplitude)
    upper = torch.full((num_envs, 10), args.target_limit_rad / amplitude)
    branch_safety = {}
    valid = template["valid"].bool() & ~baseline["failure"].bool()

    for joint in range(10):
        plus_data = by_joint[joint]["positive"]
        minus_data = by_joint[joint]["negative"]
        if plus_data["snapshot_fingerprint"] != minus_data["snapshot_fingerprint"]:
            raise RuntimeError(f"Joint {joint} probes did not start from one snapshot.")
        plus = plus_data["candidate"]
        minus = minus_data["candidate"]
        plus_metric = physical_metric(plus_data, plus, window_steps)
        minus_metric = physical_metric(minus_data, minus, window_steps)
        plus_safe = plus_data["valid"].bool() & ~plus["failure"].bool()
        minus_safe = minus_data["valid"].bool() & ~minus["failure"].bool()
        central = (plus_metric - minus_metric) / (2.0 * METRIC_SCALES)
        plus_one_sided = (plus_metric - baseline_metric) / METRIC_SCALES
        minus_one_sided = (baseline_metric - minus_metric) / METRIC_SCALES
        column = torch.where(
            (plus_safe & minus_safe).unsqueeze(1), central,
            torch.where(plus_safe.unsqueeze(1), plus_one_sided, minus_one_sided),
        )
        neither = ~(plus_safe | minus_safe)
        column[neither] = 0.0
        response[:, :, joint] = column
        upper[~plus_safe, joint] = torch.minimum(
            upper[~plus_safe, joint], torch.zeros_like(upper[~plus_safe, joint])
        )
        lower[~minus_safe, joint] = torch.maximum(
            lower[~minus_safe, joint], torch.zeros_like(lower[~minus_safe, joint])
        )
        lower[neither, joint] = 0.0
        upper[neither, joint] = 0.0
        branch_safety[str(joint)] = {
            "positive_failures": int((~plus_safe & plus_data["valid"].bool()).sum()),
            "negative_failures": int((~minus_safe & minus_data["valid"].bool()).sum()),
            "response_frobenius_mean": float(torch.linalg.vector_norm(column, dim=1).mean()),
        }

    identity = torch.eye(10).unsqueeze(0).expand(num_envs, -1, -1)
    normal = response.transpose(1, 2) @ response + args.ridge * identity
    rhs = -(response.transpose(1, 2) @ normalized_baseline.unsqueeze(-1)).squeeze(-1)
    solution = torch.linalg.solve(normal, rhs)
    solution = torch.maximum(torch.minimum(solution, upper), lower)
    residual_rad = amplitude * solution
    norm = torch.linalg.vector_norm(residual_rad, dim=1, keepdim=True)
    residual_rad *= torch.clamp(args.max_residual_norm_rad / norm.clamp_min(1.0e-9), max=1.0)
    # new change
    # Strict finite differences can amplify tiny long-horizon contact changes.
    # The physical clips are cyclic, so smooth only solved actions of the same
    # motion across adjacent phases before recomputing predicted benefit.
    motion_id = template["motion_id"].long()
    if args.phase_smoothing_radius < 0:
        raise ValueError("--phase_smoothing_radius must be non-negative.")
    if args.phase_smoothing_radius:
        smoothed = residual_rad.clone()
        radius = args.phase_smoothing_radius
        offsets = list(range(-radius, radius + 1))
        weights = torch.tensor(
            [float(radius + 1 - abs(offset)) for offset in offsets]
        )
        weights /= weights.sum()
        for motion in torch.unique(motion_id):
            indices = torch.nonzero(motion_id == motion, as_tuple=False).squeeze(1)
            order = torch.argsort(template["source_time_s"][indices])
            ordered = residual_rad[indices[order]]
            filtered = torch.zeros_like(ordered)
            for weight, offset in zip(weights, offsets, strict=True):
                filtered += weight * torch.roll(ordered, shifts=offset, dims=0)
            smoothed[indices[order]] = filtered
        residual_rad = smoothed.clamp(
            min=-args.target_limit_rad, max=args.target_limit_rad
        )
        norm = torch.linalg.vector_norm(residual_rad, dim=1, keepdim=True)
        residual_rad *= torch.clamp(
            args.max_residual_norm_rad / norm.clamp_min(1.0e-9), max=1.0
        )
    predicted = normalized_baseline + (response @ (residual_rad / amplitude).unsqueeze(-1)).squeeze(-1)
    baseline_cost = normalized_baseline.square().mean(1)
    predicted_cost = predicted.square().mean(1)
    predicted_improvement = (baseline_cost - predicted_cost) / baseline_cost.clamp_min(1.0e-8)
    accepted = valid & (predicted_improvement >= args.min_predicted_improvement)
    residual_rad[~accepted] = 0.0

    env_index = torch.arange(num_envs)
    phase_rank = torch.div(env_index, 8, rounding_mode="floor")
    split_code = phase_rank % 10
    split = torch.full((num_envs,), 2, dtype=torch.int64)
    split[split_code <= 6] = 0
    split[(split_code == 7) | (split_code == 8)] = 1
    split_names = {0: "train", 1: "validation", 2: "test"}
    dataset = {
        "schema": "StationaryTemporalSupportFDTeacherDatasetV2",
        "support_adapter": features,
        # new change
        "support_temporal_adapter": temporal_features,
        "previous_support_residual_rad": template[
            "previous_support_residual_rad"
        ].float(),
        "steps_since_residual_update": template[
            "steps_since_residual_update"
        ].long(),
        "support_target_rad": residual_rad,
        "support_target_actor": residual_rad / ACTION_SCALE_RAD,
        "response_normalized_per_probe": response,
        "baseline_metric_physical": baseline_metric,
        "predicted_metric_normalized": predicted,
        "predicted_relative_improvement": predicted_improvement,
        "accepted": accepted,
        "valid_baseline": valid,
        "lower_bound_probe_units": lower,
        "upper_bound_probe_units": upper,
        "motion_id": motion_id,
        "source_time_s": template["source_time_s"],
        "split": split,
        "split_names": split_names,
        "support_action_ids": tuple(template["support_action_ids"]),
        "support_joint_names": tuple(template["support_joint_names"]),
        "metric_names": METRIC_NAMES,
        "metric_scales": METRIC_SCALES,
        "amplitude_rad": amplitude,
        "target_limit_rad": args.target_limit_rad,
        "max_residual_norm_rad": args.max_residual_norm_rad,
        "ridge": args.ridge,
        "window_s": args.window_s,
        "phase_smoothing_radius": args.phase_smoothing_radius,
        "source_manifest": str(args.manifest.resolve()),
        "source_policy": "frozen model_1375",
        "teacher_type": "strict_paired_local_finite_difference_constrained_ridge",
        # new change
        "feature_order": (
            "support_adapter[87],previous_support_residual_rad[10],"
            "elapsed_fraction_since_update[1]"
        ),
        "dataset_aggregation": True,
        "uses_q_error_teacher": False,
        "ppo_updates": False,
        "phase_smoothing_radius": args.phase_smoothing_radius,
    }
    args.output_dataset.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args.output_dataset)
    baseline_margin = com_margin_m(baseline, window_steps)
    report = {
        "schema": dataset["schema"],
        "samples": num_envs,
        "accepted": int(accepted.sum()),
        "accepted_fraction": float(accepted.float().mean()),
        "baseline_failures_excluded": int((~valid & template["valid"].bool()).sum()),
        "target_abs_mean_mrad": float(residual_rad.abs().mean() * 1000.0),
        "target_abs_max_mrad": float(residual_rad.abs().max() * 1000.0),
        "target_norm_mean_mrad": float(torch.linalg.vector_norm(residual_rad, dim=1).mean() * 1000.0),
        "predicted_improvement_mean_accepted": float(predicted_improvement[accepted].mean()) if accepted.any() else 0.0,
        "baseline_com_support_margin_mean_m": float(baseline_margin[valid].mean()),
        "split_counts": {
            split_names[index]: int(((split == index) & accepted).sum()) for index in range(3)
        },
        "branch_safety": branch_safety,
        "metric_names": METRIC_NAMES,
        "metric_scales": METRIC_SCALES.tolist(),
        "ppo_updates": False,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
