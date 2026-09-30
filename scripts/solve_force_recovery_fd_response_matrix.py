#!/usr/bin/env python3
"""Solve a bounded 12DOF recovery target from paired FD trajectories."""

# new change
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


# new change
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--response_time_s", type=float, default=1.0)
    parser.add_argument("--average_window_s", type=float, default=0.2)
    parser.add_argument("--step_dt", type=float, default=0.02)
    parser.add_argument("--ridge", type=float, default=1.0e6)
    parser.add_argument("--target_scale", type=float, default=1.0)
    parser.add_argument("--target_limit_rad", type=float, default=0.001)
    return parser.parse_args()


# new change
def support_objective(result: dict[str, torch.Tensor], end: int, width: int) -> torch.Tensor:
    """Dimensionless endpoint objective in the verified 87D support layout."""

    features = result["support_feature_trajectory"][:, end - width : end].mean(1).double()
    # fixed_foot_xy_error and COM offset were scaled by 10 in the observation.
    foot_velocity_xy = torch.stack(
        (features[:, 65], features[:, 66], features[:, 68], features[:, 69]), dim=1
    )
    return torch.cat(
        (
            features[:, 60:62] / 0.50,  # root XY position [m]
            features[:, 62:64] / 0.20,  # root XY velocity [m/s]
            (features[:, 75:79] / 10.0) / 0.20,  # fixed feet XY [m]
            foot_velocity_xy / 0.20,  # support-link XY velocity [m/s]
            (features[:, 52:54] / 10.0) / 0.06,  # COM/support offset [m]
            features[:, 54:56] / 0.15,  # torso projected-gravity XY
            (features[:, 57:59] / 0.25) / 0.40,  # base angular velocity [rad/s]
        ),
        dim=1,
    )


# new change
def main() -> None:
    args = parse_args()
    if args.response_time_s <= 0.0 or args.average_window_s <= 0.0:
        raise ValueError("Response and averaging windows must be positive.")
    if args.ridge <= 0.0:
        raise ValueError("--ridge must be positive.")
    if not 0.0 < args.target_scale <= 1.0:
        raise ValueError("--target_scale must be in (0, 1].")
    if not 0.0 < args.target_limit_rad <= 0.002:
        raise ValueError("--target_limit_rad must be in (0, 0.002].")

    manifest = json.loads((args.fd_dir / "manifest.json").read_text())
    if not manifest.get("completed", False) or manifest.get("ppo_updates", True):
        raise RuntimeError("FD matrix is incomplete or contains PPO updates.")
    end = int(round(args.response_time_s / args.step_dt))
    width = int(round(args.average_window_s / args.step_dt))
    if width < 1 or end < width:
        raise ValueError("Invalid response/averaging window.")

    joint_names = tuple(manifest["joint_names"])
    amplitude = float(manifest["amplitude_rad"])
    by_joint: dict[int, dict[str, dict[str, object]]] = {}
    all_data: list[dict[str, object]] = []
    for _, info in manifest["probes"].items():
        data = torch.load(info["paired"], map_location="cpu", weights_only=False)
        if not data.get("strict_pairing", False) or not data.get(
            "force_conditioned_pairing", False
        ):
            raise RuntimeError(f"Probe is not strict force-paired: {info['paired']}")
        if data.get("ppo_updates", True):
            raise RuntimeError(f"Probe contains PPO updates: {info['paired']}")
        if not bool(data["force_replay_match"].all()) or not bool(
            data["force_candidate_match"].all()
        ):
            raise RuntimeError(f"Force/IK mismatch: {info['paired']}")
        by_joint.setdefault(int(info["joint_index"]), {})[str(info["sign"])] = data
        all_data.append(data)
    if len(by_joint) != len(joint_names) or any(
        set(pair) != {"positive", "negative"} for pair in by_joint.values()
    ):
        raise RuntimeError("Every support joint needs one positive and negative probe.")

    reference = all_data[0]
    baseline = support_objective(reference["baseline"], end, width)
    response_columns = []
    valid_columns = []
    for joint_index in range(len(joint_names)):
        positive = by_joint[joint_index]["positive"]
        negative = by_joint[joint_index]["negative"]
        positive_metric = support_objective(positive["candidate"], end, width)
        negative_metric = support_objective(negative["candidate"], end, width)
        response_columns.append((positive_metric - negative_metric) / (2.0 * amplitude))
        valid_columns.extend((positive["valid"].bool(), negative["valid"].bool()))
    response = torch.stack(response_columns, dim=2)
    common_valid = torch.stack(valid_columns, dim=1).all(dim=1)

    transpose = response.transpose(1, 2)
    regularizer = args.ridge * torch.eye(
        len(joint_names), dtype=response.dtype
    ).unsqueeze(0)
    rhs = -(transpose @ baseline.unsqueeze(-1))
    raw_target = torch.linalg.solve(transpose @ response + regularizer, rhs).squeeze(-1)
    support_target = (raw_target * args.target_scale).clamp(
        -args.target_limit_rad, args.target_limit_rad
    )
    support_target[~common_valid] = 0.0
    predicted = baseline + (response @ support_target.unsqueeze(-1)).squeeze(-1)
    baseline_norm = torch.linalg.vector_norm(baseline, dim=1)
    predicted_norm = torch.linalg.vector_norm(predicted, dim=1)

    force_features = reference.get("force_gate_features")
    if force_features is None:
        force_features = reference["force_context_trajectory"][
            "force_context_force_gate_features_trajectory"
        ][:, 0]
    state_features = torch.cat(
        (
            reference["support_recovery_features"].float(),
            reference["support_temporal_features"].float(),
            force_features.float(),
        ),
        dim=-1,
    )
    output = args.output or (args.fd_dir / "response_matrix_teacher.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema": "ForceRecovery12DOFResponseMatrixTeacherV1",
        "support_target_rad": support_target.float(),
        "raw_target_rad": raw_target.float(),
        "valid": common_valid,
        "state_features": state_features,
        "motion_id": reference["motion_id"],
        "source_time_s": reference["source_time_s"],
        "baseline_objective": baseline.float(),
        "predicted_objective": predicted.float(),
        "response_matrix": response.float(),
        "support_joint_names": reference["support_joint_names"],
        "support_action_ids": reference["support_action_ids"],
        "snapshot_fingerprint": reference["snapshot_fingerprint"],
        "response_time_s": args.response_time_s,
        "average_window_s": args.average_window_s,
        "ridge": args.ridge,
        "target_scale": args.target_scale,
        "target_limit_rad": args.target_limit_rad,
        "strict_pairing": True,
        "ppo_updates": False,
    }
    torch.save(artifact, output)
    valid_target = support_target[common_valid]
    ratio = predicted_norm[common_valid] / baseline_norm[common_valid].clamp_min(1.0e-8)
    summary = {
        "schema": artifact["schema"],
        "output": str(output.resolve()),
        "valid_envs": int(common_valid.sum()),
        "response_time_s": args.response_time_s,
        "average_window_s": args.average_window_s,
        "ridge": args.ridge,
        "mean_abs_target_mrad": float(valid_target.abs().mean() * 1000.0),
        "p95_max_target_mrad": float(
            torch.quantile(valid_target.abs().amax(dim=1), 0.95) * 1000.0
        ),
        "saturation_fraction": float(
            (valid_target.abs() >= args.target_limit_rad * 0.999).float().mean()
        ),
        "predicted_objective_ratio_mean": float(ratio.mean()),
        "predicted_improved_fraction": float((ratio < 1.0).float().mean()),
        "strict_pairing": True,
        "ppo_updates": False,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
