#!/usr/bin/env python3
"""Offline-initialize the V6 10DOF support head from measured base state."""

# new change
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from tensordict import TensorDict


# new change
ACTION_SCALE_RAD = 0.30
SUPPORT_ACTION_IDS = (0, 1, 9, 10, 13, 14, 3, 4, 17, 18)
SUPPORT_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)


# new change
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--source_checkpoint", type=Path, required=True)
    parser.add_argument("--legacy_jit", type=Path, required=True)
    parser.add_argument("--models_py", type=Path, required=True)
    parser.add_argument("--output_checkpoint", type=Path, required=True)
    parser.add_argument("--output_jit", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--limit_rad", type=float, default=0.012)
    parser.add_argument("--velocity_horizon_s", type=float, default=0.30)
    parser.add_argument("--com_weight", type=float, default=0.50)
    parser.add_argument("--error_scale_m", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# new change
def load_models(path: Path):
    spec = importlib.util.spec_from_file_location("base_state_aware_v6_models", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# new change
def make_observation(batch: int, device: torch.device) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.zeros(batch, 705, device=device),
            "left_arm_adapter": torch.zeros(batch, 37, device=device),
            "right_arm_adapter": torch.zeros(batch, 37, device=device),
            "waist_adapter": torch.zeros(batch, 25, device=device),
            "support_adapter": torch.zeros(batch, 87, device=device),
        },
        batch_size=[batch],
        device=device,
    )


# new change
def audit_support_mapping() -> None:
    compat_path = (
        Path(__file__).resolve().parents[1]
        # new change
        / "softmimic_gym/softmimic_gym/assets/robots/g1_compat.py"
    )
    namespace: dict[str, object] = {}
    exec(compile(compat_path.read_text(encoding="utf-8"), str(compat_path), "exec"), namespace)
    canonical = tuple(namespace["G1_SOFTMIMIC_29_JOINT_NAMES"])
    resolved = tuple(canonical[index] for index in SUPPORT_ACTION_IDS)
    if resolved != SUPPORT_JOINT_NAMES:
        raise RuntimeError(
            "Strict 29DOF support mapping drifted: "
            f"expected={SUPPORT_JOINT_NAMES}, resolved={resolved}."
        )


# new change
def extract_features(artifact_path: Path) -> torch.Tensor:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact.get("ppo_updates", True):
        raise RuntimeError(f"Refusing PPO-derived artifact: {artifact_path}")
    result = artifact.get("result")
    if result is None:
        result = artifact.get("baseline")
    if result is None or "support_feature_trajectory" not in result:
        raise KeyError(f"No support trajectory in {artifact_path}")
    features = result["support_feature_trajectory"].float()
    if features.ndim != 3 or features.shape[-1] != 87:
        raise ValueError(f"Unexpected support features {tuple(features.shape)}")
    valid = artifact.get("valid")
    if valid is not None:
        features = features[valid.bool()]
    finite = torch.isfinite(features).all(dim=-1)
    return features[finite]


# new change
def analytic_target(
    features: torch.Tensor,
    limit_rad: float,
    velocity_horizon_s: float,
    com_weight: float,
    error_scale_m: float,
) -> torch.Tensor:
    # 87D layout is fixed by G1SupportRecoveryAwareTeacherV3ObservationsCfg:
    # COM offset 52:54, root XY error 60:62, and base linear velocity 62:65.
    com_offset_b = features[:, 52:54] / 10.0
    root_error_b = features[:, 60:62]
    root_velocity_b = features[:, 62:64]
    capture_error_b = (
        root_error_b
        + velocity_horizon_s * root_velocity_b
        + com_weight * com_offset_b
    )
    axis = limit_rad * torch.tanh(capture_error_b / error_scale_m)
    sagittal = axis[:, 0]
    lateral = axis[:, 1]
    return torch.stack(
        (
            sagittal,
            sagittal,
            -sagittal,
            -sagittal,
            sagittal,
            sagittal,
            -lateral,
            -lateral,
            lateral,
            lateral,
        ),
        dim=1,
    )


# new change
def main() -> None:
    args = parse_args()
    audit_support_mapping()
    if args.limit_rad <= 0.0 or args.limit_rad > 0.060:
        raise ValueError("--limit_rad must be in (0, 0.060].")
    if args.error_scale_m <= 0.0:
        raise ValueError("--error_scale_m must be positive.")
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    features = torch.cat([extract_features(path) for path in args.artifact], dim=0)
    targets_rad = analytic_target(
        features,
        args.limit_rad,
        args.velocity_horizon_s,
        args.com_weight,
        args.error_scale_m,
    )
    targets_actor = targets_rad / ACTION_SCALE_RAD
    permutation = torch.randperm(features.shape[0])
    split_at = int(0.90 * features.shape[0])
    train_index = permutation[:split_at]
    validation_index = permutation[split_at:]

    models = load_models(args.models_py)
    actor = models.StandCoordinationStationaryBaseStateAwareV6MLPModel(
        obs=make_observation(1, device),
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=29,
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
    ).to(device)
    checkpoint = torch.load(args.source_checkpoint, map_location=device, weights_only=False)
    missing, unexpected = actor.load_state_dict(checkpoint["actor_state_dict"], strict=False)
    allowed_missing_prefixes = (
        # new change
        "left_arm_",
        "right_arm_",
        "left_residual_adapter.",
        "right_residual_adapter.",
        "waist_residual_adapter.",
        "support_residual_adapter.",
        "left_adapter_",
        "right_adapter_",
        "waist_adapter_",
        "support_adapter_",
        "support_feature_",
        "support_output_scale",
        "analytic_error_gain",
    )
    disallowed_missing = [key for key in missing if not key.startswith(allowed_missing_prefixes)]
    disallowed_unexpected = [key for key in unexpected if key != "distribution.std_param"]
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={disallowed_missing}, unexpected={disallowed_unexpected}"
        )

    legacy = torch.jit.load(str(args.legacy_jit), map_location=device).eval()
    probe = make_observation(2048, device)
    for key in probe.keys():
        probe[key] = torch.randn_like(probe[key])
    actor.eval()
    with torch.inference_mode():
        zero_action_error = (actor(probe) - legacy(probe["policy"])).abs().max()
        zero_support_error = actor._support_output(probe["support_adapter"]).abs().max()
    if float(zero_action_error) > 1.0e-7 or float(zero_support_error) != 0.0:
        raise RuntimeError(
            "V6 source is not model_1375-equivalent: "
            f"action={float(zero_action_error)}, support={float(zero_support_error)}"
        )

    feature_mean = features[train_index].mean(dim=0).to(device)
    feature_scale = features[train_index].std(dim=0, unbiased=False).clamp_min(0.02).to(device)
    actor.support_feature_mean.copy_(feature_mean)
    actor.support_feature_scale.copy_(feature_scale)
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    for parameter in actor.support_residual_adapter.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(actor.support_residual_adapter.parameters(), lr=args.learning_rate)
    best_rmse_mrad = float("inf")
    best_epoch = 0
    best_state = None
    reports: list[dict[str, float | int]] = []
    actor.train()
    for epoch in range(args.epochs):
        order = train_index[torch.randperm(train_index.numel())]
        for start in range(0, order.numel(), args.batch_size):
            index = order[start : start + args.batch_size]
            prediction = actor._support_output(features[index].to(device))
            loss = torch.nn.functional.smooth_l1_loss(
                prediction, targets_actor[index].to(device), beta=0.005
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.support_residual_adapter.parameters(), 1.0)
            optimizer.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            actor.eval()
            with torch.inference_mode():
                prediction = actor._support_output(features[validation_index].to(device))
                error_rad = ACTION_SCALE_RAD * (
                    prediction - targets_actor[validation_index].to(device)
                )
                report = {
                    "epoch": epoch + 1,
                    "validation_mae_mrad": float(error_rad.abs().mean() * 1000.0),
                    "validation_rmse_mrad": float(error_rad.square().mean().sqrt() * 1000.0),
                    "validation_max_mrad": float(error_rad.abs().max() * 1000.0),
                }
            reports.append(report)
            print(report, flush=True)
            if report["validation_rmse_mrad"] < best_rmse_mrad:
                best_rmse_mrad = float(report["validation_rmse_mrad"])
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in actor.support_residual_adapter.state_dict().items()
                }
            actor.train()
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected.")
    actor.support_residual_adapter.load_state_dict(best_state)
    actor.eval()

    output_checkpoint = dict(checkpoint)
    output_actor_state = {key: value.detach().cpu() for key, value in actor.state_dict().items()}
    if "distribution.std_param" in checkpoint["actor_state_dict"]:
        output_actor_state["distribution.std_param"] = checkpoint["actor_state_dict"][
            "distribution.std_param"
        ].detach().cpu()
    output_checkpoint["actor_state_dict"] = output_actor_state
    output_checkpoint["optimizer_state_dict"] = None
    provenance = {
        "stationary_base_state_aware_v6_supervised_init": True,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "artifacts": [str(path.resolve()) for path in args.artifact],
        "support_action_ids": list(SUPPORT_ACTION_IDS),
        "support_joint_names": list(SUPPORT_JOINT_NAMES),
        "target_limit_rad": args.limit_rad,
        "velocity_horizon_s": args.velocity_horizon_s,
        "com_weight": args.com_weight,
        "error_scale_m": args.error_scale_m,
        "sample_count": int(features.shape[0]),
        "zero_init_equivalence_max_abs_action": float(zero_action_error),
        "best_validation_epoch": best_epoch,
        "best_validation_rmse_mrad": best_rmse_mrad,
        "ppo_updates": False,
    }
    output_checkpoint["infos"] = {**(checkpoint.get("infos") or {}), **provenance}
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output_checkpoint)
    scripted = torch.jit.script(actor.as_jit())
    args.output_jit.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(args.output_jit))
    summary = {
        **provenance,
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_jit": str(args.output_jit.resolve()),
        "selected_validation": next(item for item in reports if item["epoch"] == best_epoch),
        "target_mean_abs_mrad": float(targets_rad.abs().mean() * 1000.0),
        "target_max_abs_mrad": float(targets_rad.abs().max() * 1000.0),
        "trained_parameters": [
            name for name, parameter in actor.named_parameters() if parameter.requires_grad
        ],
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


# new change
if __name__ == "__main__":
    main()
