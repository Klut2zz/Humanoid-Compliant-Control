#!/usr/bin/env python3
"""Offline initialize the 87D FoundationV2 support head from FD labels."""

# new change
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from tensordict import TensorDict


# new change
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source_checkpoint", type=Path, required=True)
    parser.add_argument("--legacy_jit", type=Path, required=True)
    parser.add_argument("--models_py", type=Path, required=True)
    parser.add_argument("--output_checkpoint", type=Path, required=True)
    parser.add_argument("--output_jit", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    # new change
    parser.add_argument("--inference_scale", type=float, default=1.0)
    # new change
    parser.add_argument(
        "--disable_feature_normalization",
        action="store_true",
        help="A/B the legacy raw 87D input while retaining validation selection.",
    )
    return parser.parse_args()


# new change
def load_models(path: Path):
    spec = importlib.util.spec_from_file_location("local_support_fd_models", path)
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
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=False)
    if dataset.get("teacher_type") != "strict_paired_local_finite_difference_constrained_ridge":
        raise RuntimeError("Refusing a non-FD teacher dataset.")
    if dataset.get("uses_q_error_teacher", True) or dataset.get("ppo_updates", True):
        raise RuntimeError("Dataset must be local-FD inference data, not old q-error/PPO data.")
    features = dataset["support_adapter"].float()
    targets = dataset["support_target_actor"].float()
    split = dataset["split"].long()
    accepted = dataset["accepted"].bool()
    if features.shape[1] != 87 or targets.shape != (features.shape[0], 10):
        raise ValueError(f"Unexpected dataset shapes: {features.shape}, {targets.shape}.")

    models = load_models(args.models_py)
    actor = models.StandCoordinationStationaryFoundationV2MLPModel(
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
    disallowed_missing = [
        key for key in missing
        if not (
            "residual_adapter" in key
            or "adapter_limits" in key
            or "adapter_action_ids" in key
            or "adapter_projection" in key
            or key == "analytic_error_gain"
            # new change
            or key in (
                "support_feature_mean",
                "support_feature_scale",
                "support_output_scale",
            )
        )
    ]
    disallowed_unexpected = [key for key in unexpected if key != "distribution.std_param"]
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={disallowed_missing}, unexpected={disallowed_unexpected}"
        )

    # new change
    # Verify that the untouched source actor and all zero heads are still a
    # strict deterministic equivalent of model_1375 before fitting labels.
    legacy = torch.jit.load(str(args.legacy_jit), map_location=device).eval()
    probe = make_observation(2048, device)
    probe["policy"] = torch.randn_like(probe["policy"])
    probe["left_arm_adapter"] = torch.randn_like(probe["left_arm_adapter"])
    probe["right_arm_adapter"] = torch.randn_like(probe["right_arm_adapter"])
    probe["waist_adapter"] = torch.randn_like(probe["waist_adapter"])
    probe["support_adapter"] = torch.randn_like(probe["support_adapter"])
    actor.eval()
    with torch.inference_mode():
        zero_error = (actor(probe) - legacy(probe["policy"])).abs()
        zero_support = actor._support_output(probe["support_adapter"]).abs().max()
    if float(zero_error.max()) > 1.0e-7 or float(zero_support) != 0.0:
        raise RuntimeError(
            f"Pretraining source is not equivalent: action={float(zero_error.max())}, "
            f"support={float(zero_support)}"
        )

    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    for parameter in actor.support_residual_adapter.parameters():
        parameter.requires_grad_(True)
    # new change
    # Statistics are computed only from the training phase split.  They are
    # stored as model buffers and used solely by the support head.
    train_index = torch.nonzero(split == 0, as_tuple=False).squeeze(1)
    validation_index = torch.nonzero(split == 1, as_tuple=False).squeeze(1)
    feature_mean = features[train_index].mean(0).to(device)
    feature_scale = features[train_index].std(0, unbiased=False).clamp_min(0.02).to(device)
    if args.disable_feature_normalization:
        feature_mean.zero_()
        feature_scale.fill_(1.0)
    actor.support_feature_mean.copy_(feature_mean)
    actor.support_feature_scale.copy_(feature_scale)
    # new change
    # FoundationV2 inherited broad PPO-era limits (up to 30 mrad).  A
    # distilled local teacher must never extrapolate beyond the labels that
    # passed strict combined replay.  Store per-joint hard envelopes in the
    # existing model buffer before fitting and JIT export.
    target_envelope_rad = (
        dataset["support_target_rad"].abs().max(dim=0).values * 1.05
    ).clamp(min=0.0005, max=float(dataset["target_limit_rad"]))
    actor.support_adapter_limits.copy_(target_envelope_rad.to(device) / ACTION_SCALE_RAD)
    optimizer = torch.optim.Adam(
        actor.support_residual_adapter.parameters(), lr=args.learning_rate
    )
    reports = []
    # new change
    best_validation_rmse_mrad = float("inf")
    best_epoch = 0
    best_support_state = None
    actor.train()
    for epoch in range(args.epochs):
        order = train_index[torch.randperm(train_index.numel())]
        loss_sum = 0.0
        count = 0
        for start in range(0, order.numel(), args.batch_size):
            index = order[start : start + args.batch_size]
            x = features[index].to(device)
            y = targets[index].to(device)
            weight = torch.where(
                accepted[index].to(device),
                torch.ones(index.numel(), device=device),
                torch.full((index.numel(),), 0.25, device=device),
            )
            prediction = actor._support_output(x)
            per_sample = (prediction - y).square().mean(1)
            loss = (per_sample * weight).sum() / weight.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.support_residual_adapter.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss) * index.numel()
            count += index.numel()
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == args.epochs:
            actor.eval()
            with torch.inference_mode():
                prediction = actor._support_output(features[validation_index].to(device))
                error_rad = ACTION_SCALE_RAD * (
                    prediction - targets[validation_index].to(device)
                )
                report = {
                    "epoch": epoch + 1,
                    "train_weighted_mse_actor": loss_sum / max(count, 1),
                    "validation_mae_mrad": float(error_rad.abs().mean() * 1000.0),
                    "validation_rmse_mrad": float(torch.sqrt(error_rad.square().mean()) * 1000.0),
                    "validation_max_mrad": float(error_rad.abs().max() * 1000.0),
                }
                reports.append(report)
                print(report, flush=True)
                if report["validation_rmse_mrad"] < best_validation_rmse_mrad:
                    best_validation_rmse_mrad = report["validation_rmse_mrad"]
                    best_epoch = epoch + 1
                    best_support_state = {
                        key: value.detach().cpu().clone()
                        for key, value in actor.support_residual_adapter.state_dict().items()
                    }
            actor.train()

    # new change
    if best_support_state is None:
        raise RuntimeError("No validation checkpoint was recorded.")
    actor.support_residual_adapter.load_state_dict(best_support_state)
    # new change
    if not 0.0 < args.inference_scale <= 1.0:
        raise ValueError("--inference_scale must be in (0, 1].")
    actor.support_output_scale.fill_(args.inference_scale)
    actor.eval()
    output_checkpoint = dict(checkpoint)
    output_actor_state = {
        key: value.detach().cpu() for key, value in actor.state_dict().items()
    }
    # new change: retain the source exploration parameter for future loader
    # compatibility even though frozen deterministic export has no distribution.
    if "distribution.std_param" in checkpoint["actor_state_dict"]:
        output_actor_state["distribution.std_param"] = checkpoint[
            "actor_state_dict"
        ]["distribution.std_param"].detach().cpu()
    output_checkpoint["actor_state_dict"] = output_actor_state
    output_checkpoint["optimizer_state_dict"] = None
    output_checkpoint["infos"] = {
        **(checkpoint.get("infos") or {}),
        "stationary_local_support_fd_supervised_init": True,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "teacher_type": dataset["teacher_type"],
        "uses_q_error_teacher": False,
        "ppo_updates": False,
        "zero_init_equivalence_max_abs_action": float(zero_error.max()),
        "reports": reports,
        "best_validation_epoch": best_epoch,
        "best_validation_rmse_mrad": best_validation_rmse_mrad,
        "support_feature_normalization": not args.disable_feature_normalization,
        "inference_scale": args.inference_scale,
        "support_hard_limit_rad": target_envelope_rad.tolist(),
    }
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output_checkpoint)
    scripted = torch.jit.script(actor.as_jit())
    args.output_jit.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(args.output_jit))
    summary = {
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_jit": str(args.output_jit.resolve()),
        "sample_count": int(features.shape[0]),
        "accepted_count": int(accepted.sum()),
        "zero_init_equivalence_max_abs_action": float(zero_error.max()),
        "zero_init_support_max_abs": float(zero_support),
        "selected_validation": next(
            report for report in reports if report["epoch"] == best_epoch
        ),
        "support_feature_normalization": not args.disable_feature_normalization,
        "inference_scale": args.inference_scale,
        "support_hard_limit_rad": target_envelope_rad.tolist(),
        "trained_parameters": [
            name for name, parameter in actor.named_parameters() if parameter.requires_grad
        ],
        "ppo_updates": False,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


# new change
ACTION_SCALE_RAD = 0.30


if __name__ == "__main__":
    main()
