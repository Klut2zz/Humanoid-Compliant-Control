"""Offline-distill online 10DOF stationary-support feedback into the V2 head."""

# new change
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn


# new change
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--features", required=True)
parser.add_argument("--source_checkpoint", required=True)
parser.add_argument("--output_checkpoint", required=True)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--learning_rate", type=float, default=1.0e-3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--max_samples",
    type=int,
    default=0,
    help="Optional deterministic sample cap; zero uses the complete dataset.",
)
args = parser.parse_args()


# new change
SUPPORT_ACTION_IDS = (0, 1, 9, 10, 13, 14, 3, 4, 17, 18)
SUPPORT_TARGET_LIMITS_RAD = (
    0.006,
    0.006,
    0.006,
    0.006,
    0.006,
    0.006,
    0.006,
    0.006,
    0.006,
    0.006,
)


# new change
def make_support_adapter() -> nn.ModuleDict:
    def head(hidden: int, output: int) -> nn.Sequential:
        module = nn.Sequential(
            nn.Linear(79, hidden),
            nn.ELU(),
            nn.Linear(hidden, output),
        )
        nn.init.zeros_(module[-1].weight)
        nn.init.zeros_(module[-1].bias)
        return module

    return nn.ModuleDict(
        {
            "sagittal": head(48, 6),
            "lateral": head(32, 4),
        }
    )


# new change
def support_output_actor(
    adapter: nn.ModuleDict,
    features: torch.Tensor,
    limits_actor: torch.Tensor,
) -> torch.Tensor:
    raw = torch.cat(
        (
            adapter["sagittal"](features),
            adapter["lateral"](features),
        ),
        dim=-1,
    )
    return torch.tanh(raw) * limits_actor


# new change
def main() -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0.0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load(args.features, map_location="cpu", weights_only=False)
    features = dataset["support_adapter"].float()
    targets_actor = dataset["support_target_actor"].float()
    targets_rad = dataset["support_target_rad"].float()
    if features.ndim != 2 or features.shape[1] != 79:
        raise ValueError(f"Expected support features [N,79], got {features.shape}.")
    if targets_actor.shape != (features.shape[0], 10):
        raise ValueError(
            f"Expected support targets [N,10], got {targets_actor.shape}."
        )
    if tuple(dataset.get("support_action_ids", ())) != SUPPORT_ACTION_IDS:
        raise ValueError(
            "Dataset support action order does not match the V2 model: "
            f"{dataset.get('support_action_ids')}."
        )
    if abs(float(dataset.get("teacher_gain", -1.0)) - 0.10) > 1.0e-8:
        raise ValueError("SupervisedInit requires an online teacher gain of 0.10.")
    if abs(float(dataset.get("teacher_limit_rad", -1.0)) - 0.006) > 1.0e-8:
        raise ValueError("SupervisedInit requires a teacher limit of 0.006 rad.")

    if args.max_samples > 0 and features.shape[0] > args.max_samples:
        selected = torch.randperm(features.shape[0])[: args.max_samples]
        features = features[selected]
        targets_actor = targets_actor[selected]
        targets_rad = targets_rad[selected]
    permutation = torch.randperm(features.shape[0])
    split = int(0.9 * features.shape[0])
    train_indices = permutation[:split]
    validation_indices = permutation[split:]
    train_features = features[train_indices]
    train_targets = targets_actor[train_indices]
    validation_features = features[validation_indices].to(device)
    validation_targets_actor = targets_actor[validation_indices].to(device)
    validation_targets_rad = targets_rad[validation_indices].to(device)

    limits_rad = torch.tensor(
        SUPPORT_TARGET_LIMITS_RAD, device=device, dtype=torch.float32
    )
    limits_actor = limits_rad / 0.30
    adapter = make_support_adapter().to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.learning_rate)
    epoch_reports = []
    for epoch in range(args.epochs):
        order = torch.randperm(train_features.shape[0])
        train_loss_sum = 0.0
        train_count = 0
        for start in range(0, train_features.shape[0], args.batch_size):
            indices = order[start : start + args.batch_size]
            batch_features = train_features[indices].to(device)
            batch_targets = train_targets[indices].to(device)
            prediction = support_output_actor(
                adapter, batch_features, limits_actor
            )
            loss = torch.mean(torch.square(prediction - batch_targets))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * batch_features.shape[0]
            train_count += batch_features.shape[0]
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            with torch.inference_mode():
                validation_prediction_actor = support_output_actor(
                    adapter, validation_features, limits_actor
                )
                validation_error_rad = 0.30 * (
                    validation_prediction_actor - validation_targets_actor
                )
                report = {
                    "epoch": epoch + 1,
                    "train_mse_actor": train_loss_sum / max(train_count, 1),
                    "validation_rmse_rad": float(
                        torch.sqrt(torch.mean(validation_error_rad.square()))
                    ),
                    "validation_mae_rad": float(
                        validation_error_rad.abs().mean()
                    ),
                    "validation_max_rad": float(
                        validation_error_rad.abs().max()
                    ),
                }
                epoch_reports.append(report)
                print(report, flush=True)

    with torch.inference_mode():
        validation_prediction_actor = support_output_actor(
            adapter, validation_features, limits_actor
        )
        prediction_rad = 0.30 * validation_prediction_actor
        error_rad = prediction_rad - validation_targets_rad
        saturation = (
            validation_targets_rad.abs()
            >= float(dataset["teacher_limit_rad"]) - 1.0e-7
        ).float()
        per_joint = {}
        for index, action_id in enumerate(SUPPORT_ACTION_IDS):
            per_joint[str(action_id)] = {
                "target_mae_rad": float(
                    validation_targets_rad[:, index].abs().mean()
                ),
                "prediction_mae_rad": float(
                    prediction_rad[:, index].abs().mean()
                ),
                "validation_rmse_rad": float(
                    torch.sqrt(torch.mean(error_rad[:, index].square()))
                ),
                "validation_max_rad": float(error_rad[:, index].abs().max()),
                "target_saturation_fraction": float(
                    saturation[:, index].mean()
                ),
            }

    checkpoint = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    actor_state = checkpoint["actor_state_dict"]
    trained_state = copy.deepcopy(adapter).cpu().state_dict()
    for key, value in trained_state.items():
        actor_state[f"support_residual_adapter.{key}"] = value
    checkpoint["optimizer_state_dict"] = None
    checkpoint["infos"] = {
        **(checkpoint.get("infos") or {}),
        "stationary_support_control_v2_supervised_init": True,
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "feature_file": str(Path(args.features).resolve()),
        "feature_order": dataset.get("feature_order"),
        "support_action_ids": SUPPORT_ACTION_IDS,
        "support_target_limits_rad": SUPPORT_TARGET_LIMITS_RAD,
        "teacher_gain": float(dataset["teacher_gain"]),
        "teacher_limit_rad": float(dataset["teacher_limit_rad"]),
        "feature_seed": dataset.get("seed"),
        "feature_steps": dataset.get("steps"),
        "feature_num_envs": dataset.get("num_envs"),
        "sample_count": int(features.shape[0]),
        "epoch_reports": epoch_reports,
        "per_joint_validation": per_joint,
    }
    output = Path(args.output_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(
        {
            "output": str(output),
            "samples": int(features.shape[0]),
            "teacher_gain": float(dataset["teacher_gain"]),
            "teacher_limit_rad": float(dataset["teacher_limit_rad"]),
            "final": epoch_reports[-1],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
