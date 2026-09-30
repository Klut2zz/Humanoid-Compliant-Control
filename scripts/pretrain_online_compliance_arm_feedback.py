"""Offline-distil the accepted analytic q/dq teacher into two 71D arm heads."""

# new change
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn


# new change
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", required=True)
parser.add_argument("--source_checkpoint", required=True)
parser.add_argument("--output_checkpoint", required=True)
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--learning_rate", type=float, default=1.0e-3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--target_limit_rad", type=float, default=0.020)
parser.add_argument("--action_scale", type=float, default=0.15)
args = parser.parse_args()


# new change
def make_head() -> nn.Sequential:
    head = nn.Sequential(nn.Linear(71, 96), nn.ELU(), nn.Linear(96, 7))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


# new change
def bounded_output(
    head: nn.Module, features: torch.Tensor, limit_actor: float
) -> torch.Tensor:
    return torch.tanh(head(features)) * limit_actor


# new change
def train_side(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    limit_actor: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    order = torch.randperm(features.shape[0])
    split = max(1, int(0.9 * features.shape[0]))
    train_ids = order[:split]
    validation_ids = order[split:]
    head = make_head().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.learning_rate)

    for _ in range(args.epochs):
        epoch_order = train_ids[torch.randperm(train_ids.shape[0])]
        for start in range(0, epoch_order.shape[0], args.batch_size):
            ids = epoch_order[start : start + args.batch_size]
            batch_features = features.index_select(0, ids).to(device)
            batch_targets = targets.index_select(0, ids).to(device)
            prediction = bounded_output(head, batch_features, limit_actor)
            loss = torch.mean((prediction - batch_targets) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    with torch.inference_mode():
        validation_features = features.index_select(0, validation_ids).to(device)
        validation_targets = targets.index_select(0, validation_ids).to(device)
        prediction = bounded_output(head, validation_features, limit_actor)
        error_actor = prediction - validation_targets
        error_rad = error_actor * args.action_scale
        summary = {
            "validation_samples": int(validation_ids.numel()),
            "rmse_mrad": float(1000.0 * torch.sqrt(torch.mean(error_rad**2))),
            "mae_mrad": float(1000.0 * torch.mean(torch.abs(error_rad))),
            "max_mrad": float(1000.0 * torch.max(torch.abs(error_rad))),
            "teacher_saturation_fraction": float(
                torch.mean(
                    (
                        torch.abs(validation_targets) >= limit_actor - 1.0e-5
                    ).float()
                )
            ),
        }
    return copy.deepcopy(head).cpu().state_dict(), summary


# new change
def main() -> None:
    if args.target_limit_rad <= 0.0 or args.action_scale <= 0.0:
        raise ValueError("target_limit_rad and action_scale must be positive.")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=False)
    features = dataset["online_compliance"].float()
    targets = dataset["arm_target_actor"].float()
    if features.ndim != 2 or features.shape[1] != 71:
        raise ValueError(f"Expected [N,71] features, got {tuple(features.shape)}")
    if targets.shape != (features.shape[0], 14):
        raise ValueError(
            f"Expected [N,14] arm targets, got {tuple(targets.shape)}"
        )
    limit_actor = args.target_limit_rad / args.action_scale
    if torch.max(torch.abs(targets)).item() > limit_actor + 1.0e-4:
        raise ValueError("Teacher targets exceed the configured physical limit.")

    trained = {}
    summaries = {}
    for side, target_slice in (("left_arm", slice(0, 7)), ("right_arm", slice(7, 14))):
        trained[side], summaries[side] = train_side(
            features,
            targets[:, target_slice],
            device=device,
            limit_actor=limit_actor,
        )

    checkpoint = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    actor_state = checkpoint["actor_state_dict"]
    for side, state in trained.items():
        prefix = f"compliance_residual_adapter.{side}."
        for key, value in state.items():
            actor_state[prefix + key] = value
    checkpoint["optimizer_state_dict"] = None
    checkpoint["infos"] = {
        **(checkpoint.get("infos") or {}),
        "online_compliance_arm_feedback_supervised": True,
        "teacher_dataset": str(Path(args.dataset).resolve()),
        "teacher_alpha": dataset.get("alpha"),
        "teacher_beta_s": dataset.get("beta_s"),
        "target_limit_rad": args.target_limit_rad,
        "samples": int(features.shape[0]),
        "summaries": summaries,
    }
    output = Path(args.output_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    summary_path = output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint": str(output.resolve()),
                "dataset": str(Path(args.dataset).resolve()),
                "samples": int(features.shape[0]),
                "device": str(device),
                "summaries": summaries,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(summary_path.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
