"""Offline-distill bounded analytic arm feedback into the structured heads."""

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
parser.add_argument("--alpha", type=float, default=0.25)
# new change
parser.add_argument(
    "--motion_conditioned_alpha",
    type=float,
    default=None,
    help="Optional replacement alpha on --motion_conditioned_ids only.",
)
# new change
parser.add_argument(
    "--motion_conditioned_ids",
    nargs="*",
    type=int,
    default=(),
    help="Motion ids that use --motion_conditioned_alpha.",
)
# new change
parser.add_argument(
    "--initialize_from_source_adapters",
    action="store_true",
    help="Initialize all structured heads from the source checkpoint.",
)
# new change
parser.add_argument(
    "--proximal_only",
    action="store_true",
    help="Optimize only shoulder and elbow heads; preserve wrist heads exactly.",
)
# new change
parser.add_argument(
    "--distill_increment_from_source",
    action="store_true",
    help=(
        "Use the loaded source adapter output as the exact teacher target and "
        "add --motion_conditioned_alpha feedback only on conditioned samples."
    ),
)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--learning_rate", type=float, default=1.0e-3)
parser.add_argument("--seed", type=int, default=42)
# new change
parser.add_argument(
    "--shoulder_elbow_limit_rad",
    type=float,
    default=0.03,
    help="Target-space correction limit for shoulder and elbow joints.",
)
# new change
parser.add_argument(
    "--wrist_limit_rad",
    type=float,
    default=0.0375,
    help="Target-space correction limit for wrist joints.",
)
# new change
parser.add_argument(
    "--max_samples",
    type=int,
    default=0,
    help="Optional deterministic sample cap; zero uses the complete dataset.",
)
args = parser.parse_args()


# new change
def make_adapter() -> nn.ModuleDict:
    def head(hidden: int, output: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(37, hidden), nn.ELU(), nn.Linear(hidden, output))

    return nn.ModuleDict(
        {
            "shoulder": head(64, 3),
            "elbow": head(32, 1),
            "wrist": head(64, 3),
        }
    )


# new change
def adapter_output(
    adapter: nn.ModuleDict, features: torch.Tensor, limits: torch.Tensor
) -> torch.Tensor:
    raw = torch.cat(
        (
            adapter["shoulder"](features),
            adapter["elbow"](features),
            adapter["wrist"](features),
        ),
        dim=-1,
    )
    return torch.tanh(raw) * limits


# new change
def main() -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load(args.features, map_location="cpu", weights_only=False)
    # new change
    checkpoint = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    actor_state = checkpoint["actor_state_dict"]
    if args.motion_conditioned_alpha is not None and not args.motion_conditioned_ids:
        raise ValueError(
            "--motion_conditioned_alpha requires at least one --motion_conditioned_ids value."
        )
    if args.motion_conditioned_ids and "motion_id" not in dataset:
        raise ValueError(
            "The feature dataset has no motion_id field; recollect it with the updated "
            "smoke_general_tracking.py before motion-conditioned distillation."
        )
    # new change
    if args.shoulder_elbow_limit_rad <= 0.0 or args.wrist_limit_rad <= 0.0:
        raise ValueError("Adapter target limits must be positive.")
    limits_rad = torch.tensor(
        [
            args.shoulder_elbow_limit_rad,
            args.shoulder_elbow_limit_rad,
            args.shoulder_elbow_limit_rad,
            args.shoulder_elbow_limit_rad,
            args.wrist_limit_rad,
            args.wrist_limit_rad,
            args.wrist_limit_rad,
        ],
        device=device,
    )
    limits = limits_rad / 0.15
    trained = {}
    summaries = {}
    for side in ("left", "right"):
        features = dataset[f"{side}_arm_adapter"].float()
        # new change
        motion_ids = dataset.get(
            "motion_id", torch.full((features.shape[0],), -1, dtype=torch.long)
        ).long()
        # new change
        if args.max_samples > 0 and features.shape[0] > args.max_samples:
            sample_order = torch.randperm(features.shape[0])[: args.max_samples]
            features = features[sample_order]
            motion_ids = motion_ids[sample_order]
        permutation = torch.randperm(features.shape[0])
        split = int(0.9 * features.shape[0])
        train_features = features[permutation[:split]]
        validation_features = features[permutation[split:]]
        # new change
        train_motion_ids = motion_ids[permutation[:split]]
        validation_motion_ids = motion_ids[permutation[split:]]
        adapter = make_adapter().to(device)
        # new change
        if args.initialize_from_source_adapters:
            prefix = f"{side}_arm_residual_adapter."
            source_adapter_state = {
                key.removeprefix(prefix): value
                for key, value in actor_state.items()
                if key.startswith(prefix)
            }
            adapter.load_state_dict(source_adapter_state, strict=True)
        # new change
        teacher_adapter = None
        if args.distill_increment_from_source:
            if not args.initialize_from_source_adapters:
                raise ValueError(
                    "--distill_increment_from_source requires "
                    "--initialize_from_source_adapters."
                )
            if args.motion_conditioned_alpha is None:
                raise ValueError(
                    "Incremental distillation requires --motion_conditioned_alpha "
                    "as the additional feedback gain."
                )
            teacher_adapter = copy.deepcopy(adapter).eval()
            for parameter in teacher_adapter.parameters():
                parameter.requires_grad = False
        if args.proximal_only:
            for parameter in adapter["wrist"].parameters():
                parameter.requires_grad = False
        optimizer = torch.optim.Adam(
            [parameter for parameter in adapter.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
        )
        for _ in range(args.epochs):
            order = torch.randperm(train_features.shape[0])
            for start in range(0, train_features.shape[0], args.batch_size):
                batch_indices = order[start : start + args.batch_size]
                batch = train_features[batch_indices].to(device)
                # new change
                batch_alpha = torch.full(
                    (batch.shape[0], 1), args.alpha, device=device, dtype=batch.dtype
                )
                if args.motion_conditioned_alpha is not None:
                    batch_motion_ids = train_motion_ids[batch_indices].to(device)
                    conditioned = torch.zeros_like(batch_motion_ids, dtype=torch.bool)
                    for motion_id in args.motion_conditioned_ids:
                        conditioned |= batch_motion_ids == motion_id
                    batch_alpha[conditioned] = args.motion_conditioned_alpha
                # new change
                if teacher_adapter is not None:
                    with torch.no_grad():
                        target = adapter_output(teacher_adapter, batch, limits)
                    extra_feedback = (
                        -args.motion_conditioned_alpha * batch[:, 14:21] / 0.15
                    )
                    target = target.clone()
                    target[conditioned, :4] += extra_feedback[conditioned, :4]
                    target = torch.maximum(torch.minimum(target, limits), -limits)
                else:
                    target = torch.maximum(
                        torch.minimum(-batch_alpha * batch[:, 14:21] / 0.15, limits),
                        -limits,
                    )
                prediction = adapter_output(adapter, batch, limits)
                # new change
                trained_width = 4 if args.proximal_only else 7
                loss = torch.mean(
                    (prediction[:, :trained_width] - target[:, :trained_width]) ** 2
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        with torch.inference_mode():
            validation_features = validation_features.to(device)
            # new change
            validation_alpha = torch.full(
                (validation_features.shape[0], 1),
                args.alpha,
                device=device,
                dtype=validation_features.dtype,
            )
            if args.motion_conditioned_alpha is not None:
                validation_motion_ids = validation_motion_ids.to(device)
                conditioned = torch.zeros_like(validation_motion_ids, dtype=torch.bool)
                for motion_id in args.motion_conditioned_ids:
                    conditioned |= validation_motion_ids == motion_id
                validation_alpha[conditioned] = args.motion_conditioned_alpha
            # new change
            if teacher_adapter is not None:
                target = adapter_output(teacher_adapter, validation_features, limits)
                extra_feedback = (
                    -args.motion_conditioned_alpha
                    * validation_features[:, 14:21]
                    / 0.15
                )
                target = target.clone()
                target[conditioned, :4] += extra_feedback[conditioned, :4]
                target = torch.maximum(torch.minimum(target, limits), -limits)
            else:
                target = torch.maximum(
                    torch.minimum(
                        -validation_alpha * validation_features[:, 14:21] / 0.15,
                        limits,
                    ),
                    -limits,
                )
            prediction = adapter_output(adapter, validation_features, limits)
            # new change
            trained_width = 4 if args.proximal_only else 7
            error = prediction[:, :trained_width] - target[:, :trained_width]
            summaries[side] = {
                "validation_rmse_actor": float(torch.sqrt(torch.mean(error**2))),
                "validation_max_actor": float(torch.max(torch.abs(error))),
                "target_saturation_fraction": float(
                    torch.mean(
                        (
                            torch.abs(target[:, :trained_width])
                            >= limits[:trained_width] - 1.0e-6
                        ).float()
                    )
                ),
            }
        trained[side] = copy.deepcopy(adapter).cpu().state_dict()

    for side in ("left", "right"):
        prefix = f"{side}_arm_residual_adapter."
        for key, value in trained[side].items():
            actor_state[prefix + key] = value
    checkpoint["optimizer_state_dict"] = None
    checkpoint["infos"] = {
        **(checkpoint.get("infos") or {}),
        "general_tracking_supervised_arm_adapter": True,
        "alpha": args.alpha,
        # new change
        "motion_conditioned_alpha": args.motion_conditioned_alpha,
        "motion_conditioned_ids": list(args.motion_conditioned_ids),
        "initialize_from_source_adapters": args.initialize_from_source_adapters,
        "proximal_only": args.proximal_only,
        "distill_increment_from_source": args.distill_increment_from_source,
        # new change
        "shoulder_elbow_limit_rad": args.shoulder_elbow_limit_rad,
        "wrist_limit_rad": args.wrist_limit_rad,
        "feature_order": dataset.get("feature_order"),
        "feature_seed": dataset.get("seed"),
        "feature_steps": dataset.get("steps"),
        "feature_num_envs": dataset.get("num_envs"),
        "summaries": summaries,
    }
    output = Path(args.output_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print({"output": str(output), "samples": features.shape[0], **summaries})


if __name__ == "__main__":
    main()
