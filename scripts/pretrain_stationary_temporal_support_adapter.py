#!/usr/bin/env python3
"""Offline train a time-context support adapter from strict paired DAgger labels."""

# new change
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn


# new change
class TemporalSupportAdapter(nn.Module):
    """98D state/memory/clock input to bounded 10DOF actor residual."""

    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        output_limit_actor: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_scale", feature_scale)
        self.register_buffer("output_limit_actor", output_limit_actor)
        self.network = nn.Sequential(
            nn.Linear(98, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 10),
        )
        # Strict zero initialization makes the adapter exactly equivalent to
        # frozen model_1375 before any offline fitting.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale
        return torch.tanh(self.network(normalized)) * self.output_limit_actor


# new change
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


# new change
def masked_indices(data: dict, split_code: int) -> torch.Tensor:
    mask = data["accepted"].bool() & data["valid_baseline"].bool()
    return torch.nonzero(mask & (data["split"].long() == split_code), as_tuple=False).squeeze(1)


# new change
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    if data.get("schema") != "StationaryTemporalSupportDatasetAggregationV1":
        raise RuntimeError("Expected the strict temporal dataset aggregation artifact.")
    if data.get("ppo_updates", True) or not data.get("strict_pairing", False):
        raise RuntimeError("Offline initialization requires strict inference-only pairs.")
    features = data["support_temporal_adapter"].float()
    targets = data["support_target_actor"].float()
    train_index = masked_indices(data, 0)
    validation_index = masked_indices(data, 1)
    test_index = masked_indices(data, 2)
    if min(train_index.numel(), validation_index.numel(), test_index.numel()) == 0:
        raise RuntimeError("Train/validation/test accepted splits must all be non-empty.")

    train_features = features[train_index]
    feature_mean = train_features.mean(0)
    feature_scale = train_features.std(0, unbiased=False).clamp_min(1.0e-5)
    # Keep the offline controller inside exactly the label envelope.  The tiny
    # floor avoids a numerically dead output when one joint was unused.
    output_limit_actor = targets[train_index].abs().amax(0).clamp_min(1.0e-6)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TemporalSupportAdapter(
        feature_mean.to(device), feature_scale.to(device), output_limit_actor.to(device)
    ).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    zero_probe = model(features[: min(128, len(features))].to(device))
    zero_max = float(zero_probe.abs().max())
    if zero_max != 0.0:
        raise RuntimeError(f"Zero-initialized adapter is not equivalent: {zero_max}.")
    zero_jit = torch.jit.script(copy.deepcopy(model).eval())
    zero_jit.save(str(args.output_dir / "temporal_support_adapter_zero.jit"))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    best_state = None
    best_validation = float("inf")
    history = []
    for epoch in range(args.epochs):
        permutation = train_index[torch.randperm(train_index.numel(), generator=generator)]
        model.train()
        epoch_loss = 0.0
        seen = 0
        for start in range(0, permutation.numel(), args.batch_size):
            index = permutation[start : start + args.batch_size]
            prediction = model(features[index].to(device))
            target = targets[index].to(device)
            loss = torch.nn.functional.smooth_l1_loss(
                prediction, target, beta=0.001
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss) * index.numel()
            seen += index.numel()
        model.eval()
        with torch.inference_mode():
            validation_prediction = model(features[validation_index].to(device))
            validation_rmse = float(
                torch.sqrt(
                    torch.mean(
                        (validation_prediction - targets[validation_index].to(device)).square()
                    )
                )
            )
        history.append({
            "epoch": epoch + 1,
            "train_smooth_l1": epoch_loss / max(seen, 1),
            "validation_rmse_actor": validation_rmse,
        })
        if validation_rmse < best_validation:
            best_validation = validation_rmse
            best_state = copy.deepcopy(model.state_dict())

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        predictions = {
            name: model(features[index].to(device)).cpu()
            for name, index in (
                ("train", train_index),
                ("validation", validation_index),
                ("test", test_index),
            )
        }
    indices = {"train": train_index, "validation": validation_index, "test": test_index}
    metrics = {}
    for name, prediction in predictions.items():
        error_rad = (prediction - targets[indices[name]]) * 0.30
        metrics[name] = {
            "samples": int(indices[name].numel()),
            "rmse_mrad": float(torch.sqrt(error_rad.square().mean()) * 1000.0),
            "mae_mrad": float(error_rad.abs().mean() * 1000.0),
            "max_mrad": float(error_rad.abs().max() * 1000.0),
        }

    checkpoint = {
        "schema": "StationaryTemporalSupportAdapterOfflineV1",
        "model_state_dict": model.state_dict(),
        "feature_order": data["feature_order"],
        "support_action_ids": data["support_action_ids"],
        "source_dataset": str(args.dataset.resolve()),
        "metrics": metrics,
        "zero_equivalence_max_actor": zero_max,
        "ppo_updates": False,
    }
    torch.save(checkpoint, args.output_dir / "temporal_support_adapter.pt")
    scripted = torch.jit.script(copy.deepcopy(model).eval())
    scripted.save(str(args.output_dir / "temporal_support_adapter.jit"))
    summary = {
        "schema": checkpoint["schema"],
        "zero_equivalence_max_actor": zero_max,
        "best_validation_rmse_actor": best_validation,
        "metrics": metrics,
        "output_limit_mrad": (output_limit_actor * 0.30 * 1000.0).tolist(),
        "epochs": args.epochs,
        "ppo_updates": False,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
