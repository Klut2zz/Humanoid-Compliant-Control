"""Strict policy-level equivalence check for StationaryFoundationV2.

# new change
This diagnostic avoids Isaac Kit lifecycle/rendering effects.  It loads the
frozen model_1375 actor weights into the new zero-initialized structured actor
and compares deterministic actions against the accepted legacy JIT policy.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from tensordict import TensorDict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--legacy_jit", type=Path, required=True)
    parser.add_argument("--models_py", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    # new change
    parser.add_argument(
        "--model_class",
        default="StandCoordinationStationaryFoundationV2MLPModel",
    )
    return parser.parse_args()


def load_models_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "stationary_foundation_v2_models", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import models module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive.")
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    observation = TensorDict(
        {
            "policy": torch.randn(args.samples, 705, device=device, dtype=dtype),
            "left_arm_adapter": torch.randn(
                args.samples, 37, device=device, dtype=dtype
            ),
            "right_arm_adapter": torch.randn(
                args.samples, 37, device=device, dtype=dtype
            ),
            "waist_adapter": torch.randn(
                args.samples, 25, device=device, dtype=dtype
            ),
            "support_adapter": torch.randn(
                args.samples, 87, device=device, dtype=dtype
            ),
        },
        batch_size=[args.samples],
        device=device,
    )
    models = load_models_module(args.models_py)
    # new change
    model_class = getattr(models, args.model_class)
    actor = model_class(
        obs=observation,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=29,
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
    ).to(device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    missing, unexpected = actor.load_state_dict(
        checkpoint["actor_state_dict"], strict=False
    )
    disallowed_missing = [
        key
        for key in missing
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
    disallowed_unexpected = [
        key for key in unexpected if key != "distribution.std_param"
    ]
    actor.eval()
    legacy = torch.jit.load(str(args.legacy_jit), map_location=device)
    legacy.eval()
    with torch.inference_mode():
        candidate_action = actor(observation)
        legacy_action = legacy(observation["policy"])
        action_error = torch.abs(candidate_action - legacy_action)
        # new change
        exported_actor = torch.jit.script(actor.as_jit())
        exported_input = torch.cat(
            (
                observation["policy"],
                observation["left_arm_adapter"],
                observation["right_arm_adapter"],
                observation["waist_adapter"],
                observation["support_adapter"],
            ),
            dim=-1,
        )
        exported_action = exported_actor(exported_input)
        export_error = torch.abs(exported_action - candidate_action)
        support_output = actor._support_output(observation["support_adapter"])
        left_output = actor._adapter_output(
            actor.left_arm_residual_adapter,
            observation["left_arm_adapter"],
            actor.left_arm_adapter_limits,
        )
        right_output = actor._adapter_output(
            actor.right_arm_residual_adapter,
            observation["right_arm_adapter"],
            actor.right_arm_adapter_limits,
        )
        waist_output = actor._waist_output(observation["waist_adapter"])
    report = {
        "checkpoint": str(args.checkpoint),
        "legacy_jit": str(args.legacy_jit),
        "samples": args.samples,
        "seed": args.seed,
        "model_class": args.model_class,
        "device": str(device),
        "max_abs_action_difference": float(action_error.max().item()),
        "mean_abs_action_difference": float(action_error.mean().item()),
        "max_abs_jit_export_difference": float(export_error.max().item()),
        "max_abs_support_output": float(support_output.abs().max().item()),
        "max_abs_left_adapter_output": float(left_output.abs().max().item()),
        "max_abs_right_adapter_output": float(right_output.abs().max().item()),
        "max_abs_waist_adapter_output": float(waist_output.abs().max().item()),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "disallowed_missing_keys": disallowed_missing,
        "disallowed_unexpected_keys": disallowed_unexpected,
    }
    report["passed"] = bool(
        report["max_abs_action_difference"] <= 1.0e-7
        and report["max_abs_jit_export_difference"] <= 1.0e-7
        and report["max_abs_support_output"] == 0.0
        and report["max_abs_left_adapter_output"] == 0.0
        and report["max_abs_right_adapter_output"] == 0.0
        and report["max_abs_waist_adapter_output"] == 0.0
        and not disallowed_missing
        and not disallowed_unexpected
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
