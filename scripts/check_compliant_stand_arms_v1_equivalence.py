"""Strict model_1375 equivalence check for CompliantStand ArmsV1.

# new change
This is a policy-level test and does not launch Isaac Sim.  The accepted
model_1375 weights are loaded into the new force-conditioned wrapper while all
new heads retain zero initialization.  Deterministic actions must therefore
match the permanent baseline exactly for arbitrary observations.
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
    return parser.parse_args()


def load_models_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "compliant_stand_arms_v1_models", path
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
    observation = TensorDict(
        {
            "policy": torch.randn(args.samples, 705, device=device),
            "left_arm_adapter": torch.randn(args.samples, 37, device=device),
            "right_arm_adapter": torch.randn(args.samples, 37, device=device),
            "online_compliance": torch.randn(args.samples, 54, device=device),
        },
        batch_size=[args.samples],
        device=device,
    )
    models = load_models_module(args.models_py)
    actor = models.StandCoordinationOnlineComplianceResidualAdapterMLPModel(
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
    allowed_tokens = (
        "residual_adapter",
        "adapter_limits",
        "adapter_action_ids",
        "adapter_projection",
        "analytic_error_gain",
    )
    disallowed_missing = [
        key for key in missing if not any(token in key for token in allowed_tokens)
    ]
    disallowed_unexpected = [
        key for key in unexpected if key != "distribution.std_param"
    ]
    actor.eval()
    legacy = torch.jit.load(str(args.legacy_jit), map_location=device).eval()
    with torch.inference_mode():
        candidate_action = actor(observation)
        legacy_action = legacy(observation["policy"])
        action_error = torch.abs(candidate_action - legacy_action)
        compliance = actor._compliance_output(observation["online_compliance"])
        left = actor._adapter_output(
            actor.left_arm_residual_adapter,
            observation["left_arm_adapter"],
            actor.left_arm_adapter_limits,
        )
        right = actor._adapter_output(
            actor.right_arm_residual_adapter,
            observation["right_arm_adapter"],
            actor.right_arm_adapter_limits,
        )
        exported_actor = torch.jit.script(actor.as_jit())
        exported_input = torch.cat(
            (
                observation["policy"],
                observation["left_arm_adapter"],
                observation["right_arm_adapter"],
                observation["online_compliance"],
            ),
            dim=-1,
        )
        export_error = torch.abs(exported_actor(exported_input) - candidate_action)
    report = {
        "checkpoint": str(args.checkpoint),
        "legacy_jit": str(args.legacy_jit),
        "samples": args.samples,
        "seed": args.seed,
        "device": str(device),
        "max_abs_action_difference": float(action_error.max()),
        "mean_abs_action_difference": float(action_error.mean()),
        "max_abs_jit_export_difference": float(export_error.max()),
        "max_abs_compliance_output": float(compliance.abs().max()),
        "max_abs_left_adapter_output": float(left.abs().max()),
        "max_abs_right_adapter_output": float(right.abs().max()),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "disallowed_missing_keys": disallowed_missing,
        "disallowed_unexpected_keys": disallowed_unexpected,
    }
    report["passed"] = bool(
        report["max_abs_action_difference"] <= 1.0e-7
        and report["max_abs_jit_export_difference"] <= 1.0e-7
        and report["max_abs_compliance_output"] == 0.0
        and report["max_abs_left_adapter_output"] == 0.0
        and report["max_abs_right_adapter_output"] == 0.0
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
