#!/usr/bin/env python3
"""Pad the legacy critic input with zero columns for V7 exact equivalence."""

# new change
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


# new change
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_checkpoint", type=Path, required=True)
    parser.add_argument("--output_checkpoint", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--expected_old_dim", type=int, default=867)
    parser.add_argument("--expected_new_dim", type=int, default=927)
    return parser.parse_args()


# new change
def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    critic_state = {
        key: value.detach().cpu().clone()
        for key, value in checkpoint["critic_state_dict"].items()
    }
    first_key = "mlp.0.weight"
    old_weight = critic_state[first_key]
    if tuple(old_weight.shape) != (512, args.expected_old_dim):
        raise RuntimeError(
            f"Unexpected legacy critic input: {tuple(old_weight.shape)}; "
            f"expected (512, {args.expected_old_dim})."
        )
    if args.expected_new_dim <= args.expected_old_dim:
        raise ValueError("New critic dimension must be greater than the legacy dimension.")
    padded_weight = torch.zeros(
        old_weight.shape[0], args.expected_new_dim, dtype=old_weight.dtype
    )
    padded_weight[:, : args.expected_old_dim] = old_weight
    critic_state[first_key] = padded_weight

    # Random arbitrary V7-only features must have exactly zero influence at
    # initialization.  This checks both the column order assumption and the
    # padding operation before a checkpoint is written.
    generator = torch.Generator().manual_seed(42)
    old_obs = torch.randn(4096, args.expected_old_dim, generator=generator)
    extra_obs = torch.randn(
        4096, args.expected_new_dim - args.expected_old_dim, generator=generator
    )
    old_pre_activation = old_obs @ old_weight.T
    new_pre_activation = torch.cat((old_obs, extra_obs), dim=1) @ padded_weight.T
    equivalence_max_abs = float((old_pre_activation - new_pre_activation).abs().max())
    # BLAS may choose a different accumulation path for the wider matrix even
    # though every appended column is exactly zero.  Keep a sub-micro-unit
    # numerical tolerance while separately checking the padded block itself.
    padded_block_max_abs = float(
        padded_weight[:, args.expected_old_dim :].abs().max()
    )
    if equivalence_max_abs > 1.0e-6 or padded_block_max_abs != 0.0:
        raise RuntimeError(f"Critic zero-padding is not exact: {equivalence_max_abs}")

    output = dict(checkpoint)
    output["critic_state_dict"] = critic_state
    output["optimizer_state_dict"] = None
    provenance = {
        "stationary_critic_aware_v7_checkpoint": True,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "old_critic_input_dim": args.expected_old_dim,
        "new_critic_input_dim": args.expected_new_dim,
        "new_critic_columns": args.expected_new_dim - args.expected_old_dim,
        "critic_initial_equivalence_max_abs": equivalence_max_abs,
        "new_critic_block_max_abs": padded_block_max_abs,
        "actor_state_unchanged": True,
        "ppo_updates": False,
    }
    output["infos"] = {**(checkpoint.get("infos") or {}), **provenance}
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output_checkpoint)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2), flush=True)


# new change
if __name__ == "__main__":
    main()
