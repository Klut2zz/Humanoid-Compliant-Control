#!/usr/bin/env python3
"""Create a smooth bounded 12DOF teacher from all safe finite-difference probes."""

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
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--target_scale", type=float, default=1.0)
    parser.add_argument("--target_limit_rad", type=float, default=0.001)
    parser.add_argument("--minimum_failure_delay_steps", type=int, default=2)
    return parser.parse_args()


# new change
def main() -> None:
    args = parse_args()
    if args.temperature <= 0.0:
        raise ValueError("--temperature must be positive.")
    if not 0.0 < args.target_scale <= 1.0:
        raise ValueError("--target_scale must be in (0, 1].")
    if not 0.0 < args.target_limit_rad <= 0.002:
        raise ValueError("--target_limit_rad must be in (0, 0.002].")
    if args.minimum_failure_delay_steps < 1:
        raise ValueError("--minimum_failure_delay_steps must be positive.")

    manifest = json.loads((args.fd_dir / "manifest.json").read_text())
    if not manifest.get("completed", False) or manifest.get("ppo_updates", True):
        raise RuntimeError("Finite-difference matrix is incomplete or PPO-updated.")
    probe_items = sorted(manifest["probes"].items())
    if len(probe_items) != 2 * len(manifest["joint_names"]):
        raise RuntimeError("Finite-difference matrix does not contain all +/- probes.")

    data_list = []
    for name, info in probe_items:
        data = torch.load(info["paired"], map_location="cpu", weights_only=False)
        if not data.get("strict_pairing", False) or not data.get(
            "force_conditioned_pairing", False
        ):
            raise RuntimeError(f"Probe is not strict force-paired: {name}.")
        if data.get("ppo_updates", True):
            raise RuntimeError(f"Probe contains PPO updates: {name}.")
        if not bool(data["force_replay_match"].all()) or not bool(
            data["force_candidate_match"].all()
        ):
            raise RuntimeError(f"Force/IK replay mismatch: {name}.")
        if max(map(float, data["snapshot_deltas"].values()), default=0.0) != 0.0:
            raise RuntimeError(f"Snapshot mismatch: {name}.")
        if max(
            map(float, data["baseline_replay_deltas"].values()), default=0.0
        ) != 0.0:
            raise RuntimeError(f"Baseline replay mismatch: {name}.")
        data_list.append(data)

    reference = data_list[0]
    for data in data_list[1:]:
        if data["snapshot_fingerprint"] != reference["snapshot_fingerprint"]:
            raise RuntimeError("Probe snapshot fingerprints differ.")
        for key in ("failure", "first_failure_step", "score"):
            if not torch.equal(data["baseline"][key], reference["baseline"][key]):
                raise RuntimeError(f"Baseline {key} differs across probes.")

    target_stack = torch.stack(
        [data["candidate_target_rad"].float() for data in data_list], dim=1
    )
    score_delta = torch.stack(
        [data["paired_score_delta"].float() for data in data_list], dim=1
    )
    candidate_failure = torch.stack(
        [data["candidate"]["failure"].bool() for data in data_list], dim=1
    )
    candidate_first_failure = torch.stack(
        [
            data["candidate"]["first_failure_step"].long()
            for data in data_list
        ],
        dim=1,
    )
    valid_stack = torch.stack(
        [data["valid"].bool() for data in data_list], dim=1
    )
    common_valid = valid_stack.all(dim=1)
    baseline_failure = reference["baseline"]["failure"].bool()
    baseline_first_failure = reference["baseline"]["first_failure_step"].long()

    # new change
    # Candidate branches that introduce failure are excluded entirely.  For
    # an already failing baseline, retain only failure avoidance or a genuine
    # delay.  A zero-action option is always present in the softmax.
    preserves_success = (~baseline_failure[:, None]) & ~candidate_failure
    avoids_failure = baseline_failure[:, None] & ~candidate_failure
    delays_failure = (
        baseline_failure[:, None]
        & candidate_failure
        & (
            candidate_first_failure
            >= baseline_first_failure[:, None] + args.minimum_failure_delay_steps
        )
    )
    eligible = valid_stack & (preserves_success | avoids_failure | delays_failure)
    clipped_delta = score_delta.clamp(-1.0, 1.0)
    candidate_logits = torch.where(
        eligible,
        -clipped_delta / args.temperature,
        torch.full_like(clipped_delta, -float("inf")),
    )
    logits_with_zero = torch.cat(
        (torch.zeros(score_delta.shape[0], 1), candidate_logits), dim=1
    )
    weights_with_zero = torch.softmax(logits_with_zero, dim=1)
    candidate_weights = weights_with_zero[:, 1:]
    raw_target = (
        candidate_weights[:, :, None] * target_stack
    ).sum(dim=1) * args.target_scale
    support_target_rad = raw_target.clamp(
        min=-args.target_limit_rad, max=args.target_limit_rad
    )
    support_target_rad[~common_valid] = 0.0

    entropy = -(
        weights_with_zero
        * torch.log(weights_with_zero.clamp_min(1.0e-12))
    ).sum(dim=1)
    expected_paired_delta = (
        candidate_weights * score_delta
    ).sum(dim=1)
    expected_paired_delta[~common_valid] = 0.0
    force_features = reference["force_gate_features"]
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

    output = args.output or (args.fd_dir / "soft_response_teacher.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema": "ForceRecovery12DOFSoftResponseTeacherV1",
        "support_target_rad": support_target_rad,
        "raw_target_rad": raw_target,
        "valid": common_valid,
        "state_features": state_features,
        "motion_id": reference["motion_id"],
        "source_time_s": reference["source_time_s"],
        "candidate_weights": candidate_weights,
        "zero_weight": weights_with_zero[:, 0],
        "weight_entropy": entropy,
        "expected_paired_score_delta": expected_paired_delta,
        "support_joint_names": reference["support_joint_names"],
        "support_action_ids": reference["support_action_ids"],
        "snapshot_fingerprint": reference["snapshot_fingerprint"],
        "temperature": args.temperature,
        "target_scale": args.target_scale,
        "target_limit_rad": args.target_limit_rad,
        "strict_pairing": True,
        "ppo_updates": False,
    }
    torch.save(artifact, output)
    valid_target = support_target_rad[common_valid]
    summary = {
        "schema": artifact["schema"],
        "output": str(output.resolve()),
        "valid_envs": int(common_valid.sum()),
        "baseline_failures": int((baseline_failure & common_valid).sum()),
        "temperature": args.temperature,
        "target_scale": args.target_scale,
        "target_limit_rad": args.target_limit_rad,
        "mean_abs_target_mrad": float(valid_target.abs().mean() * 1000.0),
        "p95_max_target_mrad": float(
            torch.quantile(valid_target.abs().amax(dim=1), 0.95) * 1000.0
        ),
        "max_target_mrad": float(valid_target.abs().max() * 1000.0),
        "mean_active_dofs": float(
            (valid_target.abs() > 1.0e-5).sum(dim=1).float().mean()
        ),
        "mean_zero_weight": float(weights_with_zero[common_valid, 0].mean()),
        "mean_weight_entropy": float(entropy[common_valid].mean()),
        "expected_paired_score_delta_mean": float(
            expected_paired_delta[common_valid].mean()
        ),
        "strict_pairing": True,
        "ppo_updates": False,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
