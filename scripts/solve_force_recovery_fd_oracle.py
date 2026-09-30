#!/usr/bin/env python3
"""Solve a safe per-snapshot single-axis oracle from strict paired 12DOF probes."""

# new change
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


# new change
PROJECT_ROOT = Path(__file__).absolute().parents[1]
DEFAULT_FD_DIR = PROJECT_ROOT / (
    "logs/stand_coordination_v1/compliant_stand_smooth_4to7_stress/"
    "recovery_12dof_fd_v1/seed0_event1001_phase000_waist"
)


# new change
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd_dir", type=Path, default=DEFAULT_FD_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <fd_dir>/oracle_single_axis_teacher.pt.",
    )
    parser.add_argument(
        "--score_improvement_margin",
        type=float,
        default=0.002,
        help="Minimum paired score decrease required unless a failure is avoided.",
    )
    parser.add_argument(
        "--minimum_failure_delay_steps",
        type=int,
        default=2,
        help="Minimum delay required when both baseline and candidate fail.",
    )
    return parser.parse_args()


# new change
def _require_equal(left: object, right: object, label: str) -> None:
    if torch.is_tensor(left) and torch.is_tensor(right):
        equal = left.shape == right.shape and torch.equal(left, right)
    else:
        equal = left == right
    if not equal:
        raise RuntimeError(f"Strict-pair invariant differs across probes: {label}.")


# new change
def _force_snapshot_features(data: dict[str, object]) -> torch.Tensor:
    trajectory = data.get("force_context_trajectory", {})
    feature = trajectory.get("force_context_force_gate_features_trajectory")
    if feature is None:
        raise RuntimeError("Finite-difference artifact lacks force gate trajectory.")
    if feature.ndim != 3 or feature.shape[1] < 1:
        raise RuntimeError(f"Unexpected force feature shape: {tuple(feature.shape)}.")
    return feature[:, 0].float()


# new change
def main() -> None:
    args = parse_args()
    if args.score_improvement_margin < 0.0:
        raise ValueError("--score_improvement_margin must be non-negative.")
    if args.minimum_failure_delay_steps < 1:
        raise ValueError("--minimum_failure_delay_steps must be positive.")

    manifest_path = args.fd_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("completed", False):
        raise RuntimeError(f"Finite-difference matrix is incomplete: {manifest_path}.")
    if manifest.get("ppo_updates", True):
        raise RuntimeError("Refusing data marked as PPO-updated.")

    probe_items = sorted(manifest["probes"].items())
    expected_probe_count = 2 * len(manifest["joint_names"])
    if len(probe_items) != expected_probe_count:
        raise RuntimeError(
            f"Expected {expected_probe_count} probes, found {len(probe_items)}."
        )

    reference: dict[str, object] | None = None
    probe_names: list[str] = []
    targets: list[torch.Tensor] = []
    score_deltas: list[torch.Tensor] = []
    candidate_failures: list[torch.Tensor] = []
    candidate_first_failure_steps: list[torch.Tensor] = []
    all_valid: list[torch.Tensor] = []
    paired_paths: list[str] = []

    for probe_name, probe_info in probe_items:
        paired_path = Path(probe_info["paired"])
        data = torch.load(paired_path, map_location="cpu", weights_only=False)
        if not data.get("strict_pairing", False):
            raise RuntimeError(f"Probe is not strictly paired: {paired_path}.")
        if not data.get("force_conditioned_pairing", False):
            raise RuntimeError(f"Probe lacks strict force matching: {paired_path}.")
        if data.get("ppo_updates", True):
            raise RuntimeError(f"Probe unexpectedly contains PPO updates: {paired_path}.")
        if not bool(data["force_replay_match"].all()):
            raise RuntimeError(f"Force replay mismatch: {paired_path}.")
        if not bool(data["force_candidate_match"].all()):
            raise RuntimeError(f"Candidate force mismatch: {paired_path}.")
        if max(map(float, data["snapshot_deltas"].values()), default=0.0) != 0.0:
            raise RuntimeError(f"Candidate snapshot mismatch: {paired_path}.")
        if max(
            map(float, data["baseline_replay_deltas"].values()), default=0.0
        ) != 0.0:
            raise RuntimeError(f"Baseline replay mismatch: {paired_path}.")
        if int(data["baseline_replay_failure_mismatch"]) != 0:
            raise RuntimeError(f"Baseline failure replay mismatch: {paired_path}.")

        if reference is None:
            reference = data
        else:
            for key in (
                "snapshot_fingerprint",
                "support_action_ids",
                "support_joint_names",
                "motion_id",
                "source_time_s",
                "support_recovery_features",
                "support_temporal_features",
            ):
                _require_equal(reference[key], data[key], key)
            for key in ("failure", "first_failure_step", "score"):
                _require_equal(reference["baseline"][key], data["baseline"][key], key)

        probe_names.append(probe_name)
        targets.append(data["candidate_target_rad"].float())
        score_deltas.append(data["paired_score_delta"].float())
        candidate_failures.append(data["candidate"]["failure"].bool())
        candidate_first_failure_steps.append(
            data["candidate"]["first_failure_step"].long()
        )
        all_valid.append(data["valid"].bool())
        paired_paths.append(str(paired_path.resolve()))

    assert reference is not None
    target_stack = torch.stack(targets, dim=1)
    score_delta_stack = torch.stack(score_deltas, dim=1)
    failure_stack = torch.stack(candidate_failures, dim=1)
    first_failure_stack = torch.stack(candidate_first_failure_steps, dim=1)
    valid_stack = torch.stack(all_valid, dim=1)
    num_envs, num_probes, num_joints = target_stack.shape
    # new change
    # A warmup-stage failure can make an environment invalid before the
    # reconstructed intervention snapshot.  Keep its target at zero and
    # exclude it from every oracle statistic instead of counting the stale
    # baseline failure in the report.
    common_valid = valid_stack.all(dim=1)

    baseline_failure = reference["baseline"]["failure"].bool()
    baseline_first_failure = reference["baseline"]["first_failure_step"].long()
    baseline_score = reference["baseline"]["score"].float()

    # new change
    # A candidate must never introduce failure into a baseline-success state.
    # For an already failing state it must either avoid failure, or postpone
    # failure by a measurable amount.  This rejects apparent score gains that
    # merely trade away long-horizon safety.
    baseline_success = ~baseline_failure[:, None]
    avoids_failure = baseline_failure[:, None] & ~failure_stack
    delays_failure = (
        baseline_failure[:, None]
        & failure_stack
        & (
            first_failure_stack
            >= baseline_first_failure[:, None] + args.minimum_failure_delay_steps
        )
    )
    preserves_success = baseline_success & ~failure_stack
    safety_eligible = valid_stack & (preserves_success | avoids_failure | delays_failure)
    score_eligible = score_delta_stack < -args.score_improvement_margin
    eligible = safety_eligible & (avoids_failure | score_eligible)

    masked_delta = torch.where(
        eligible,
        score_delta_stack,
        torch.full_like(score_delta_stack, float("inf")),
    )
    best_delta, best_probe = masked_delta.min(dim=1)
    selected = torch.isfinite(best_delta)
    support_target_rad = torch.zeros((num_envs, num_joints), dtype=torch.float32)
    selected_candidate_failure = baseline_failure.clone()
    selected_first_failure = baseline_first_failure.clone()
    selected_score_delta = torch.zeros(num_envs, dtype=torch.float32)
    selected_probe_index = torch.full((num_envs,), -1, dtype=torch.long)
    env_ids = torch.arange(num_envs)
    if selected.any():
        selected_envs = env_ids[selected]
        selected_probes = best_probe[selected]
        support_target_rad[selected] = target_stack[selected_envs, selected_probes]
        selected_candidate_failure[selected] = failure_stack[
            selected_envs, selected_probes
        ]
        selected_first_failure[selected] = first_failure_stack[
            selected_envs, selected_probes
        ]
        selected_score_delta[selected] = score_delta_stack[
            selected_envs, selected_probes
        ]
        selected_probe_index[selected] = selected_probes

    if float(support_target_rad.abs().max()) > float(manifest["amplitude_rad"]) + 1e-9:
        raise RuntimeError("Solved oracle exceeds finite-difference amplitude.")

    selection_counts = {
        probe_names[index]: int((selected_probe_index == index).sum())
        for index in range(num_probes)
        if int((selected_probe_index == index).sum()) > 0
    }
    zero_count = int((selected_probe_index < 0).sum())
    predicted_failure_count = int(
        (selected_candidate_failure & common_valid).sum()
    )
    predicted_avoided = int(
        (baseline_failure & ~selected_candidate_failure & common_valid).sum()
    )
    predicted_new = int(
        (~baseline_failure & selected_candidate_failure & common_valid).sum()
    )
    force_features = _force_snapshot_features(reference)
    state_features = torch.cat(
        (
            reference["support_recovery_features"].float(),
            reference["support_temporal_features"].float(),
            force_features,
        ),
        dim=-1,
    )

    output = args.output or (args.fd_dir / "oracle_single_axis_teacher.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema": "ForceRecovery12DOFSingleAxisOracleV1",
        "support_target_rad": support_target_rad,
        "support_joint_names": tuple(manifest["joint_names"]),
        "support_action_ids": tuple(manifest["action_ids"]),
        "support_action_scales_rad": tuple(manifest["action_scales_rad"]),
        "teacher_limit_rad": float(manifest["amplitude_rad"]),
        "selected_probe_index": selected_probe_index,
        "selected_probe_names": tuple(probe_names),
        "selected_score_delta": selected_score_delta,
        "selected_candidate_failure": selected_candidate_failure,
        "selected_first_failure_step": selected_first_failure,
        "baseline_failure": baseline_failure,
        "baseline_first_failure_step": baseline_first_failure,
        "baseline_score": baseline_score,
        # new change
        "valid": common_valid,
        "state_features": state_features,
        "support_recovery_features": reference["support_recovery_features"],
        "support_temporal_features": reference["support_temporal_features"],
        "force_gate_features": force_features,
        "previous_support_residual_rad": reference.get(
            "previous_support_residual_rad"
        ),
        "steps_since_residual_update": reference.get(
            "steps_since_residual_update"
        ),
        "motion_id": reference["motion_id"],
        "source_time_s": reference["source_time_s"],
        "snapshot_fingerprint": reference["snapshot_fingerprint"],
        "strict_pairing": True,
        "ppo_updates": False,
        "selection_policy": {
            "score_improvement_margin": args.score_improvement_margin,
            "minimum_failure_delay_steps": args.minimum_failure_delay_steps,
            "never_introduce_failure": True,
            "prefer_failure_avoidance": True,
        },
        "paired_paths": tuple(paired_paths),
    }
    torch.save(artifact, output)

    summary = {
        "schema": artifact["schema"],
        "fd_dir": str(args.fd_dir.resolve()),
        "output": str(output.resolve()),
        "num_envs": num_envs,
        "num_probes": num_probes,
        "state_feature_dim": int(state_features.shape[-1]),
        "valid_envs": int(common_valid.sum()),
        "baseline_failures": int((baseline_failure & common_valid).sum()),
        "predicted_oracle_failures": predicted_failure_count,
        "predicted_failures_avoided": predicted_avoided,
        "predicted_new_failures": predicted_new,
        "selected_nonzero": int((selected & common_valid).sum()),
        "selected_zero": int((~selected & common_valid).sum()),
        "selected_score_delta_mean_all": float(
            selected_score_delta[common_valid].mean()
        ),
        "selected_score_delta_mean_nonzero": (
            float(selected_score_delta[selected & common_valid].mean())
            if (selected & common_valid).any()
            else 0.0
        ),
        "selection_counts": selection_counts,
        "strict_pairing": True,
        "ppo_updates": False,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
