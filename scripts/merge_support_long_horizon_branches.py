"""Verify deterministic replay and merge exact baseline/candidate branch artifacts."""

# new change
from __future__ import annotations

import argparse
import pathlib

import torch


# new change
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline", required=True)
parser.add_argument("--baseline_replay", required=True)
parser.add_argument("--candidate", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--replay_tol", type=float, default=1.0e-7)
# new change
parser.add_argument(
    "--require_force_context",
    action="store_true",
    help=(
        "Reject any sample whose sampled/applied force event or IK acceptance "
        "does not match across baseline replay and candidate branches."
    ),
)
args = parser.parse_args()


# new change
def _max_delta(left, right, key):
    return float((left[key] - right[key]).abs().max())


# new change
def _per_env_equal(left, right, tolerance):
    if left.shape != right.shape:
        raise RuntimeError(
            f"Force context shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}"
        )
    flattened_left = left.reshape(left.shape[0], -1)
    flattened_right = right.reshape(right.shape[0], -1)
    if left.dtype == torch.bool or not left.dtype.is_floating_point:
        return torch.all(flattened_left == flattened_right, dim=1)
    return torch.amax(
        torch.abs(flattened_left - flattened_right), dim=1
    ) <= tolerance


# new change
def main():
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=False)
    replay = torch.load(args.baseline_replay, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    if baseline["branch"] != "baseline" or replay["branch"] != "baseline":
        raise ValueError("Both baseline inputs must be baseline artifacts.")
    if candidate["branch"] != "candidate":
        raise ValueError("Candidate input is not a candidate artifact.")
    # new change
    if baseline.get("candidate_mode") != replay.get("candidate_mode"):
        raise RuntimeError(
            "Baseline and baseline-replay metadata modes differ."
        )
    # new change
    # Candidate mode does not affect either baseline action.  Allow a newer
    # closed-loop candidate to reuse an older strictly audited baseline pair;
    # snapshot equality and full force/IK trajectory matching are still
    # mandatory below.
    # new change
    if args.require_force_context:
        for name, artifact in (
            ("baseline", baseline),
            ("baseline_replay", replay),
            ("candidate", candidate),
        ):
            if not artifact.get("force_conditioned_pairing", False):
                raise RuntimeError(
                    f"{name} was not collected with --require_force_context."
                )
        if not (
            baseline["force_gate_feature_names"]
            == replay["force_gate_feature_names"]
            == candidate["force_gate_feature_names"]
        ):
            raise RuntimeError("Force gate feature layouts differ across branches.")
    fingerprints = {
        baseline["snapshot_fingerprint"],
        replay["snapshot_fingerprint"],
        candidate["snapshot_fingerprint"],
    }
    if len(fingerprints) != 1:
        raise RuntimeError(f"Snapshot fingerprints differ: {sorted(fingerprints)}")

    snapshot_deltas = {
        key: _max_delta(baseline["snapshot"], candidate["snapshot"], key)
        for key in baseline["snapshot"]
        # new change: intervention metadata is deliberately different while
        # the reconstructed physical simulator state must remain identical.
        if key != "candidate_target_rad"
        if baseline["snapshot"][key].dtype != torch.bool
    }
    replay_deltas = {
        key: _max_delta(baseline["result"], replay["result"], key)
        for key in (
            "cost_trajectory",
            "root_state_trajectory",
            "joint_state_trajectory",
            "action_trajectory",
            "pre_failure_cost_mean",
            "score",
            # new change
            "com_support_xy_distance_m_trajectory",
            "torso_tilt_rad_trajectory",
            # new change
            "support_feature_trajectory",
            # new change
            "support_foot_position_w_trajectory",
            "whole_body_com_w_trajectory",
        )
        if key in baseline["result"] and key in replay["result"]
    }
    replay_failure_mismatch = int(
        (baseline["result"]["failure"] != replay["result"]["failure"])
        .sum()
        .item()
    )
    # new change
    force_context_keys = sorted(
        key
        for key in baseline["result"]
        if key.startswith("force_context_") and key.endswith("_trajectory")
    )
    force_replay_match = torch.ones(
        baseline["valid"].shape[0], dtype=torch.bool
    )
    force_candidate_match = torch.ones_like(force_replay_match)
    force_replay_mismatch_by_key = {}
    force_candidate_mismatch_by_key = {}
    if args.require_force_context:
        if not force_context_keys:
            raise RuntimeError("No force-context trajectories were recorded.")
        missing_replay = [key for key in force_context_keys if key not in replay["result"]]
        missing_candidate = [
            key for key in force_context_keys if key not in candidate["result"]
        ]
        if missing_replay or missing_candidate:
            raise RuntimeError(
                "Force-context trajectory keys differ: "
                f"replay_missing={missing_replay}, candidate_missing={missing_candidate}."
            )
        for key in force_context_keys:
            replay_key_match = _per_env_equal(
                baseline["result"][key], replay["result"][key], args.replay_tol
            )
            candidate_key_match = _per_env_equal(
                baseline["result"][key], candidate["result"][key], args.replay_tol
            )
            force_replay_match &= replay_key_match
            force_candidate_match &= candidate_key_match
            force_replay_mismatch_by_key[key] = int((~replay_key_match).sum())
            force_candidate_mismatch_by_key[key] = int(
                (~candidate_key_match).sum()
            )
    replay_max = max(replay_deltas.values(), default=0.0)
    snapshot_max = max(snapshot_deltas.values(), default=0.0)
    if snapshot_max > args.replay_tol:
        raise RuntimeError(
            f"Reconstructed snapshots differ by {snapshot_max}, refusing merge."
        )
    if replay_max > args.replay_tol or replay_failure_mismatch:
        raise RuntimeError(
            "Baseline process replay is not deterministic: "
            f"max={replay_max}, failure_mismatch={replay_failure_mismatch}."
        )
    # new change
    if args.require_force_context and not torch.all(force_replay_match):
        raise RuntimeError(
            "Baseline force-event replay is not deterministic; refusing merge: "
            f"{force_replay_mismatch_by_key}."
        )

    baseline_result = baseline["result"]
    candidate_result = candidate["result"]
    score_delta = candidate_result["score"] - baseline_result["score"]
    failure_delta = candidate_result["failure"].to(torch.int8) - baseline_result[
        "failure"
    ].to(torch.int8)
    beneficial = (failure_delta < 0) | (
        (failure_delta == 0) & (score_delta < 0.0)
    )
    # new change
    # Candidate changes may alter the physical trajectory, but a supervision
    # pair is valid only when its sampled force, wrist, event clock, applied
    # wrench and IK status stayed identical to baseline for the full horizon.
    valid = (
        baseline["valid"]
        & candidate["valid"]
        & force_replay_match
        & force_candidate_match
    )
    if args.require_force_context and not torch.any(valid):
        raise RuntimeError("No strictly force-matched candidate pairs remain.")
    force_gate_features = (
        baseline["snapshot"].get("force_context_force_gate_features")
        if args.require_force_context
        else None
    )
    force_conditioned_gate_input = (
        torch.cat(
            (
                baseline["snapshot"]["support_recovery_features"],
                force_gate_features,
            ),
            dim=-1,
        )
        if force_gate_features is not None
        else None
    )
    output_artifact = {
        # new change: keep reconstruction metadata so downstream state-gate
        # training can split whole phase-offset groups without filename parsing.
        "snapshot_fingerprint_metadata": baseline[
            "snapshot_fingerprint_metadata"
        ],
        "phase_offset": float(
            baseline["snapshot_fingerprint_metadata"]["phase_offset"]
        ),
        "support_recovery_features": baseline["snapshot"][
            "support_recovery_features"
        ],
        # new change
        # Dataset aggregation keeps controller memory alongside physical state.
        # These tensors participate in the snapshot fingerprint, hence every
        # finite-difference branch has identical residual/time context.
        "support_temporal_features": baseline["snapshot"].get(
            "support_temporal_features"
        ),
        "previous_support_residual_rad": baseline["snapshot"].get(
            "previous_support_residual_rad"
        ),
        "steps_since_residual_update": baseline["snapshot"].get(
            "steps_since_residual_update"
        ),
        # new change
        # The baseline always carries a zero/legacy intervention.  The merged
        # finite-difference sample must expose the actual candidate vector.
        "candidate_target_rad": candidate["snapshot"]["candidate_target_rad"],
        "valid": valid,
        "baseline": baseline_result,
        "candidate": candidate_result,
        "paired_score_delta": score_delta,
        "paired_failure_delta": failure_delta,
        "candidate_beneficial": beneficial,
        "motion_id": baseline["snapshot"]["motion_id"],
        "source_time_s": baseline["snapshot"]["motion_count"] * 0.02,
        "snapshot_fingerprint": baseline["snapshot_fingerprint"],
        "baseline_replay_deltas": replay_deltas,
        "baseline_replay_failure_mismatch": replay_failure_mismatch,
        "snapshot_deltas": snapshot_deltas,
        "strict_pairing": True,
        # new change
        "force_conditioned_pairing": bool(args.require_force_context),
        "force_gate_features": force_gate_features,
        "force_gate_feature_names": baseline.get("force_gate_feature_names", ()),
        "force_conditioned_gate_input": force_conditioned_gate_input,
        "force_conditioned_gate_input_dim": (
            int(force_conditioned_gate_input.shape[-1])
            if force_conditioned_gate_input is not None
            else 0
        ),
        "force_context_trajectory": (
            {
                key: baseline_result[key]
                for key in force_context_keys
            }
            if args.require_force_context
            else {}
        ),
        "force_replay_match": force_replay_match,
        "force_candidate_match": force_candidate_match,
        "force_replay_mismatch_by_key": force_replay_mismatch_by_key,
        "force_candidate_mismatch_by_key": force_candidate_mismatch_by_key,
        "deterministic_process_reconstruction": True,
        "ppo_updates": False,
        "uses_v3_checkpoint": False,
        "support_action_ids": baseline["support_action_ids"],
        "support_joint_names": baseline["support_joint_names"],
        "teacher_gain": baseline["teacher_gain"],
        "teacher_limit_rad": baseline["teacher_limit_rad"],
        # new change
        "candidate_mode": candidate.get("candidate_mode", "legacy_unknown"),
        "closed_loop_candidate": bool(
            candidate.get("closed_loop_candidate", False)
        ),
        # new change
        # Baseline action is independent of pulse length.  The supervision
        # artifact must describe the candidate dwell time, especially when a
        # single audited baseline/replay pair is reused for a dwell sweep.
        "pulse_steps": candidate["pulse_steps"],
        "outcome_horizon_steps": baseline["outcome_horizon_steps"],
        "outcome_horizon_s": baseline["outcome_horizon_s"],
        "failure_penalty": baseline["failure_penalty"],
        # new change
        "short_horizon_metric_schema": baseline.get(
            "short_horizon_metric_schema"
        ),
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_artifact, output)
    print(
        {
            "output": str(output),
            "snapshot_fingerprint": baseline["snapshot_fingerprint"],
            "snapshot_max_delta": snapshot_max,
            "baseline_replay_max_delta": replay_max,
            "baseline_replay_failure_mismatch": replay_failure_mismatch,
            "valid": int(valid.sum()),
            "baseline_failures": int((baseline_result["failure"] & valid).sum()),
            "candidate_failures": int((candidate_result["failure"] & valid).sum()),
            "candidate_beneficial_fraction": float(beneficial[valid].float().mean()),
            "paired_score_delta_mean": float(score_delta[valid].mean()),
            # new change
            "force_replay_match": int(force_replay_match.sum()),
            "force_candidate_match": int(force_candidate_match.sum()),
            "force_gate_input_dim": (
                int(force_conditioned_gate_input.shape[-1])
                if force_conditioned_gate_input is not None
                else 0
            ),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
