"""Evaluate the accepted V2 multi-target gate with frozen model_1375."""

# new change
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).absolute().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "softmimic_gym"))

from isaaclab.app import AppLauncher


# new change
ZERO_FORCE_TASK = (
    "Isaac-G1-StandCoordination-V2FrozenInference-ZeroForce-"
    "100Pct-29DOF-v0"
)
LOW_FORCE_TASK = (
    "Isaac-G1-StandCoordination-V2FrozenInference-LowForce4to7N-"
    "100Pct-29DOF-v0"
)
SUPPORT_ACTION_IDS = (0, 1, 9, 10, 13, 14, 3, 4, 17, 18)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--force_mode", choices=("zero", "low"), required=True)
parser.add_argument(
    # new change
    "--branch",
    choices=("baseline", "gate_disabled", "gated", "continuous_candidate"),
    required=True,
)
parser.add_argument("--jit_policy", required=True)
parser.add_argument("--gate_artifact", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--force_event_seed", type=int, default=None)
parser.add_argument("--runtime_ood_scale", type=float, default=1.0)
parser.add_argument("--record_step_digest", action="store_true")
# new change
parser.add_argument(
    "--decision_trace",
    default=None,
    help=(
        "Optional .pt output with per-decision inputs, gate outputs and "
        "future baseline failure labels. This never changes the action."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# new change
class MultiTargetSafetyGateV2:
    """Frozen runtime form of the accepted five-member V2 ensemble."""

    def __init__(self, artifact, device):
        import torch
        from torch import nn

        if artifact.get("model_class") != (
            "BootstrapEnsembleMultiTargetForceSafetyGateV2"
        ):
            raise ValueError("Unexpected V2 gate artifact class.")
        if not artifact.get("accepted_for_frozen_inference", False):
            raise ValueError("V2 artifact has not passed frozen inference acceptance.")
        if artifact.get("ppo_updates", True) or artifact.get(
            "uses_v3_checkpoint", True
        ):
            raise ValueError("V2 artifact has disallowed PPO/V3 provenance.")
        architecture = artifact["architecture"]
        if architecture != {"input_dim": 110, "hidden_dims": (160, 96)}:
            raise ValueError(f"Unexpected V2 architecture: {architecture}")

        class Member(nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = nn.Sequential(
                    nn.Linear(110, 160),
                    nn.SiLU(),
                    nn.LayerNorm(160),
                    nn.Linear(160, 96),
                    nn.SiLU(),
                    nn.LayerNorm(96),
                )
                self.risk_head = nn.Linear(96, 1)
                self.prevented_failure_head = nn.Linear(96, 1)
                self.com_improvement_head = nn.Linear(96, 3)
                self.tilt_improvement_head = nn.Linear(96, 3)

            def forward(self, features):
                latent = self.trunk(features)
                return (
                    self.risk_head(latent).squeeze(-1),
                    self.prevented_failure_head(latent).squeeze(-1),
                    self.com_improvement_head(latent),
                    self.tilt_improvement_head(latent),
                )

        self.models = []
        for state_dict in artifact["ensemble_state_dicts"]:
            model = Member().to(device)
            model.load_state_dict(state_dict)
            model.eval()
            self.models.append(model)
        self.mean = artifact["feature_mean"].to(device)
        self.scale = artifact["feature_scale"].to(device)
        self.train = artifact["train_features_normalized"].to(device)
        self.ood_threshold = float(artifact["ood_threshold"])
        self.thresholds = artifact["thresholds"]
        self.action_ids = torch.tensor(
            SUPPORT_ACTION_IDS, device=device, dtype=torch.long
        )
        self.gain = 0.10
        self.limit_rad = 0.006
        self.pulse_steps = 5
        self.decision_interval = 250
        self.action_scale_rad = 0.30

    def apply_runtime_safety(self, ood_scale):
        if not 0.0 < ood_scale <= 1.0:
            raise ValueError("--runtime_ood_scale must be in (0, 1].")
        self.ood_threshold *= ood_scale

    def _nearest_distance(self, normalized):
        import torch

        values = []
        divisor = math.sqrt(normalized.shape[1])
        for start in range(0, normalized.shape[0], 128):
            distance = torch.cdist(normalized[start : start + 128], self.train)
            values.append(distance.min(dim=1).values / divisor)
        return torch.cat(values)

    def decide(self, support_features, force_features, motion_id):
        import torch
        from torch.nn import functional as F

        motion_onehot = F.one_hot(motion_id.long(), num_classes=8).float()
        features = torch.cat(
            (support_features, force_features, motion_onehot), dim=-1
        )
        if features.shape[-1] != 110:
            raise RuntimeError(f"Expected 110D V2 input, got {features.shape}.")
        normalized = (features - self.mean) / self.scale
        risk = []
        prevented = []
        com = []
        tilt = []
        with torch.inference_mode():
            for model in self.models:
                risk_logit, prevented_logit, com_value, tilt_value = model(
                    normalized
                )
                risk.append(torch.sigmoid(risk_logit))
                prevented.append(torch.sigmoid(prevented_logit))
                com.append(com_value)
                tilt.append(tilt_value)
        risk = torch.stack(risk)
        prevented = torch.stack(prevented)
        com = torch.stack(com)
        tilt = torch.stack(tilt)
        uncertainty = self.thresholds["uncertainty_multiplier"]
        risk_upper = risk.mean(0) + uncertainty * risk.std(0, unbiased=False)
        prevented_lower = prevented.mean(0) - uncertainty * prevented.std(
            0, unbiased=False
        )
        short_lower = torch.cat(
            (
                com.mean(0) - uncertainty * com.std(0, unbiased=False),
                tilt.mean(0) - uncertainty * tilt.std(0, unbiased=False),
            ),
            dim=-1,
        ).mean(dim=-1)
        ood_distance = self._nearest_distance(normalized)
        benefit = (
            prevented_lower
            >= self.thresholds["prevented_probability_threshold"]
        ) | (
            short_lower >= self.thresholds["short_improvement_threshold"]
        )
        selected = (
            (risk_upper <= self.thresholds["risk_threshold"])
            & benefit
            & (
                ood_distance
                <= self.ood_threshold * self.thresholds["ood_scale"]
            )
        )
        candidate = torch.clamp(
            -self.gain * support_features[:, 20:30],
            min=-self.limit_rad,
            max=self.limit_rad,
        )
        return {
            "selected": selected,
            "candidate": candidate,
            # new change
            "features": features,
            "normalized_features": normalized,
            "risk_upper": risk_upper,
            "prevented_lower": prevented_lower,
            "short_lower": short_lower,
            "ood_distance": ood_distance,
            "benefit": benefit,
        }


# new change
def _force_gate_features(command, count, device, force_mode):
    import torch

    if force_mode == "zero":
        return torch.zeros((count, 15), device=device)
    required = (
        "_sampled_peak_force_world",
        "_accepted_peak_force_world",
        "_sampled_body_slot",
        "_event_time",
        "_ramp_time",
        "_hold_time",
        "_solution_pending",
        "ik_feasible_scale",
        "force_world",
        "active_force_mask",
        "online_phase",
    )
    missing = [name for name in required if not hasattr(command, name)]
    if missing:
        raise RuntimeError(f"Low-force V2 context is missing {missing}.")
    sampled_force = command._sampled_peak_force_world
    magnitude = torch.linalg.vector_norm(sampled_force, dim=-1, keepdim=True)
    direction = sampled_force / magnitude.clamp_min(1e-8)
    wrist = torch.nn.functional.one_hot(
        command._sampled_body_slot, num_classes=2
    ).to(sampled_force.dtype)
    duration = 2.0 * command._ramp_time + command._hold_time
    signed_phase = torch.clamp(
        command._event_time / duration.clamp_min(1e-6), -1.0, 1.0
    ).unsqueeze(-1)
    accepted_magnitude = torch.linalg.vector_norm(
        command._accepted_peak_force_world, dim=-1, keepdim=True
    )
    applied_magnitude = torch.linalg.vector_norm(
        command.force_world, dim=-1, keepdim=True
    )
    envelope = torch.where(
        accepted_magnitude > 1e-8,
        applied_magnitude / accepted_magnitude.clamp_min(1e-8),
        torch.zeros_like(applied_magnitude),
    )
    pending = command._solution_pending
    accepted = (~pending) & (command.ik_feasible_scale > 0.0)
    rejected = (
        (~pending)
        & (command._event_time >= 0.0)
        & (command.ik_feasible_scale <= 0.0)
    )
    return torch.cat(
        (
            magnitude / 7.0,
            direction,
            wrist,
            signed_phase,
            command.online_phase,
            envelope,
            pending.to(sampled_force.dtype).unsqueeze(-1),
            accepted.to(sampled_force.dtype).unsqueeze(-1),
            rejected.to(sampled_force.dtype).unsqueeze(-1),
            command.active_force_mask.to(sampled_force.dtype).unsqueeze(-1),
            command.ik_feasible_scale.unsqueeze(-1),
        ),
        dim=-1,
    )


# new change
def _stability_cost(trend):
    import torch

    return (
        1.5 * trend[:, 0].abs()
        + 0.5 * torch.relu(trend[:, 1])
        + 2.0 * trend[:, 2].abs()
        + 0.5 * torch.relu(trend[:, 3])
        + trend[:, 4].abs()
        + 0.5 * trend[:, 5].abs()
        + 2.0 * torch.relu(0.25 - trend[:, 6])
        + 0.5 * trend[:, 7].abs()
    )


# new change
def _term(env, name, count):
    import torch

    try:
        return env.termination_manager.get_term(name).bool()
    except (KeyError, ValueError):
        return torch.zeros(count, device=env.device, dtype=torch.bool)


# new change
def _update_digest(digest, name, value):
    tensor = value.detach().contiguous().cpu()
    digest.update(name.encode())
    digest.update(tensor.numpy().tobytes())


# new change
def main():
    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils import parse_env_cfg

    import softmimic_gym.tasks  # noqa: F401

    if args.num_envs < 1 or args.steps < 1 or args.warmup_steps < 0:
        raise ValueError("Invalid rollout size.")
    task = ZERO_FORCE_TASK if args.force_mode == "zero" else LOW_FORCE_TASK
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.recorders = None
    env_cfg.sim.physx.enable_enhanced_determinism = True
    env_cfg.episode_length_s = max(
        40.0,
        (args.warmup_steps + args.steps + 20)
        * env_cfg.sim.dt
        * env_cfg.decimation,
    )
    for group_name in ("policy", "support_adapter"):
        group = getattr(env_cfg.observations, group_name)
        if hasattr(group, "enable_corruption"):
            group.enable_corruption = False
    env = gym.make(task, cfg=env_cfg)
    try:
        unwrapped = env.unwrapped
        device = unwrapped.device
        torch.manual_seed(args.seed)
        policy = torch.jit.load(args.jit_policy, map_location=device).eval()
        gate_data = torch.load(
            args.gate_artifact, map_location="cpu", weights_only=False
        )
        gate = MultiTargetSafetyGateV2(gate_data, device)
        gate.apply_runtime_safety(args.runtime_ood_scale)
        command = unwrapped.command_manager.get_term("upper_body_joints")
        event_seed = args.force_event_seed
        if event_seed is None:
            event_seed = args.seed
        if args.force_mode == "low":
            if not hasattr(command, "torch_gen"):
                raise RuntimeError("Low-force command has no private generator.")
            command.torch_gen.manual_seed(int(event_seed))
        observations, _ = env.reset()

        for _ in range(args.warmup_steps):
            with torch.inference_mode():
                action = policy(observations["policy"]).detach()
            observations, _, _, _, _ = unwrapped.step(action)

        first_failure_step = torch.full(
            (args.num_envs,), args.steps, device=device, dtype=torch.long
        )
        failure_count = torch.zeros(
            args.num_envs, device=device, dtype=torch.long
        )
        bad_orientation_count = torch.zeros_like(failure_count)
        base_contact_count = torch.zeros_like(failure_count)
        motion_failure_count = torch.zeros(8, device=device, dtype=torch.long)
        max_torso_tilt = torch.zeros(args.num_envs, device=device)
        max_com_offset = torch.zeros_like(max_torso_tilt)
        max_root_speed = torch.zeros_like(max_torso_tilt)
        max_foot_speed = torch.zeros_like(max_torso_tilt)
        max_fixed_foot_error_m = torch.zeros_like(max_torso_tilt)
        stability_cost_sum = torch.zeros_like(max_torso_tilt)
        rollout_digest = hashlib.sha256()

        pulse_mask = torch.zeros(
            args.num_envs, device=device, dtype=torch.bool
        )
        pulse_target = torch.zeros((args.num_envs, 10), device=device)
        pulse_remaining = 0
        decisions = 0
        selected_total = 0
        risk_veto_total = 0
        ood_veto_total = 0
        correction_abs_sum = 0.0
        correction_elements = 0
        correction_saturated = 0
        risk_upper_sum = 0.0
        prevented_lower_sum = 0.0
        short_lower_sum = 0.0
        prediction_elements = 0
        # new change
        continuous_candidate_env_steps = 0
        force_sum = 0.0
        force_samples = 0
        force_active = 0
        force_peak = 0.0
        # new change
        # Trace collection is deliberately CPU-backed and observational.  It
        # is valid for baseline/gate_disabled diagnosis because it never
        # modifies the policy action.
        decision_trace = {
            name: []
            for name in (
                "step_index",
                "features",
                "normalized_features",
                "candidate",
                "selected",
                "risk_upper",
                "prevented_lower",
                "short_lower",
                "ood_distance",
                "benefit",
                "motion_id",
            )
        }
        failure_trace = []

        start_digest = hashlib.sha256()
        _update_digest(start_digest, "policy", observations["policy"])
        _update_digest(start_digest, "support", observations["support_adapter"])
        start_fingerprint = start_digest.hexdigest()

        for step_index in range(args.steps):
            support = observations["support_adapter"]
            if support.shape[-1] != 87:
                raise RuntimeError(f"Expected 87D support state, got {support.shape}.")
            trend = support[:, 79:87]
            max_com_offset = torch.maximum(max_com_offset, trend[:, 0].abs())
            max_torso_tilt = torch.maximum(max_torso_tilt, trend[:, 2].abs())
            max_root_speed = torch.maximum(max_root_speed, trend[:, 4].abs())
            max_foot_speed = torch.maximum(max_foot_speed, trend[:, 5].abs())
            max_fixed_foot_error_m = torch.maximum(
                max_fixed_foot_error_m,
                support[:, 75:79].abs().amax(dim=1) / 10.0,
            )
            stability_cost_sum += _stability_cost(trend)

            if args.branch != "baseline" and step_index % gate.decision_interval == 0:
                force_features = _force_gate_features(
                    command, args.num_envs, device, args.force_mode
                )
                motion_id = command.motion_ids % 8
                prediction = gate.decide(support, force_features, motion_id)
                pulse_mask = prediction["selected"]
                if args.branch == "gate_disabled":
                    pulse_mask = torch.zeros_like(pulse_mask)
                pulse_target = prediction["candidate"]
                pulse_remaining = gate.pulse_steps
                decisions += args.num_envs
                selected_total += int(pulse_mask.sum())
                risk_veto_total += int(
                    (prediction["risk_upper"] > gate.thresholds["risk_threshold"]).sum()
                )
                ood_veto_total += int(
                    (
                        prediction["ood_distance"]
                        > gate.ood_threshold * gate.thresholds["ood_scale"]
                    ).sum()
                )
                risk_upper_sum += float(prediction["risk_upper"].sum())
                prevented_lower_sum += float(prediction["prevented_lower"].sum())
                short_lower_sum += float(prediction["short_lower"].sum())
                prediction_elements += args.num_envs
                # new change
                if args.decision_trace is not None:
                    decision_trace["step_index"].append(
                        torch.full(
                            (args.num_envs,),
                            step_index,
                            device="cpu",
                            dtype=torch.long,
                        )
                    )
                    for name in (
                        "features",
                        "normalized_features",
                        "candidate",
                        "selected",
                        "risk_upper",
                        "prevented_lower",
                        "short_lower",
                        "ood_distance",
                        "benefit",
                    ):
                        decision_trace[name].append(
                            prediction[name].detach().cpu()
                        )
                    decision_trace["motion_id"].append(
                        motion_id.detach().cpu()
                    )

            with torch.inference_mode():
                action = policy(observations["policy"]).detach()
                # new change
                if args.branch == "continuous_candidate":
                    applied = torch.clamp(
                        -gate.gain * support[:, 20:30],
                        min=-gate.limit_rad,
                        max=gate.limit_rad,
                    )
                    action[:, gate.action_ids] += applied / gate.action_scale_rad
                    correction_abs_sum += float(applied.abs().sum())
                    correction_elements += applied.numel()
                    correction_saturated += int(
                        (applied.abs() >= gate.limit_rad - 1e-8).sum()
                    )
                    continuous_candidate_env_steps += args.num_envs
                elif args.branch == "gated" and pulse_remaining > 0:
                    applied = pulse_target * pulse_mask[:, None]
                    action[:, gate.action_ids] += applied / gate.action_scale_rad
                    correction_abs_sum += float(applied.abs().sum())
                    correction_elements += int(pulse_mask.sum()) * 10
                    correction_saturated += int(
                        (applied[pulse_mask].abs() >= gate.limit_rad - 1e-8).sum()
                    )
                    pulse_remaining -= 1
            motion_before = (command.motion_ids % 8).clone()
            observations, _, terminated, truncated, _ = unwrapped.step(action)
            if args.record_step_digest:
                _update_digest(rollout_digest, "action", action)
                _update_digest(
                    rollout_digest,
                    "root",
                    unwrapped.scene["robot"].data.root_state_w,
                )
                _update_digest(
                    rollout_digest,
                    "joint",
                    unwrapped.scene["robot"].data.joint_pos,
                )
                _update_digest(rollout_digest, "policy", observations["policy"])
                _update_digest(
                    rollout_digest, "support", observations["support_adapter"]
                )
            bad_orientation = _term(unwrapped, "bad_orientation", args.num_envs)
            base_contact = _term(unwrapped, "base_contact", args.num_envs)
            failed = terminated.bool() & ~truncated.bool()
            # new change
            if args.decision_trace is not None:
                failure_trace.append(failed.detach().cpu())
            first = failed & (first_failure_step == args.steps)
            first_failure_step[first] = step_index + 1
            failure_count += failed.long()
            bad_orientation_count += (failed & bad_orientation).long()
            base_contact_count += (failed & base_contact).long()
            for motion_value in range(8):
                motion_failure_count[motion_value] += (
                    failed & (motion_before == motion_value)
                ).sum()
            if bool(failed.any()):
                pulse_mask[failed] = False

            force_world = getattr(command, "force_world", None)
            if force_world is not None:
                magnitude = torch.linalg.vector_norm(force_world, dim=-1)
                force_sum += float(magnitude.sum())
                force_samples += magnitude.numel()
                force_active += int((magnitude > 1e-6).sum())
                force_peak = max(force_peak, float(magnitude.max()))

        failed_once = first_failure_step < args.steps
        summary = {
            "schema": "V2FrozenInferenceRolloutV1",
            "task": task,
            "force_mode": args.force_mode,
            "branch": args.branch,
            "seed": args.seed,
            "force_event_seed": int(event_seed),
            "num_envs": args.num_envs,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "start_fingerprint": start_fingerprint,
            "step_digest_recorded": args.record_step_digest,
            "rollout_digest": (
                rollout_digest.hexdigest() if args.record_step_digest else None
            ),
            "decision_trace": args.decision_trace,
            "failed_envs": int(failed_once.sum()),
            "total_failures": int(failure_count.sum()),
            "bad_orientation_failures": int(bad_orientation_count.sum()),
            "base_contact_failures": int(base_contact_count.sum()),
            "first_failure_step_min": (
                int(first_failure_step[failed_once].min())
                if bool(failed_once.any())
                else None
            ),
            "first_failure_step_mean": (
                float(first_failure_step[failed_once].float().mean())
                if bool(failed_once.any())
                else None
            ),
            "motion_failure_count": motion_failure_count.cpu().tolist(),
            "max_torso_tilt_rad": float(max_torso_tilt.max()),
            "p95_max_torso_tilt_rad": float(torch.quantile(max_torso_tilt, 0.95)),
            "max_com_offset_m": float(max_com_offset.max()),
            "p95_max_com_offset_m": float(torch.quantile(max_com_offset, 0.95)),
            "max_root_speed_m_s": float(max_root_speed.max()),
            "p95_max_root_speed_m_s": float(torch.quantile(max_root_speed, 0.95)),
            "max_foot_speed_m_s": float(max_foot_speed.max()),
            "p95_max_foot_speed_m_s": float(torch.quantile(max_foot_speed, 0.95)),
            "max_fixed_foot_error_m": float(max_fixed_foot_error_m.max()),
            "p95_max_fixed_foot_error_m": float(
                torch.quantile(max_fixed_foot_error_m, 0.95)
            ),
            "mean_stability_cost": float((stability_cost_sum / args.steps).mean()),
            "gate_decisions": decisions,
            "gate_selected": selected_total,
            "gate_selected_fraction": selected_total / max(decisions, 1),
            # new change
            "candidate_application_mode": (
                "continuous_q_error_feedback"
                if args.branch == "continuous_candidate"
                else "gate_pulse_or_none"
            ),
            "continuous_candidate_env_steps": continuous_candidate_env_steps,
            "gate_risk_veto": risk_veto_total,
            "gate_ood_veto": ood_veto_total,
            "mean_risk_upper": risk_upper_sum / max(prediction_elements, 1),
            "mean_prevented_lower": prevented_lower_sum / max(prediction_elements, 1),
            "mean_short_lower": short_lower_sum / max(prediction_elements, 1),
            "mean_applied_correction_mrad": (
                1000.0 * correction_abs_sum / max(correction_elements, 1)
            ),
            "correction_saturation_fraction": (
                correction_saturated / max(correction_elements, 1)
            ),
            "mean_force_n": force_sum / max(force_samples, 1),
            "active_force_fraction": force_active / max(force_samples, 1),
            "peak_force_n": force_peak,
            "gate_artifact": str(pathlib.Path(args.gate_artifact).resolve()),
            "jit_policy": str(pathlib.Path(args.jit_policy).resolve()),
            "policy_frozen": True,
            "gate_frozen": True,
            "ppo_updates": False,
            "uses_v3_checkpoint": False,
            "gate_decision_interval_steps": gate.decision_interval,
            "gate_pulse_steps": gate.pulse_steps,
            "runtime_ood_scale": args.runtime_ood_scale,
            "runtime_ood_threshold": gate.ood_threshold,
            "thresholds": gate.thresholds,
        }
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n")
        # new change
        if args.decision_trace is not None:
            if not decision_trace["step_index"]:
                raise RuntimeError(
                    "Decision trace requested but the selected branch made no "
                    "gate decisions; use gate_disabled or gated."
                )
            failures = torch.stack(failure_trace, dim=0)
            trace = {
                name: torch.cat(values, dim=0)
                for name, values in decision_trace.items()
            }
            decision_steps = trace["step_index"].reshape(-1, args.num_envs)[:, 0]
            future_failure = {}
            for horizon_steps, horizon_name in (
                (25, "0p5s"),
                (50, "1p0s"),
                (100, "2p0s"),
                (250, "5p0s"),
            ):
                labels = []
                for decision_step in decision_steps.tolist():
                    end = min(args.steps, decision_step + horizon_steps)
                    labels.append(
                        failures[decision_step:end].any(dim=0)
                    )
                future_failure[horizon_name] = torch.cat(labels, dim=0)
            trace.update(
                {
                    "schema": "V2FrozenInferenceDecisionTraceV1",
                    "task": task,
                    "force_mode": args.force_mode,
                    "branch": args.branch,
                    "seed": args.seed,
                    "force_event_seed": int(event_seed),
                    "num_envs": args.num_envs,
                    "steps": args.steps,
                    "decision_interval_steps": gate.decision_interval,
                    "future_failure": future_failure,
                    "feature_mean": gate.mean.detach().cpu(),
                    "feature_scale": gate.scale.detach().cpu(),
                    "ood_threshold_runtime": gate.ood_threshold
                    * gate.thresholds["ood_scale"],
                    "thresholds": gate.thresholds,
                    "policy_frozen": True,
                    "gate_frozen": True,
                    "ppo_updates": False,
                }
            )
            trace_output = pathlib.Path(args.decision_trace)
            trace_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(trace, trace_output)
        print(summary, flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
