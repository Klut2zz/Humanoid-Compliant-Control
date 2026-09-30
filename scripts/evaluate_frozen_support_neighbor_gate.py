"""Long-horizon inference for frozen model_1375 with a fitted support gate."""

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
    "Isaac-G1-StandCoordination-FrozenNeighborGate-ZeroForce-"
    "100Pct-29DOF-v0"
)
LOW_FORCE_TASK = (
    "Isaac-G1-StandCoordination-FrozenNeighborGate-LowForce4to7N-"
    "100Pct-29DOF-v0"
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--force_mode", choices=("zero", "low"), default="zero")
parser.add_argument("--branch", choices=("baseline", "gated"), required=True)
parser.add_argument("--jit_policy", required=True)
parser.add_argument("--gate_artifact", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
# new change
parser.add_argument(
    "--runtime_ood_scale",
    type=float,
    default=1.0,
    help="Conservative multiplier on the fitted state-OOD radius.",
)
parser.add_argument(
    "--runtime_score_deadband",
    type=float,
    default=None,
    help="Optional stricter required predicted benefit magnitude.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# new change
class PairedPhaseStateNeighborGate:
    """Runtime form of the accepted non-parametric causal gate."""

    def __init__(self, artifact, device):
        import torch

        if artifact.get("model_class") != (
            "PairedPhaseNeighborGateWithStateOODVeto"
        ):
            raise ValueError("Unexpected neighbor-gate artifact class.")
        if not artifact.get("strict_pairing", False):
            raise ValueError("Gate artifact is not based on strict pairs.")
        if artifact.get("ppo_updates", True) or artifact.get(
            "uses_v3_checkpoint", True
        ):
            raise ValueError("Gate artifact has disallowed PPO/V3 provenance.")
        self.features = artifact["features"].to(device)
        self.score = artifact["paired_score_delta"].to(device)
        self.failure = artifact["paired_failure_delta"].to(device)
        self.motion = artifact["motion_id"].to(device)
        self.columns = torch.tensor(
            artifact["state_safety_columns"], device=device, dtype=torch.long
        )
        self.state_scale = artifact["state_scale"].to(device)
        self.state_ood_threshold = float(artifact["state_ood_threshold"])
        self.neighbor_count = int(artifact["neighbor_count"])
        self.confidence_multiplier = float(
            artifact["confidence_sem_multiplier"]
        )
        self.score_deadband = float(artifact["score_deadband"])
        self.score_clip = float(artifact["score_clip"])
        metadata = artifact["controller_metadata"]
        self.gain = float(metadata["teacher_gain"])
        self.limit_rad = float(metadata["teacher_limit_rad"])
        self.pulse_steps = int(metadata["pulse_steps"])
        self.decision_interval = int(metadata["outcome_horizon_steps"])
        self.action_scale_rad = float(metadata["action_scale_rad"])
        self.action_ids = torch.tensor(
            artifact["support_action_ids"], device=device, dtype=torch.long
        )
        self.motion_indices = {
            int(motion_id): torch.where(self.motion == motion_id)[0]
            for motion_id in self.motion.unique().tolist()
        }

    def apply_runtime_safety(self, ood_scale, score_deadband):
        if not 0.0 < ood_scale <= 1.0:
            raise ValueError("--runtime_ood_scale must be in (0, 1].")
        self.state_ood_threshold *= ood_scale
        if score_deadband is not None:
            if score_deadband < self.score_deadband:
                raise ValueError(
                    "Runtime score deadband may only become more conservative."
                )
            self.score_deadband = score_deadband

    def decide(self, query, motion_id):
        import torch

        selected = torch.zeros(query.shape[0], device=query.device, dtype=torch.bool)
        failure_veto = torch.zeros_like(selected)
        state_ood = torch.zeros_like(selected)
        predicted_upper = torch.full_like(
            selected, float("inf"), dtype=query.dtype
        )
        candidate = torch.clamp(
            -self.gain * query[:, 20:30],
            min=-self.limit_rad,
            max=self.limit_rad,
        )
        for value in motion_id.unique().tolist():
            value = int(value)
            query_indices = torch.where(motion_id == value)[0]
            train_indices = self.motion_indices.get(value)
            if train_indices is None or train_indices.numel() == 0:
                state_ood[query_indices] = True
                continue
            phase_distance = torch.cdist(
                query[query_indices, 50:52],
                self.features[train_indices, 50:52],
            )
            k = min(self.neighbor_count, train_indices.numel())
            neighbors = phase_distance.topk(k, largest=False).indices
            local_score = self.score[train_indices][neighbors].clamp(
                -self.score_clip, self.score_clip
            )
            predicted_upper[query_indices] = local_score.mean(dim=1) + (
                self.confidence_multiplier
                * local_score.std(dim=1)
                / math.sqrt(k)
            )
            failure_veto[query_indices] = (
                self.failure[train_indices][neighbors] > 0
            ).any(dim=1)
            local_state = self.features[train_indices][neighbors][
                :, :, self.columns
            ].mean(dim=1)
            state_distance = torch.sqrt(
                (
                    (
                        (query[query_indices][:, self.columns] - local_state)
                        / self.state_scale
                    )
                    ** 2
                ).mean(dim=1)
            )
            state_ood[query_indices] = (
                state_distance > self.state_ood_threshold
            )
        selected = (
            (predicted_upper < -self.score_deadband)
            & ~failure_veto
            & ~state_ood
        )
        return selected, candidate, failure_veto, state_ood, predicted_upper


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
def _fingerprint(observations, command, seed):
    digest = hashlib.sha256()
    for name in ("policy", "support_adapter"):
        value = observations[name].detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    digest.update(command.motion_ids.detach().contiguous().cpu().numpy().tobytes())
    digest.update(str(seed).encode())
    return digest.hexdigest()


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
        40.0, (args.warmup_steps + args.steps + 20) * env_cfg.sim.dt * env_cfg.decimation
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
        gate = PairedPhaseStateNeighborGate(gate_data, device)
        gate.apply_runtime_safety(
            args.runtime_ood_scale, args.runtime_score_deadband
        )
        observations, _ = env.reset()
        command = unwrapped.command_manager.get_term("upper_body_joints")

        for _ in range(args.warmup_steps):
            with torch.inference_mode():
                action = policy(observations["policy"]).detach()
            observations, _, _, _, _ = unwrapped.step(action)

        start_fingerprint = _fingerprint(observations, command, args.seed)
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

        pulse_mask = torch.zeros(
            args.num_envs, device=device, dtype=torch.bool
        )
        pulse_target = torch.zeros(
            args.num_envs, 10, device=device
        )
        pulse_remaining = 0
        gate_decisions = 0
        selected_total = 0
        failure_veto_total = 0
        state_ood_total = 0
        correction_abs_sum = 0.0
        correction_elements = 0
        correction_saturated = 0
        force_sum = 0.0
        force_samples = 0
        force_active = 0
        force_peak = 0.0

        for step_index in range(args.steps):
            features = observations["support_adapter"]
            if features.shape[-1] != 87:
                raise RuntimeError(
                    f"Expected 87D support state, got {features.shape}."
                )
            trend = features[:, 79:87]
            max_com_offset = torch.maximum(max_com_offset, trend[:, 0].abs())
            max_torso_tilt = torch.maximum(max_torso_tilt, trend[:, 2].abs())
            max_root_speed = torch.maximum(max_root_speed, trend[:, 4].abs())
            max_foot_speed = torch.maximum(max_foot_speed, trend[:, 5].abs())
            max_fixed_foot_error_m = torch.maximum(
                max_fixed_foot_error_m,
                features[:, 75:79].abs().amax(dim=1) / 10.0,
            )
            stability_cost_sum += _stability_cost(trend)

            if (
                args.branch == "gated"
                and step_index % gate.decision_interval == 0
            ):
                motion_id = command.motion_ids % len(gate.motion_indices)
                (
                    pulse_mask,
                    pulse_target,
                    failure_veto,
                    state_ood,
                    _,
                ) = gate.decide(features, motion_id)
                pulse_remaining = gate.pulse_steps
                gate_decisions += args.num_envs
                selected_total += int(pulse_mask.sum())
                failure_veto_total += int(failure_veto.sum())
                state_ood_total += int(state_ood.sum())

            with torch.inference_mode():
                action = policy(observations["policy"]).detach()
                if args.branch == "gated" and pulse_remaining > 0:
                    applied = pulse_target * pulse_mask[:, None]
                    action[:, gate.action_ids] += applied / gate.action_scale_rad
                    correction_abs_sum += float(applied.abs().sum())
                    correction_elements += int(pulse_mask.sum()) * 10
                    correction_saturated += int(
                        (
                            applied[pulse_mask].abs()
                            >= gate.limit_rad - 1.0e-8
                        ).sum()
                    )
                    pulse_remaining -= 1
            motion_before_step = (
                command.motion_ids % len(gate.motion_indices)
            ).clone()
            observations, _, terminated, truncated, _ = unwrapped.step(action)
            bad_orientation = _term(unwrapped, "bad_orientation", args.num_envs)
            base_contact = _term(unwrapped, "base_contact", args.num_envs)
            failed = terminated.bool() & ~truncated.bool()
            first = failed & (first_failure_step == args.steps)
            first_failure_step[first] = step_index + 1
            failure_count += failed.long()
            bad_orientation_count += (failed & bad_orientation).long()
            base_contact_count += (failed & base_contact).long()
            for motion_value in range(8):
                motion_failure_count[motion_value] += (
                    failed & (motion_before_step == motion_value)
                ).sum()
            if bool(failed.any()):
                pulse_mask[failed] = False

            force_world = getattr(command, "force_world", None)
            if force_world is not None:
                magnitude = torch.linalg.vector_norm(force_world, dim=-1)
                force_sum += float(magnitude.sum())
                force_samples += magnitude.numel()
                force_active += int((magnitude > 1.0e-6).sum())
                force_peak = max(force_peak, float(magnitude.max()))

        failed_once = first_failure_step < args.steps
        summary = {
            "task": task,
            "force_mode": args.force_mode,
            "branch": args.branch,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "start_fingerprint": start_fingerprint,
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
            "p95_max_torso_tilt_rad": float(
                torch.quantile(max_torso_tilt, 0.95)
            ),
            "max_com_offset_m": float(max_com_offset.max()),
            "p95_max_com_offset_m": float(
                torch.quantile(max_com_offset, 0.95)
            ),
            "max_root_speed_m_s": float(max_root_speed.max()),
            "p95_max_root_speed_m_s": float(
                torch.quantile(max_root_speed, 0.95)
            ),
            "max_foot_speed_m_s": float(max_foot_speed.max()),
            "p95_max_foot_speed_m_s": float(
                torch.quantile(max_foot_speed, 0.95)
            ),
            "max_fixed_foot_error_m": float(max_fixed_foot_error_m.max()),
            "p95_max_fixed_foot_error_m": float(
                torch.quantile(max_fixed_foot_error_m, 0.95)
            ),
            "mean_max_fixed_foot_error_m": float(
                max_fixed_foot_error_m.mean()
            ),
            "mean_stability_cost": float(
                (stability_cost_sum / args.steps).mean()
            ),
            "gate_decisions": gate_decisions,
            "gate_selected": selected_total,
            "gate_selected_fraction": selected_total / max(gate_decisions, 1),
            "gate_failure_veto": failure_veto_total,
            "gate_state_ood_veto": state_ood_total,
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
            "ppo_updates": False,
            "uses_v3_checkpoint": False,
            "gate_decision_interval_steps": gate.decision_interval,
            "gate_pulse_steps": gate.pulse_steps,
            "runtime_ood_scale": args.runtime_ood_scale,
            "runtime_state_ood_threshold": gate.state_ood_threshold,
            "runtime_score_deadband": gate.score_deadband,
        }
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(summary, flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
