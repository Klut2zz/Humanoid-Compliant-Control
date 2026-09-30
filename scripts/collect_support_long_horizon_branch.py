"""Run one deterministic long-horizon branch from a reconstructible snapshot."""

# new change
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).absolute().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "softmimic_gym"))

from isaaclab.app import AppLauncher


# new change
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default=(
        "Isaac-G1-StandCoordination-SupportRecoveryAwareTeacherV3-"
        "100Pct-29DOF-v0"
    ),
)
parser.add_argument("--jit_policy", required=True)
# new change
parser.add_argument(
    "--jit_extra_obs_groups",
    nargs="*",
    default=(),
    help=(
        "Observation groups appended after policy for the branch controller. "
        "Empty preserves the legacy model_1375 interface."
    ),
)
# new change
parser.add_argument(
    "--snapshot_rollout_jit_policy",
    default=None,
    help=(
        "Optional distilled support controller used only during snapshot warmup. "
        "This makes finite-difference labels cover states actually visited by the "
        "deployed sample/hold controller while all outcome branches still start "
        "from one strictly reconstructed snapshot."
    ),
)
# new change
parser.add_argument(
    "--snapshot_rollout_extra_obs_groups",
    nargs="*",
    default=(),
    help="Observation groups appended after policy for snapshot rollout JIT.",
)
# new change
parser.add_argument(
    "--snapshot_residual_hold_steps",
    type=int,
    default=125,
    help="Sample/hold period of the distilled controller used during warmup.",
)
parser.add_argument("--branch", choices=("baseline", "candidate"), required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
# new change
parser.add_argument(
    "--force_event_seed",
    type=int,
    default=None,
    help=(
        "Independent seed for wrist/force/timing sampling. Required with "
        "--require_force_context because ReferenceCommand otherwise hard-codes 42."
    ),
)
parser.add_argument("--phase_offset", type=float, default=0.0)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--outcome_horizon_s", type=float, default=2.0)
parser.add_argument("--teacher_gain", type=float, default=0.10)
parser.add_argument("--teacher_limit_rad", type=float, default=0.006)
parser.add_argument("--pulse_steps", type=int, default=5)
# new change
parser.add_argument(
    "--closed_loop_candidate",
    action="store_true",
    help=(
        "Recompute clip(-gain * (q-q_ref), limit) from the current support "
        "observation at every candidate step.  The legacy default keeps the "
        "single correction captured at the reconstructed snapshot."
    ),
)
# new change
parser.add_argument(
    "--candidate_vector_rad",
    default=None,
    help=(
        "Optional comma-separated 10DOF support residual in SUPPORT_ACTION_IDS "
        "order for strict finite-difference probes."
    ),
)
# new change
parser.add_argument(
    "--candidate_action_ids",
    nargs="*",
    type=int,
    default=None,
    help=(
        "Optional strict-29DOF action rows for a generalized finite-difference "
        "probe. Empty preserves the canonical 10DOF support rows."
    ),
)
# new change
parser.add_argument(
    "--candidate_action_scales_rad",
    nargs="*",
    type=float,
    default=None,
    help=(
        "Physical radians per actor unit for --candidate_action_ids. Empty "
        "uses 0.30 rad for every legacy support row."
    ),
)
# new change
parser.add_argument(
    "--candidate_joint_names",
    nargs="*",
    default=None,
    help="Optional labels matching --candidate_action_ids for audit metadata.",
)
# new change
parser.add_argument(
    "--candidate_tensor_pt",
    default=None,
    help=(
        "Optional torch artifact containing per-environment [N,10] "
        "support_target_rad, used to validate a solved combined teacher."
    ),
)
parser.add_argument("--failure_penalty", type=float, default=20.0)
# new change
parser.add_argument(
    "--require_force_context",
    action="store_true",
    help=(
        "Require the online low-force command and record its sampled/applied "
        "wrench, wrist, event phase and IK acceptance at every step."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# new change
SUPPORT_ACTION_IDS = (0, 1, 9, 10, 13, 14, 3, 4, 17, 18)
SUPPORT_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)


# new change
def _audit_canonical_support_mapping() -> None:
    """Reject the historical positional-label mismatch before simulation."""

    from softmimic_gym.assets.robots.g1_compat import (
        G1_SOFTMIMIC_29_JOINT_NAMES,
    )

    resolved = tuple(G1_SOFTMIMIC_29_JOINT_NAMES[index] for index in SUPPORT_ACTION_IDS)
    if resolved != SUPPORT_JOINT_NAMES:
        raise RuntimeError(
            "Canonical 29DOF support mapping drifted: "
            f"ids={SUPPORT_ACTION_IDS}, resolved={resolved}, labels={SUPPORT_JOINT_NAMES}."
        )


# new change
def _candidate_spec() -> tuple[tuple[int, ...], tuple[float, ...], tuple[str, ...]]:
    action_ids = tuple(
        SUPPORT_ACTION_IDS
        if args.candidate_action_ids is None or len(args.candidate_action_ids) == 0
        else args.candidate_action_ids
    )
    scales = tuple(
        [0.30] * len(action_ids)
        if args.candidate_action_scales_rad is None
        or len(args.candidate_action_scales_rad) == 0
        else args.candidate_action_scales_rad
    )
    names = tuple(
        SUPPORT_JOINT_NAMES
        if args.candidate_joint_names is None or len(args.candidate_joint_names) == 0
        else args.candidate_joint_names
    )
    if not action_ids or len(set(action_ids)) != len(action_ids):
        raise ValueError("Candidate action ids must be non-empty and unique.")
    if any(index < 0 or index >= 29 for index in action_ids):
        raise ValueError(f"Candidate action ids must be in [0, 28], got {action_ids}.")
    if len(scales) != len(action_ids) or any(scale <= 0.0 for scale in scales):
        raise ValueError(
            "Candidate action scales must be positive and match action ids: "
            f"ids={len(action_ids)} scales={len(scales)}."
        )
    if len(names) != len(action_ids):
        raise ValueError(
            "Candidate joint names must match action ids: "
            f"ids={len(action_ids)} names={len(names)}."
        )
    return action_ids, scales, names


# new change
def _policy_input(observations):
    import torch

    return torch.cat(
        [observations["policy"]]
        + [observations[name] for name in args.jit_extra_obs_groups],
        dim=-1,
    )


# new change
def _candidate_vector(device):
    import torch

    action_ids, _, _ = _candidate_spec()

    # new change
    if args.candidate_vector_rad is not None and args.candidate_tensor_pt is not None:
        raise ValueError(
            "--candidate_vector_rad and --candidate_tensor_pt are mutually exclusive."
        )
    if args.candidate_tensor_pt is not None:
        artifact = torch.load(
            args.candidate_tensor_pt, map_location=device, weights_only=False
        )
        values = artifact.get("support_target_rad")
        if values is None:
            raise KeyError("candidate tensor artifact lacks support_target_rad.")
        values = values.to(device=device, dtype=torch.float32)
        if values.shape != (args.num_envs, len(action_ids)):
            raise ValueError(
                "candidate tensor must have shape "
                f"({args.num_envs}, {len(action_ids)}), got {tuple(values.shape)}."
            )
        if float(values.abs().max()) > args.teacher_limit_rad + 1.0e-9:
            raise ValueError(
                "Candidate tensor exceeds --teacher_limit_rad: "
                f"max={float(values.abs().max()):.6f}."
            )
        return values
    if args.candidate_vector_rad is None:
        return None
    values = [float(value) for value in args.candidate_vector_rad.split(",")]
    if len(values) != len(action_ids):
        raise ValueError(
            "--candidate_vector_rad must match --candidate_action_ids: "
            f"values, got {len(values)}."
        )
    vector = torch.tensor(values, device=device, dtype=torch.float32)
    if float(vector.abs().max()) > args.teacher_limit_rad + 1.0e-9:
        raise ValueError(
            "Explicit candidate exceeds --teacher_limit_rad: "
            f"max={float(vector.abs().max()):.6f}."
        )
    return vector


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
def _support_metric_slices(observation_manager):
    """Resolve named terms instead of relying on a fragile flat index."""
    import math

    # IsaacLab 2.x exposes the public name list through ``active_terms``.
    names = observation_manager.active_terms["support_adapter"]
    dims = observation_manager.group_obs_term_dim["support_adapter"]
    offset = 0
    output = {}
    for name, shape in zip(names, dims, strict=True):
        width = math.prod(shape)
        output[name] = (offset, offset + width)
        offset += width
    required = {"com_support_offset", "torso_projected_gravity"}
    missing = required.difference(output)
    if missing:
        raise RuntimeError(f"Missing support metric terms: {sorted(missing)}")
    return output


# new change
def _short_horizon_support_metrics(features, slices):
    """Return physical COM-support distance [m] and torso tilt [rad]."""
    import torch

    com_start, com_end = slices["com_support_offset"]
    gravity_start, gravity_end = slices["torso_projected_gravity"]
    # This observation term is intentionally scaled by 10.0 in the task.
    com_offset_m = features[:, com_start:com_end] / 10.0
    gravity = features[:, gravity_start:gravity_end].reshape(
        features.shape[0], -1, 3
    )[:, 0]
    gravity = torch.nn.functional.normalize(gravity, dim=-1)
    tilt_rad = torch.acos((-gravity[:, 2]).clamp(-1.0, 1.0))
    return torch.linalg.vector_norm(com_offset_m[:, :2], dim=-1), tilt_rad


# new change
def _disable_automatic_terminations(env_cfg):
    disabled = []
    for name in list(vars(env_cfg.terminations)):
        if name.startswith("_"):
            continue
        if getattr(env_cfg.terminations, name) is not None:
            setattr(env_cfg.terminations, name, None)
            disabled.append(name)
    return disabled


# new change
def _configure_deterministic_reconstruction(env_cfg):
    env_cfg.sim.physx.enable_enhanced_determinism = True
    command_cfg = env_cfg.commands.upper_body_joints
    command_cfg.demo_random_init = False
    command_cfg.demo_random_reset = False
    command_cfg.demo_zero_reset = False
    command_cfg.clip_length = None
    reset_cfg = env_cfg.events.reset_robot
    reset_cfg.params["jpos_range"] = (0.0, 0.0)
    reset_cfg.params["jvel_range"] = (0.0, 0.0)
    for key in reset_cfg.params["pose_range"]:
        reset_cfg.params["pose_range"][key] = (0.0, 0.0)
    for key in reset_cfg.params["velocity_range"]:
        reset_cfg.params["velocity_range"][key] = (0.0, 0.0)


# new change
def _initialize_stratified_states(
    env, motion_file_count, reserve_steps, phase_offset
):
    import torch

    command = env.command_manager.get_term("upper_body_joints")
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env.num_envs % motion_file_count != 0:
        raise ValueError(
            f"--num_envs must be divisible by {motion_file_count} motion files."
        )
    per_motion = env.num_envs // motion_file_count
    phase_rank = torch.div(env_ids, motion_file_count, rounding_mode="floor")
    fraction = ((phase_rank.float() + 0.5) / float(per_motion) + phase_offset) % 1.0
    max_times = command.motion_lib.get_max_times(env_ids)
    usable_times = (max_times - reserve_steps * env.step_dt).clamp_min(0.0)
    counts = torch.floor(fraction * usable_times / command.motion_dt).to(
        command.motion_count.dtype
    )
    command.motion_count[:] = counts
    command.motion_start.zero_()
    command.motion_durations[:] = max_times
    command._update_command(step=False)
    env._reset_idx(env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.obs_buf = env.observation_manager.compute(update_history=True)
    return env.obs_buf


# new change
def _failure_terms(env):
    import torch
    import torch.nn.functional as functional

    robot = env.scene["robot"]
    command = env.command_manager.get_term("upper_body_joints")
    current = functional.normalize(robot.data.projected_gravity_b, dim=-1)
    reference = functional.normalize(
        command.root_state["projected_gravity_b"][:, 1], dim=-1
    )
    angle = torch.acos(torch.sum(current * reference, dim=-1).clamp(-1.0, 1.0))
    bad_orientation = angle.abs() > 0.7853981633974483
    sensor = env.scene["contact_forces"]
    body_ids, _ = sensor.find_bodies(
        ["pelvis", "torso_link"], preserve_order=True
    )
    contact = torch.linalg.vector_norm(
        sensor.data.net_forces_w_history[:, :, body_ids, :], dim=-1
    )
    base_contact = torch.any(contact > 1.0, dim=(1, 2))
    return bad_orientation, base_contact


# new change
def _snapshot_fingerprint(tensors, metadata):
    digest = hashlib.sha256()
    for name in sorted(tensors):
        # new change
        # ``candidate_target_rad`` is an intervention requested *after* the
        # physical snapshot is reconstructed.  Including it in the digest
        # incorrectly makes +/− finite-difference branches look like different
        # simulator snapshots and prevents one audited baseline from being
        # shared by all 10DOF probes.
        if name == "candidate_target_rad":
            continue
        value = tensors[name].detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    for key in sorted(metadata):
        digest.update(f"{key}={metadata[key]}".encode("utf-8"))
    return digest.hexdigest()


# new change
FORCE_GATE_FEATURE_NAMES = (
    "sampled_force_magnitude_over_7n",
    "sampled_force_direction_w_x",
    "sampled_force_direction_w_y",
    "sampled_force_direction_w_z",
    "left_wrist_event",
    "right_wrist_event",
    "signed_event_phase",
    "event_phase_sin",
    "event_phase_cos",
    "force_envelope",
    "ik_pending",
    "ik_accepted",
    "ik_rejected",
    "force_active",
    "ik_feasible_scale",
)


# new change
def _online_force_context(command):
    """Read exogenous force/IK state without mutating the online generator."""
    import torch

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
        raise RuntimeError(
            "Force-conditioned pairing requires OnlineComplianceReferenceCommand; "
            f"missing {missing}."
        )
    if len(command.cfg.force_body_names) != 2 or tuple(
        command.cfg.force_body_names
    ) != ("left_wrist_yaw_link", "right_wrist_yaw_link"):
        raise RuntimeError(
            "Strict low-force pairing currently requires the validated two-wrist "
            f"generator, got {tuple(command.cfg.force_body_names)}."
        )

    sampled_force = command._sampled_peak_force_world
    sampled_magnitude = torch.linalg.vector_norm(
        sampled_force, dim=-1, keepdim=True
    )
    sampled_direction = sampled_force / sampled_magnitude.clamp_min(1.0e-8)
    wrist_onehot = torch.nn.functional.one_hot(
        command._sampled_body_slot, num_classes=2
    ).to(sampled_force.dtype)
    duration = 2.0 * command._ramp_time + command._hold_time
    signed_phase = torch.clamp(
        command._event_time / duration.clamp_min(1.0e-6), -1.0, 1.0
    ).unsqueeze(-1)
    accepted_magnitude = torch.linalg.vector_norm(
        command._accepted_peak_force_world, dim=-1, keepdim=True
    )
    applied_magnitude = torch.linalg.vector_norm(
        command.force_world, dim=-1, keepdim=True
    )
    envelope = torch.where(
        accepted_magnitude > 1.0e-8,
        applied_magnitude / accepted_magnitude.clamp_min(1.0e-8),
        torch.zeros_like(applied_magnitude),
    )
    pending = command._solution_pending
    accepted = (~pending) & (command.ik_feasible_scale > 0.0)
    rejected = (
        (~pending)
        & (command._event_time >= 0.0)
        & (command.ik_feasible_scale <= 0.0)
    )
    active = command.active_force_mask
    gate_features = torch.cat(
        (
            sampled_magnitude / 7.0,
            sampled_direction,
            wrist_onehot,
            signed_phase,
            command.online_phase,
            envelope,
            pending.to(sampled_force.dtype).unsqueeze(-1),
            accepted.to(sampled_force.dtype).unsqueeze(-1),
            rejected.to(sampled_force.dtype).unsqueeze(-1),
            active.to(sampled_force.dtype).unsqueeze(-1),
            command.ik_feasible_scale.unsqueeze(-1),
        ),
        dim=-1,
    )
    if gate_features.shape[-1] != len(FORCE_GATE_FEATURE_NAMES):
        raise RuntimeError("Force gate feature layout changed unexpectedly.")
    return {
        "sampled_peak_force_world": sampled_force.detach().clone(),
        "accepted_peak_force_world": command._accepted_peak_force_world.detach().clone(),
        "applied_force_world": command.force_world.detach().clone(),
        "sampled_body_slot": command._sampled_body_slot.detach().clone(),
        "event_time_s": command._event_time.detach().clone(),
        "ramp_time_s": command._ramp_time.detach().clone(),
        "hold_time_s": command._hold_time.detach().clone(),
        "event_phase_sincos": command.online_phase.detach().clone(),
        "force_envelope": envelope.squeeze(-1).detach().clone(),
        "ik_pending": pending.detach().clone(),
        "ik_accepted": accepted.detach().clone(),
        "ik_rejected": rejected.detach().clone(),
        "force_active": active.detach().clone(),
        "ik_feasible_scale": command.ik_feasible_scale.detach().clone(),
        "ik_active_residual_m": command.ik_active_residual_m.detach().clone(),
        "ik_active_relative_residual": command.ik_active_relative_residual.detach().clone(),
        "ik_direction_cosine": command.ik_direction_cosine.detach().clone(),
        "ik_max_foot_motion_m": command.ik_max_foot_motion_m.detach().clone(),
        "ik_com_residual_m": command.ik_com_residual_m.detach().clone(),
        "force_gate_features": gate_features.detach().clone(),
    }


# new change
def main():
    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils import parse_env_cfg

    import softmimic_gym.tasks  # noqa: F401

    # new change
    _audit_canonical_support_mapping()

    if not 2.0 <= args.outcome_horizon_s <= 5.0:
        raise ValueError("--outcome_horizon_s must be in [2, 5].")
    if not 0.0 <= args.phase_offset < 1.0:
        raise ValueError("--phase_offset must be in [0, 1).")
    if args.num_envs < 1 or args.warmup_steps < 0:
        raise ValueError("Invalid environment or warmup count.")
    # new change
    if args.snapshot_rollout_jit_policy is not None and args.snapshot_residual_hold_steps < 1:
        raise ValueError("--snapshot_residual_hold_steps must be positive.")
    # new change
    if args.require_force_context and args.force_event_seed is None:
        raise ValueError(
            "--force_event_seed is required for force-conditioned collection; "
            "the base ReferenceCommand generator otherwise always uses seed 42."
        )

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.recorders = None
    disabled_terminations = _disable_automatic_terminations(env_cfg)
    _configure_deterministic_reconstruction(env_cfg)
    if hasattr(env_cfg.observations.policy, "enable_corruption"):
        env_cfg.observations.policy.enable_corruption = False
    env = gym.make(args.task, cfg=env_cfg)
    try:
        unwrapped = env.unwrapped
        device = unwrapped.device
        policy = torch.jit.load(args.jit_policy, map_location=device).eval()
        # new change
        snapshot_rollout_policy = None
        if args.snapshot_rollout_jit_policy is not None:
            snapshot_rollout_policy = torch.jit.load(
                args.snapshot_rollout_jit_policy, map_location=device
            ).eval()
        command = unwrapped.command_manager.get_term("upper_body_joints")
        # new change
        candidate_action_ids, candidate_action_scales, candidate_joint_names = (
            _candidate_spec()
        )
        # new change
        support_metric_slices = _support_metric_slices(
            unwrapped.observation_manager
        )
        # new change
        # ReferenceCommand historically seeds its private generator with 42,
        # independently of env_cfg.seed.  Reseed only this external collector's
        # command instance before the reconstructing reset so separate event
        # groups truly contain different wrists, directions and timings.
        if args.require_force_context:
            command.torch_gen.manual_seed(int(args.force_event_seed))
        horizon_steps = int(round(args.outcome_horizon_s / unwrapped.step_dt))
        if not 0 < args.pulse_steps <= horizon_steps:
            raise ValueError("--pulse_steps must be inside the horizon.")
        observations = _initialize_stratified_states(
            unwrapped,
            len(command.cfg.demo_recording_path),
            args.warmup_steps + horizon_steps + 3,
            args.phase_offset,
        )
        # new change
        # Reconstruct the offset-state distribution visited by the previously
        # distilled support controller.  We retain its held residual and clock
        # as part of the controller state, so every +/- probe is paired in both
        # simulator state and temporal controller context.
        held_snapshot_residual_actor = torch.zeros(
            (args.num_envs, 29), device=device, dtype=observations["policy"].dtype
        )
        steps_since_snapshot_update = torch.zeros(
            args.num_envs, device=device, dtype=torch.long
        )
        for warmup_index in range(args.warmup_steps):
            with torch.inference_mode():
                baseline_action = policy(_policy_input(observations)).detach()
                action = baseline_action
                if snapshot_rollout_policy is not None:
                    if warmup_index % args.snapshot_residual_hold_steps == 0:
                        snapshot_input = torch.cat(
                            [observations["policy"]]
                            + [
                                observations[name]
                                for name in args.snapshot_rollout_extra_obs_groups
                            ],
                            dim=-1,
                        )
                        candidate_action = snapshot_rollout_policy(snapshot_input).detach()
                        held_snapshot_residual_actor.copy_(
                            candidate_action - baseline_action
                        )
                        steps_since_snapshot_update.zero_()
                    action = baseline_action + held_snapshot_residual_actor
            observations, _, _, _, _ = unwrapped.step(action)
            if snapshot_rollout_policy is not None:
                steps_since_snapshot_update += 1

        robot = unwrapped.scene["robot"]
        # new change
        support_foot_ids, _ = robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        body_masses = robot.data.default_mass.to(
            device=device, dtype=robot.data.body_com_pos_w.dtype
        )
        whole_body_com_w = (
            robot.data.body_com_pos_w * body_masses.unsqueeze(-1)
        ).sum(dim=1) / body_masses.sum(dim=1, keepdim=True)
        features = observations["support_adapter"].detach().clone()
        # new change
        support_ids = torch.tensor(
            candidate_action_ids, device=device, dtype=torch.long
        )
        candidate_scales = torch.tensor(
            candidate_action_scales,
            device=device,
            dtype=observations["policy"].dtype,
        )
        previous_support_residual_rad = (
            held_snapshot_residual_actor.index_select(1, support_ids)
            * candidate_scales
        )
        elapsed_fraction = (
            steps_since_snapshot_update.to(features.dtype)
            / float(max(args.snapshot_residual_hold_steps, 1))
        ).clamp(0.0, 1.0).unsqueeze(-1)
        support_temporal_features = torch.cat(
            (features, previous_support_residual_rad, elapsed_fraction), dim=-1
        )
        # new change
        force_context = (
            _online_force_context(command) if args.require_force_context else {}
        )
        # new change
        explicit_candidate = _candidate_vector(device)
        if explicit_candidate is None:
            if candidate_action_ids != SUPPORT_ACTION_IDS:
                raise ValueError(
                    "Generalized candidate rows require an explicit vector/tensor; "
                    "the legacy q-error teacher is defined only for 10DOF support."
                )
            candidate_target = torch.clamp(
                -args.teacher_gain * features[:, 20:30],
                min=-args.teacher_limit_rad,
                max=args.teacher_limit_rad,
            )
            candidate_mode = (
                "q_error_closed_loop"
                if args.closed_loop_candidate
                else "q_error_snapshot"
            )
        else:
            if args.closed_loop_candidate:
                raise ValueError(
                    "--closed_loop_candidate and --candidate_vector_rad are "
                    "mutually exclusive."
                )
            candidate_target = (
                explicit_candidate.unsqueeze(0).expand(args.num_envs, -1).clone()
                if explicit_candidate.ndim == 1
                else explicit_candidate.clone()
            )
            candidate_mode = (
                "explicit_finite_difference"
                if explicit_candidate.ndim == 1
                else "solved_per_environment_combined_teacher"
            )
        snapshot = {
            "policy_observation": observations["policy"].detach().clone(),
            "support_recovery_features": features,
            "root_state_w": robot.data.root_state_w.detach().clone(),
            "joint_pos": robot.data.joint_pos.detach().clone(),
            "joint_vel": robot.data.joint_vel.detach().clone(),
            "joint_pos_target": robot.data.joint_pos_target.detach().clone(),
            "motion_id": (command.motion_ids % len(command.cfg.demo_recording_path))
            .detach()
            .clone(),
            "motion_count": command.motion_count.detach().clone(),
            "episode_length": unwrapped.episode_length_buf.detach().clone(),
            "candidate_target_rad": candidate_target,
            # new change
            "previous_support_residual_rad": previous_support_residual_rad.detach().clone(),
            "steps_since_residual_update": steps_since_snapshot_update.detach().clone(),
            "support_temporal_features": support_temporal_features.detach().clone(),
            # new change
            "support_foot_position_w": robot.data.body_com_pos_w[
                :, support_foot_ids
            ].detach().clone(),
            "whole_body_com_w": whole_body_com_w.detach().clone(),
        }
        # new change
        for name, value in force_context.items():
            snapshot[f"force_context_{name}"] = value
        fingerprint_metadata = {
            "seed": args.seed,
            # new change
            "force_event_seed": args.force_event_seed,
            "phase_offset": args.phase_offset,
            "warmup_steps": args.warmup_steps,
            "num_envs": args.num_envs,
            "task": args.task,
            "policy": pathlib.Path(args.jit_policy).name,
            # new change
            "snapshot_rollout_policy": (
                None
                if args.snapshot_rollout_jit_policy is None
                else pathlib.Path(args.snapshot_rollout_jit_policy).name
            ),
            "snapshot_rollout_extra_obs_groups": tuple(
                args.snapshot_rollout_extra_obs_groups
            ),
            "snapshot_residual_hold_steps": args.snapshot_residual_hold_steps,
        }
        fingerprint = _snapshot_fingerprint(snapshot, fingerprint_metadata)
        start_bad_orientation, start_base_contact = _failure_terms(unwrapped)
        valid = ~(start_bad_orientation | start_base_contact)

        alive = torch.ones(args.num_envs, device=device, dtype=torch.bool)
        first_failure = torch.full(
            (args.num_envs,), horizon_steps, device=device, dtype=torch.long
        )
        trajectories = {
            name: []
            for name in (
                "cost",
                "bad_orientation",
                "base_contact",
                "root_state",
                "joint_state",
                "action",
                # new change
                "com_support_xy_distance_m",
                "torso_tilt_rad",
                # new change
                "candidate_correction_rad",
                # new change
                "support_feature",
                # new change
                "support_foot_position_w",
                "whole_body_com_w",
            )
        }
        # new change
        force_trajectories = (
            {name: [] for name in force_context}
            if args.require_force_context
            else {}
        )
        for step_index in range(horizon_steps):
            with torch.inference_mode():
                action = policy(_policy_input(observations)).detach()
                # new change
                applied_candidate = torch.zeros(
                    (args.num_envs, len(candidate_action_ids)),
                    device=device,
                    dtype=action.dtype,
                )
                if args.branch == "candidate" and step_index < args.pulse_steps:
                    if args.closed_loop_candidate:
                        current_support = observations["support_adapter"]
                        applied_candidate = torch.clamp(
                            -args.teacher_gain * current_support[:, 20:30],
                            min=-args.teacher_limit_rad,
                            max=args.teacher_limit_rad,
                        )
                    else:
                        applied_candidate = candidate_target
                    action[:, support_ids] += applied_candidate / candidate_scales
            observations, _, _, _, _ = unwrapped.step(action)
            # new change
            if args.require_force_context:
                step_force_context = _online_force_context(command)
                for name, value in step_force_context.items():
                    force_trajectories[name].append(value)
            bad_orientation, base_contact = _failure_terms(unwrapped)
            failed_now = alive & (bad_orientation | base_contact)
            first_failure[failed_now] = step_index + 1
            alive &= ~(bad_orientation | base_contact)
            trajectories["cost"].append(
                _stability_cost(observations["support_adapter"][:, 79:87])
            )
            trajectories["bad_orientation"].append(bad_orientation)
            trajectories["base_contact"].append(base_contact)
            trajectories["root_state"].append(robot.data.root_state_w.clone())
            trajectories["joint_state"].append(robot.data.joint_pos.clone())
            trajectories["action"].append(action.clone())
            # new change
            trajectories["candidate_correction_rad"].append(
                applied_candidate.clone()
            )
            # new change
            # Preserve the complete named 87D support observation.  The local
            # teacher needs signed root/foot/COM/orientation/velocity responses,
            # not only the earlier scalar cost and vector norms.
            trajectories["support_feature"].append(
                observations["support_adapter"].clone()
            )
            # new change
            trajectories["support_foot_position_w"].append(
                robot.data.body_com_pos_w[:, support_foot_ids].clone()
            )
            current_whole_body_com_w = (
                robot.data.body_com_pos_w * body_masses.unsqueeze(-1)
            ).sum(dim=1) / body_masses.sum(dim=1, keepdim=True)
            trajectories["whole_body_com_w"].append(
                current_whole_body_com_w.clone()
            )
            # new change
            com_distance_m, torso_tilt_rad = _short_horizon_support_metrics(
                observations["support_adapter"], support_metric_slices
            )
            trajectories["com_support_xy_distance_m"].append(
                com_distance_m
            )
            trajectories["torso_tilt_rad"].append(torso_tilt_rad)

        trajectories = {
            name: torch.stack(values, dim=1)
            for name, values in trajectories.items()
        }
        # new change
        force_trajectories = {
            name: torch.stack(values, dim=1)
            for name, values in force_trajectories.items()
        }
        failure_trajectory = (
            trajectories["bad_orientation"] | trajectories["base_contact"]
        )
        time_index = torch.arange(horizon_steps, device=device)[None, :]
        pre_failure = time_index < first_failure[:, None]
        pre_failure_count = pre_failure.sum(dim=1).clamp_min(1)
        pre_failure_cost = (
            trajectories["cost"] * pre_failure
        ).sum(dim=1) / pre_failure_count
        failure = failure_trajectory.any(dim=1)
        result = {
            "cost_trajectory": trajectories["cost"],
            "bad_orientation_trajectory": trajectories["bad_orientation"],
            "base_contact_trajectory": trajectories["base_contact"],
            "root_state_trajectory": trajectories["root_state"],
            "joint_state_trajectory": trajectories["joint_state"],
            "action_trajectory": trajectories["action"],
            # new change
            "candidate_correction_rad_trajectory": trajectories[
                "candidate_correction_rad"
            ],
            # new change
            "support_feature_trajectory": trajectories["support_feature"],
            # new change
            "support_foot_position_w_trajectory": trajectories[
                "support_foot_position_w"
            ],
            "whole_body_com_w_trajectory": trajectories["whole_body_com_w"],
            # new change
            "com_support_xy_distance_m_trajectory": trajectories[
                "com_support_xy_distance_m"
            ],
            "torso_tilt_rad_trajectory": trajectories["torso_tilt_rad"],
            "first_failure_step": first_failure,
            "failure": failure,
            "pre_failure_cost_mean": pre_failure_cost,
            "score": pre_failure_cost + args.failure_penalty * failure.float(),
        }
        # new change
        for name, value in force_trajectories.items():
            result[f"force_context_{name}_trajectory"] = value
        artifact = {
            "branch": args.branch,
            "snapshot_fingerprint": fingerprint,
            "snapshot_fingerprint_metadata": fingerprint_metadata,
            "snapshot": {key: value.cpu() for key, value in snapshot.items()},
            "valid": valid.cpu(),
            "result": {key: value.cpu() for key, value in result.items()},
            "support_action_ids": candidate_action_ids,
            # new change
            "candidate_action_scales_rad": candidate_action_scales,
            "candidate_joint_names": candidate_joint_names,
            "jit_extra_obs_groups": tuple(args.jit_extra_obs_groups),
            "support_joint_names": SUPPORT_JOINT_NAMES,
            "teacher_gain": args.teacher_gain,
            "teacher_limit_rad": args.teacher_limit_rad,
            # new change
            "candidate_mode": candidate_mode,
            # new change
            "closed_loop_candidate": bool(args.closed_loop_candidate),
            "explicit_candidate_vector_rad": (
                None
                if explicit_candidate is None
                else explicit_candidate.detach().cpu()
            ),
            "pulse_steps": args.pulse_steps,
            "outcome_horizon_steps": horizon_steps,
            "outcome_horizon_s": horizon_steps * unwrapped.step_dt,
            "failure_penalty": args.failure_penalty,
            # new change
            "force_conditioned_pairing": bool(args.require_force_context),
            "force_gate_feature_names": FORCE_GATE_FEATURE_NAMES,
            "force_gate_feature_dim": len(FORCE_GATE_FEATURE_NAMES),
            "force_body_names": (
                tuple(command.cfg.force_body_names)
                if args.require_force_context
                else ()
            ),
            "disabled_automatic_terminations": disabled_terminations,
            "ppo_updates": False,
            "uses_v3_checkpoint": False,
            "source_checkpoint_role": "frozen model_1375 JIT only",
            # new change
            "short_horizon_metric_schema": {
                "com_support_xy_distance_m": (
                    "norm of unscaled body-frame COM-to-support-center XY"
                ),
                "torso_tilt_rad": (
                    "acos of torso projected gravity alignment with -Z"
                ),
                "support_metric_slices": support_metric_slices,
            },
        }
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, output)
        print(
            {
                "output": str(output),
                "branch": args.branch,
                "snapshot_fingerprint": fingerprint,
                "valid": int(valid.sum()),
                "failures": int((failure & valid).sum()),
                "score_mean": float(result["score"][valid].mean()),
                "horizon_s": horizon_steps * unwrapped.step_dt,
                "ppo_updates": False,
                "force_conditioned_pairing": bool(args.require_force_context),
            },
            flush=True,
        )
    # new change
    # App shutdown may otherwise hide an exception behind a clean Kit exit,
    # leaving the orchestration layer with no artifact and no useful cause.
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
