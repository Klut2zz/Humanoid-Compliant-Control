"""Collect exact-state paired long-horizon support-recovery labels.

The frozen base policy is evaluated twice from one simulator snapshot:

1. baseline: the original policy action;
2. candidate: the same action plus one bounded support-chain pulse.

No PPO optimizer or learned V3 checkpoint is used by this script.
"""

# new change
from __future__ import annotations

import argparse
import copy
import pathlib
import sys
from dataclasses import dataclass

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
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--snapshot_batches", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--phase_offset",
    type=float,
    default=0.0,
    help="Deterministic [0,1) phase offset for independent strict-pair runs.",
)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--outcome_horizon_s", type=float, default=2.0)
parser.add_argument("--teacher_gain", type=float, default=0.10)
parser.add_argument("--teacher_limit_rad", type=float, default=0.006)
parser.add_argument("--pulse_steps", type=int, default=5)
# new change
parser.add_argument(
    "--candidate_vector_rad",
    default=None,
    help=(
        "Optional comma-separated 10DOF support residual in SUPPORT_ACTION_IDS "
        "order. When set, it replaces the q-error teacher for finite-difference "
        "controllability probes."
    ),
)
parser.add_argument("--failure_penalty", type=float, default=20.0)
parser.add_argument("--audit_steps", type=int, default=20)
parser.add_argument("--audit_action_tol", type=float, default=2.0e-5)
parser.add_argument("--audit_joint_tol", type=float, default=2.0e-5)
parser.add_argument("--audit_root_tol", type=float, default=2.0e-5)
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
def _candidate_vector(device):
    """Parse one bounded, explicit support-chain finite-difference vector."""

    import torch

    if args.candidate_vector_rad is None:
        return None
    values = [float(value) for value in args.candidate_vector_rad.split(",")]
    if len(values) != len(SUPPORT_ACTION_IDS):
        raise ValueError(
            "--candidate_vector_rad must contain exactly 10 comma-separated "
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
    """Dimensionless cost from the eight recovery-aware state terms."""

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
def _clone_direct_tensors(obj):
    """Clone tensor attributes without traversing configs or simulator handles."""

    import torch

    result = {}
    for name, value in vars(obj).items():
        if isinstance(value, torch.Tensor):
            result[name] = value.detach().clone()
    return result


# new change
def _restore_direct_tensors(obj, state):
    """Restore direct tensor attributes while retaining object identity."""

    import torch

    for name, saved in state.items():
        current = getattr(obj, name, None)
        if isinstance(current, torch.Tensor) and current.shape == saved.shape:
            current.copy_(saved)
        else:
            setattr(obj, name, saved.clone())


# new change
def _capture_observation_history(observation_manager):
    histories = {}
    for group_name, terms in observation_manager._group_obs_term_history_buffer.items():
        for term_name, buffer in terms.items():
            histories[(group_name, term_name)] = {
                "pointer": int(buffer._pointer),
                "num_pushes": buffer._num_pushes.clone(),
                "buffer": None if buffer._buffer is None else buffer._buffer.clone(),
            }
    return histories


# new change
def _restore_observation_history(observation_manager, histories):
    for (group_name, term_name), state in histories.items():
        buffer = observation_manager._group_obs_term_history_buffer[group_name][term_name]
        buffer._pointer = state["pointer"]
        buffer._num_pushes.copy_(state["num_pushes"])
        if state["buffer"] is None:
            buffer._buffer = None
        elif buffer._buffer is None or buffer._buffer.shape != state["buffer"].shape:
            buffer._buffer = state["buffer"].clone()
        else:
            buffer._buffer.copy_(state["buffer"])


# new change
@dataclass
class ExactBranchSnapshot:
    """State required to replay deterministic policy branches from one point."""

    root_state_w: object
    joint_pos: object
    joint_vel: object
    joint_pos_target: object
    joint_vel_target: object
    joint_effort_target: object
    env_tensors: dict
    command_tensors: dict
    action_manager_tensors: dict
    action_term_tensors: dict
    actuator_tensors: dict
    sensor_tensors: dict
    sensor_data_tensors: dict
    observation_history: dict
    command_rng_state: object | None
    torch_rng_state: object
    cuda_rng_state: list
    common_step_counter: int
    sim_step_counter: int

    @classmethod
    def capture(cls, env):
        import torch

        robot = env.scene["robot"]
        command = env.command_manager.get_term("upper_body_joints")
        action_terms = {
            name: _clone_direct_tensors(term)
            for name, term in env.action_manager._terms.items()
        }
        actuator_tensors = {
            name: _clone_direct_tensors(actuator)
            for name, actuator in robot.actuators.items()
        }
        sensor_tensors = {
            name: _clone_direct_tensors(sensor)
            for name, sensor in env.scene.sensors.items()
        }
        sensor_data_tensors = {
            name: _clone_direct_tensors(sensor.data)
            for name, sensor in env.scene.sensors.items()
        }
        command_rng_state = None
        if hasattr(command, "torch_gen"):
            command_rng_state = command.torch_gen.get_state().clone()
        return cls(
            root_state_w=robot.data.root_state_w.detach().clone(),
            joint_pos=robot.data.joint_pos.detach().clone(),
            joint_vel=robot.data.joint_vel.detach().clone(),
            joint_pos_target=robot.data.joint_pos_target.detach().clone(),
            joint_vel_target=robot.data.joint_vel_target.detach().clone(),
            joint_effort_target=robot.data.joint_effort_target.detach().clone(),
            env_tensors=_clone_direct_tensors(env),
            command_tensors=_clone_direct_tensors(command),
            action_manager_tensors=_clone_direct_tensors(env.action_manager),
            action_term_tensors=action_terms,
            actuator_tensors=actuator_tensors,
            sensor_tensors=sensor_tensors,
            sensor_data_tensors=sensor_data_tensors,
            observation_history=_capture_observation_history(env.observation_manager),
            command_rng_state=command_rng_state,
            torch_rng_state=torch.get_rng_state().clone(),
            cuda_rng_state=[state.clone() for state in torch.cuda.get_rng_state_all()],
            common_step_counter=int(env.common_step_counter),
            sim_step_counter=int(env._sim_step_counter),
        )

    def restore(self, env):
        import torch

        robot = env.scene["robot"]
        command = env.command_manager.get_term("upper_body_joints")

        robot.write_root_state_to_sim(self.root_state_w)
        robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
        robot.set_joint_position_target(self.joint_pos_target)
        robot.set_joint_velocity_target(self.joint_vel_target)
        robot.set_joint_effort_target(self.joint_effort_target)

        _restore_direct_tensors(env, self.env_tensors)
        _restore_direct_tensors(command, self.command_tensors)
        _restore_direct_tensors(env.action_manager, self.action_manager_tensors)
        for name, state in self.action_term_tensors.items():
            _restore_direct_tensors(env.action_manager._terms[name], state)
        for name, state in self.actuator_tensors.items():
            _restore_direct_tensors(robot.actuators[name], state)

        env.scene.write_data_to_sim()
        env.sim.forward()

        for name, state in self.sensor_tensors.items():
            _restore_direct_tensors(env.scene.sensors[name], state)
        for name, state in self.sensor_data_tensors.items():
            _restore_direct_tensors(env.scene.sensors[name].data, state)
        _restore_observation_history(env.observation_manager, self.observation_history)

        if self.command_rng_state is not None:
            command.torch_gen.set_state(self.command_rng_state)
        torch.set_rng_state(self.torch_rng_state)
        torch.cuda.set_rng_state_all(self.cuda_rng_state)
        env.common_step_counter = self.common_step_counter
        env._sim_step_counter = self.sim_step_counter

        # Compute the observation without appending a second history sample.
        env.obs_buf = env.observation_manager.compute(update_history=False)
        return env.obs_buf


# new change
def _failure_terms(env):
    """Reproduce task failures without allowing ManagerBasedRLEnv to auto-reset."""

    import torch
    import torch.nn.functional as functional

    robot = env.scene["robot"]
    command = env.command_manager.get_term("upper_body_joints")
    current_gravity = functional.normalize(robot.data.projected_gravity_b, dim=-1)
    reference_gravity = functional.normalize(
        command.root_state["projected_gravity_b"][:, 1], dim=-1
    )
    cosine = torch.sum(current_gravity * reference_gravity, dim=-1).clamp(-1.0, 1.0)
    bad_orientation = torch.acos(cosine).abs() > 0.7853981633974483

    contact_sensor = env.scene["contact_forces"]
    body_ids, _ = contact_sensor.find_bodies(["pelvis", "torso_link"], preserve_order=True)
    contact_norm = torch.linalg.vector_norm(
        contact_sensor.data.net_forces_w_history[:, :, body_ids, :], dim=-1
    )
    base_contact = torch.any(contact_norm > 1.0, dim=(1, 2))
    return bad_orientation, base_contact


# new change
def _run_branch(env, observations, policy, support_ids, candidate, horizon_steps, pulse_steps):
    """Run one branch and retain the full long-horizon outcome trajectory."""

    import torch

    num_envs = observations["policy"].shape[0]
    alive = torch.ones(num_envs, device=env.device, dtype=torch.bool)
    first_failure_step = torch.full(
        (num_envs,), horizon_steps, device=env.device, dtype=torch.long
    )
    cost_trajectory = []
    orientation_trajectory = []
    base_contact_trajectory = []
    root_state_trajectory = []
    joint_state_trajectory = []
    action_trajectory = []

    for step_index in range(horizon_steps):
        with torch.inference_mode():
            action = policy(observations["policy"]).detach()
            if candidate is not None and step_index < pulse_steps:
                action[:, support_ids] += candidate / 0.30
        observations, _, _, _, _ = env.step(action)
        bad_orientation, base_contact = _failure_terms(env)
        failed_now = alive & (bad_orientation | base_contact)
        first_failure_step[failed_now] = step_index + 1
        alive &= ~(bad_orientation | base_contact)

        trend = observations["support_adapter"][:, 79:87]
        cost_trajectory.append(_stability_cost(trend))
        orientation_trajectory.append(bad_orientation)
        base_contact_trajectory.append(base_contact)
        root_state_trajectory.append(env.scene["robot"].data.root_state_w.clone())
        joint_state_trajectory.append(env.scene["robot"].data.joint_pos.clone())
        action_trajectory.append(action.clone())

    cost = torch.stack(cost_trajectory, dim=1)
    bad_orientation = torch.stack(orientation_trajectory, dim=1)
    base_contact = torch.stack(base_contact_trajectory, dim=1)
    failure = bad_orientation | base_contact
    steps = torch.arange(horizon_steps, device=env.device)[None, :]
    pre_failure = steps < first_failure_step[:, None]
    valid_count = pre_failure.sum(dim=1).clamp_min(1)
    pre_failure_mean = (cost * pre_failure).sum(dim=1) / valid_count
    score = pre_failure_mean + args.failure_penalty * failure.any(dim=1).float()
    return {
        "cost_trajectory": cost,
        "bad_orientation_trajectory": bad_orientation,
        "base_contact_trajectory": base_contact,
        "first_failure_step": first_failure_step,
        "failure": failure.any(dim=1),
        "pre_failure_cost_mean": pre_failure_mean,
        "score": score,
        "root_state_trajectory": torch.stack(root_state_trajectory, dim=1),
        "joint_state_trajectory": torch.stack(joint_state_trajectory, dim=1),
        "action_trajectory": torch.stack(action_trajectory, dim=1),
        "final_observations": observations,
    }


# new change
def _audit_snapshot_replay(env, observations, policy, support_ids, snapshot, steps):
    """Require baseline replay to match after a restore before collecting labels."""

    import torch

    start_observations = {
        key: value.detach().clone() for key, value in observations.items()
    }
    first = _run_branch(env, observations, policy, support_ids, None, steps, 0)
    restored_observations = snapshot.restore(env)
    initial_observation_deltas = {
        key: float((start_observations[key] - restored_observations[key]).abs().max())
        for key in start_observations
    }
    second = _run_branch(
        env, restored_observations, policy, support_ids, None, steps, 0
    )
    action_step_max = torch.amax(
        (first["action_trajectory"] - second["action_trajectory"]).abs(), dim=(0, 2)
    )
    joint_step_max = torch.amax(
        (first["joint_state_trajectory"] - second["joint_state_trajectory"]).abs(),
        dim=(0, 2),
    )
    root_step_max = torch.amax(
        (first["root_state_trajectory"] - second["root_state_trajectory"]).abs(),
        dim=(0, 2),
    )
    deltas = {
        "initial_observation_max": max(initial_observation_deltas.values()),
        "initial_observation_by_group": initial_observation_deltas,
        "action_max": float(
            (first["action_trajectory"] - second["action_trajectory"]).abs().max()
        ),
        "joint_max": float(
            (first["joint_state_trajectory"] - second["joint_state_trajectory"])
            .abs()
            .max()
        ),
        "root_max": float(
            (first["root_state_trajectory"] - second["root_state_trajectory"])
            .abs()
            .max()
        ),
        "cost_max": float(
            (first["cost_trajectory"] - second["cost_trajectory"]).abs().max()
        ),
        "failure_mismatch": int(
            (first["failure"] != second["failure"]).sum().item()
        ),
        "action_step_max": action_step_max.cpu().tolist(),
        "joint_step_max": joint_step_max.cpu().tolist(),
        "root_step_max": root_step_max.cpu().tolist(),
    }
    passed = (
        deltas["action_max"] <= args.audit_action_tol
        and deltas["joint_max"] <= args.audit_joint_tol
        and deltas["root_max"] <= args.audit_root_tol
        and deltas["failure_mismatch"] == 0
    )
    print({"snapshot_replay_audit": deltas, "passed": passed}, flush=True)
    if not passed:
        raise RuntimeError(
            "Exact snapshot replay audit failed; refusing to generate causal labels."
        )


# new change
def _disable_automatic_terminations(env_cfg):
    """Keep fallen branches observable instead of silently resetting them."""

    disabled = []
    for name in list(vars(env_cfg.terminations)):
        if name.startswith("_"):
            continue
        if getattr(env_cfg.terminations, name) is not None:
            setattr(env_cfg.terminations, name, None)
            disabled.append(name)
    return disabled


# new change
def _configure_first_step_pairing(env_cfg):
    """Make every pair identical before PhysX performs its first integration."""

    # PhysX otherwise permits contact-island ordering to diverge across
    # translated clones even when their exposed states and actions are equal.
    env_cfg.sim.physx.enable_enhanced_determinism = True
    # Data-collection clones are collision-filtered from one another.  Using
    # one world origin removes translation-dependent contact round-off from
    # the counterfactual comparison; this does not alter the training task.
    env_cfg.scene.env_spacing = 0.0
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
def _initialize_stratified_pair_states(
    env,
    source_ids,
    candidate_ids,
    motion_file_count,
    reserve_steps,
    phase_offset,
):
    """Reset each twin pair once at an identical stratified motion phase."""

    import torch

    command = env.command_manager.get_term("upper_body_joints")
    pair_count = source_ids.numel()
    pairs_per_motion = pair_count // motion_file_count
    pair_rank = torch.div(source_ids, motion_file_count, rounding_mode="floor")
    phase_fraction = (
        (pair_rank.float() + 0.5) / float(pairs_per_motion) + phase_offset
    ) % 1.0
    maximum_times = command.motion_lib.get_max_times(source_ids)
    usable_times = (maximum_times - reserve_steps * env.step_dt).clamp_min(0.0)
    motion_count = torch.floor(
        phase_fraction * usable_times / command.motion_dt
    ).to(command.motion_count.dtype)
    command.motion_count[source_ids] = motion_count
    command.motion_count[candidate_ids] = motion_count
    command.motion_start[source_ids] = 0
    command.motion_start[candidate_ids] = 0
    command.motion_durations[source_ids] = maximum_times
    command.motion_durations[candidate_ids] = maximum_times
    command._update_command(step=False)

    all_ids = torch.cat((source_ids, candidate_ids))
    env._reset_idx(all_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.obs_buf = env.observation_manager.compute(update_history=True)
    return env.obs_buf


# new change
def _main_sequential_rewind_rejected() -> None:
    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils import parse_env_cfg

    import softmimic_gym.tasks  # noqa: F401

    if not 2.0 <= args.outcome_horizon_s <= 5.0:
        raise ValueError("--outcome_horizon_s must be inside the requested [2, 5] s range.")
    if args.snapshot_batches != 1 or args.num_envs < 1:
        raise ValueError(
            "Strict PhysX pairing requires --snapshot_batches 1. Collect "
            "additional batches in fresh processes with --phase_offset."
        )
    if not 0.0 <= args.phase_offset < 1.0:
        raise ValueError("--phase_offset must be in [0, 1).")
    if args.teacher_gain <= 0.0 or args.teacher_limit_rad <= 0.0:
        raise ValueError("Teacher gain and limit must be positive.")

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.recorders = None
    disabled_terminations = _disable_automatic_terminations(env_cfg)
    if hasattr(env_cfg.observations.policy, "enable_corruption"):
        env_cfg.observations.policy.enable_corruption = False
    env = gym.make(args.task, cfg=env_cfg)
    try:
        observations, _ = env.reset()
        unwrapped = env.unwrapped
        device = unwrapped.device
        policy = torch.jit.load(args.jit_policy, map_location=device).eval()
        support_ids = torch.tensor(SUPPORT_ACTION_IDS, device=device, dtype=torch.long)
        horizon_steps = int(round(args.outcome_horizon_s / unwrapped.step_dt))
        actual_horizon_s = horizon_steps * unwrapped.step_dt
        if not 0 < args.pulse_steps <= horizon_steps:
            raise ValueError("--pulse_steps must be within the outcome horizon.")

        print(
            {
                "mode": "exact_snapshot_paired_long_horizon",
                "frozen_policy": str(pathlib.Path(args.jit_policy)),
                "uses_v3_checkpoint": False,
                "ppo_updates": False,
                "step_dt": float(unwrapped.step_dt),
                "horizon_steps": horizon_steps,
                "horizon_s": float(actual_horizon_s),
                "disabled_automatic_terminations": disabled_terminations,
            },
            flush=True,
        )

        feature_batches = []
        candidate_batches = []
        baseline_batches = []
        treatment_batches = []
        motion_batches = []
        source_time_batches = []
        valid_batches = []
        replay_audited = False

        for batch_index in range(args.snapshot_batches):
            if batch_index > 0:
                observations, _ = env.reset()
            for _ in range(args.warmup_steps):
                with torch.inference_mode():
                    action = policy(observations["policy"])
                observations, _, _, _, _ = env.step(action)

            command = unwrapped.command_manager.get_term("upper_body_joints")
            features = observations["support_adapter"].detach().clone()
            if features.shape[-1] != 87:
                raise RuntimeError(
                    f"Expected 87D recovery state, got {features.shape[-1]}."
                )
            candidate = torch.clamp(
                -args.teacher_gain * features[:, 20:30],
                min=-args.teacher_limit_rad,
                max=args.teacher_limit_rad,
            )
            start_bad_orientation, start_base_contact = _failure_terms(unwrapped)
            remaining_motion_steps = (
                command.motion_durations / command.motion_dt - command.motion_count
            )
            remaining_episode_steps = (
                unwrapped.max_episode_length - unwrapped.episode_length_buf
            )
            valid = (
                ~start_bad_orientation
                & ~start_base_contact
                & (remaining_motion_steps > horizon_steps + 2)
                & (remaining_episode_steps > horizon_steps + 2)
            )
            snapshot = ExactBranchSnapshot.capture(unwrapped)

            if not replay_audited and args.audit_steps > 0:
                _audit_snapshot_replay(
                    unwrapped,
                    observations,
                    policy,
                    support_ids,
                    snapshot,
                    min(args.audit_steps, horizon_steps),
                )
                observations = snapshot.restore(unwrapped)
                replay_audited = True

            baseline = _run_branch(
                unwrapped,
                observations,
                policy,
                support_ids,
                None,
                horizon_steps,
                args.pulse_steps,
            )
            observations = snapshot.restore(unwrapped)
            treatment = _run_branch(
                unwrapped,
                observations,
                policy,
                support_ids,
                candidate,
                horizon_steps,
                args.pulse_steps,
            )

            feature_batches.append(features.cpu())
            candidate_batches.append(candidate.cpu())
            valid_batches.append(valid.cpu())
            motion_batches.append(command.motion_ids.detach().clone().cpu())
            source_time_batches.append(
                (command.motion_count * command.motion_dt).detach().clone().cpu()
            )
            baseline_batches.append(
                {key: value.cpu() for key, value in baseline.items() if key != "final_observations"}
            )
            treatment_batches.append(
                {key: value.cpu() for key, value in treatment.items() if key != "final_observations"}
            )
            print(
                {
                    "snapshot_batch": batch_index + 1,
                    "valid": int(valid.sum()),
                    "baseline_failures": int((baseline["failure"] & valid).sum()),
                    "candidate_failures": int((treatment["failure"] & valid).sum()),
                    "paired_score_delta_mean": float(
                        (treatment["score"][valid] - baseline["score"][valid]).mean()
                    ) if valid.any() else float("nan"),
                },
                flush=True,
            )

        valid = torch.cat(valid_batches, dim=0)

        def concatenate_branch(batches):
            return {
                key: torch.cat([batch[key] for batch in batches], dim=0)
                for key in batches[0]
            }

        baseline = concatenate_branch(baseline_batches)
        treatment = concatenate_branch(treatment_batches)
        score_delta = treatment["score"] - baseline["score"]
        failure_delta = treatment["failure"].to(torch.int8) - baseline[
            "failure"
        ].to(torch.int8)
        beneficial = (failure_delta < 0) | (
            (failure_delta == 0) & (score_delta < 0.0)
        )
        dataset = {
            "support_recovery_features": torch.cat(feature_batches, dim=0),
            "candidate_target_rad": torch.cat(candidate_batches, dim=0),
            "valid": valid,
            "baseline": baseline,
            "candidate": treatment,
            "paired_score_delta": score_delta,
            "paired_failure_delta": failure_delta,
            "candidate_beneficial": beneficial,
            "motion_id": torch.cat(motion_batches, dim=0),
            "source_time_s": torch.cat(source_time_batches, dim=0),
            "support_action_ids": SUPPORT_ACTION_IDS,
            "support_joint_names": SUPPORT_JOINT_NAMES,
            "teacher_gain": args.teacher_gain,
            "teacher_limit_rad": args.teacher_limit_rad,
            "pulse_steps": args.pulse_steps,
            "outcome_horizon_steps": horizon_steps,
            "outcome_horizon_s": float(actual_horizon_s),
            "failure_penalty": args.failure_penalty,
            "snapshot_batches": args.snapshot_batches,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "strict_pairing": True,
            "same_simulator_snapshot": True,
            "ppo_updates": False,
            "source_checkpoint_role": "frozen model_1375 JIT only",
            "feature_order": (
                "support_v2[79],com_distance,com_outward_velocity,torso_tilt,"
                "tilt_growth,root_xy_speed,max_foot_xy_speed,min_support_load,"
                "support_load_imbalance"
            ),
        }
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, output)
        valid_score_delta = score_delta[valid]
        summary = {
            "output": str(output),
            "samples_total": int(valid.numel()),
            "samples_valid": int(valid.sum()),
            "baseline_failures": int((baseline["failure"] & valid).sum()),
            "candidate_failures": int((treatment["failure"] & valid).sum()),
            "candidate_beneficial_fraction": float(beneficial[valid].float().mean()),
            "paired_score_delta_mean": float(valid_score_delta.mean()),
            "paired_score_delta_median": float(valid_score_delta.median()),
            "horizon_s": float(actual_horizon_s),
        }
        print(summary, flush=True)
    finally:
        env.close()
        simulation_app.close()


# new change
def _copy_pair_rows(obj, source_ids, candidate_ids, excluded=()):
    """Copy batched tensor rows from source twins to candidate twins."""

    import torch

    for name, value in vars(obj).items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == source_ids.numel() + candidate_ids.numel()
            and name not in excluded
        ):
            value[candidate_ids] = value[source_ids].clone()


# new change
def _copy_pair_observation_history(observation_manager, source_ids, candidate_ids):
    for terms in observation_manager._group_obs_term_history_buffer.values():
        for buffer in terms.values():
            buffer._num_pushes[candidate_ids] = buffer._num_pushes[source_ids]
            if buffer._buffer is not None:
                buffer._buffer[:, candidate_ids] = buffer._buffer[:, source_ids].clone()


# new change
def _synchronize_physics_twins(env, source_ids, candidate_ids):
    """Replicate one state into two translated, independently integrated twins."""

    robot = env.scene["robot"]
    command = env.command_manager.get_term("upper_body_joints")
    origin_delta = (
        env.scene.env_origins[candidate_ids] - env.scene.env_origins[source_ids]
    )

    root_state = robot.data.root_state_w[source_ids].clone()
    root_state[:, :3] += origin_delta
    robot.write_root_state_to_sim(root_state, env_ids=candidate_ids)
    robot.write_joint_state_to_sim(
        robot.data.joint_pos[source_ids].clone(),
        robot.data.joint_vel[source_ids].clone(),
        env_ids=candidate_ids,
    )
    robot.set_joint_position_target(
        robot.data.joint_pos_target[source_ids].clone(), env_ids=candidate_ids
    )
    robot.set_joint_velocity_target(
        robot.data.joint_vel_target[source_ids].clone(), env_ids=candidate_ids
    )
    robot.set_joint_effort_target(
        robot.data.joint_effort_target[source_ids].clone(), env_ids=candidate_ids
    )

    _copy_pair_rows(env, source_ids, candidate_ids)
    _copy_pair_rows(
        command,
        source_ids,
        candidate_ids,
        excluded=(
            "motion_ids",
            "current_root_pose_buf",
            "commanded_root_pose_buf",
        ),
    )
    _copy_pair_rows(env.action_manager, source_ids, candidate_ids)
    for term in env.action_manager._terms.values():
        _copy_pair_rows(term, source_ids, candidate_ids)
    for actuator in robot.actuators.values():
        _copy_pair_rows(actuator, source_ids, candidate_ids)
    for sensor in env.scene.sensors.values():
        _copy_pair_rows(sensor, source_ids, candidate_ids)
        _copy_pair_rows(
            sensor.data,
            source_ids,
            candidate_ids,
            excluded=("pos_w", "contact_pos_w"),
        )
    _copy_pair_observation_history(
        env.observation_manager, source_ids, candidate_ids
    )

    # motion_ids remain the physical environment IDs.  A pair stride that is
    # divisible by the number of files guarantees the same reference file;
    # copied phase/duration tensors then select exactly the same source frame.
    command._update_command(step=False)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.obs_buf = env.observation_manager.compute(update_history=False)
    return env.obs_buf


# new change
def _local_root_state(env, env_ids):
    root_state = env.scene["robot"].data.root_state_w[env_ids].clone()
    root_state[:, :3] -= env.scene.env_origins[env_ids]
    return root_state


# new change
def _audit_physics_twins(
    env, observations, policy, source_ids, candidate_ids, audit_steps
):
    """Run baseline/baseline twins forward; reject any non-equivalent batch."""

    import torch

    observation_delta = float(
        (observations["policy"][source_ids] - observations["policy"][candidate_ids])
        .abs()
        .max()
    )
    support_observation_delta = float(
        (
            observations["support_adapter"][source_ids]
            - observations["support_adapter"][candidate_ids]
        )
        .abs()
        .max()
    )
    initial_joint_delta = float(
        (
            env.scene["robot"].data.joint_pos[source_ids]
            - env.scene["robot"].data.joint_pos[candidate_ids]
        )
        .abs()
        .max()
    )
    initial_root_delta = float(
        (_local_root_state(env, source_ids) - _local_root_state(env, candidate_ids))
        .abs()
        .max()
    )
    robot_data = env.scene["robot"].data
    dynamics_pair_deltas = {}
    for name in (
        "joint_pos_target",
        "joint_vel_target",
        "joint_effort_target",
        "joint_stiffness",
        "joint_damping",
        "joint_friction_coeff",
        "joint_armature",
        "body_mass",
        "body_com_pos_b",
        "body_com_quat_b",
    ):
        value = getattr(robot_data, name, None)
        if value is not None:
            dynamics_pair_deltas[name] = float(
                (value[source_ids] - value[candidate_ids]).abs().max()
            )
    contact_sensor = env.scene["contact_forces"]
    dynamics_pair_deltas["contact_force"] = float(
        (
            contact_sensor.data.net_forces_w[source_ids]
            - contact_sensor.data.net_forces_w[candidate_ids]
        )
        .abs()
        .max()
    )
    action_step_max = []
    joint_step_max = []
    root_step_max = []
    failure_mismatch = 0
    for _ in range(audit_steps):
        with torch.inference_mode():
            action = policy(observations["policy"]).detach()
        action_step_max.append(
            float((action[source_ids] - action[candidate_ids]).abs().max())
        )
        observations, _, _, _, _ = env.step(action)
        joint_step_max.append(
            float(
                (
                    env.scene["robot"].data.joint_pos[source_ids]
                    - env.scene["robot"].data.joint_pos[candidate_ids]
                )
                .abs()
                .max()
            )
        )
        root_step_max.append(
            float(
                (
                    _local_root_state(env, source_ids)
                    - _local_root_state(env, candidate_ids)
                )
                .abs()
                .max()
            )
        )
        bad_orientation, base_contact = _failure_terms(env)
        failure_mismatch += int(
            (
                (bad_orientation[source_ids] | base_contact[source_ids])
                != (bad_orientation[candidate_ids] | base_contact[candidate_ids])
            )
            .sum()
            .item()
        )

    audit = {
        "pair_policy_observation_max": observation_delta,
        "pair_support_observation_max": support_observation_delta,
        "initial_joint_max": initial_joint_delta,
        "initial_root_max": initial_root_delta,
        "initial_dynamics_by_field": dynamics_pair_deltas,
        "action_max": max(action_step_max, default=0.0),
        "joint_max": max(joint_step_max, default=0.0),
        "root_max": max(root_step_max, default=0.0),
        "failure_mismatch": failure_mismatch,
        "action_step_max": action_step_max,
        "joint_step_max": joint_step_max,
        "root_step_max": root_step_max,
    }
    passed = (
        observation_delta <= args.audit_action_tol
        and support_observation_delta <= args.audit_joint_tol
        and initial_joint_delta <= args.audit_joint_tol
        and initial_root_delta <= args.audit_root_tol
        and max(dynamics_pair_deltas.values(), default=0.0) <= args.audit_root_tol
        and audit["action_max"] <= args.audit_action_tol
        and audit["joint_max"] <= args.audit_joint_tol
        and audit["root_max"] <= args.audit_root_tol
        and failure_mismatch == 0
    )
    print({"twin_snapshot_audit": audit, "passed": passed}, flush=True)
    if not passed:
        raise RuntimeError(
            "Synchronized twin audit failed; refusing to generate paired labels."
        )
    return observations, audit


# new change
def _run_simultaneous_paired_branches(
    env,
    observations,
    policy,
    source_ids,
    candidate_ids,
    support_ids,
    candidate,
    horizon_steps,
    pulse_steps,
):
    """Advance baseline and candidate twins in the same physics steps."""

    import torch

    pair_count = source_ids.numel()
    baseline_alive = torch.ones(pair_count, device=env.device, dtype=torch.bool)
    candidate_alive = torch.ones_like(baseline_alive)
    baseline_first_failure = torch.full(
        (pair_count,), horizon_steps, device=env.device, dtype=torch.long
    )
    candidate_first_failure = torch.full_like(
        baseline_first_failure, horizon_steps
    )
    trajectory = {
        branch: {
            key: []
            for key in (
                "cost",
                "bad_orientation",
                "base_contact",
                "root_state",
                "joint_state",
                "action",
            )
        }
        for branch in ("baseline", "candidate")
    }

    for step_index in range(horizon_steps):
        with torch.inference_mode():
            action = policy(observations["policy"]).detach()
            if step_index < pulse_steps:
                action[candidate_ids[:, None], support_ids[None, :]] += (
                    candidate / 0.30
                )
        observations, _, _, _, _ = env.step(action)
        bad_orientation, base_contact = _failure_terms(env)

        baseline_failed = (
            bad_orientation[source_ids] | base_contact[source_ids]
        )
        candidate_failed = (
            bad_orientation[candidate_ids] | base_contact[candidate_ids]
        )
        baseline_new = baseline_alive & baseline_failed
        candidate_new = candidate_alive & candidate_failed
        baseline_first_failure[baseline_new] = step_index + 1
        candidate_first_failure[candidate_new] = step_index + 1
        baseline_alive &= ~baseline_failed
        candidate_alive &= ~candidate_failed

        trend = observations["support_adapter"][:, 79:87]
        for branch, ids in (
            ("baseline", source_ids),
            ("candidate", candidate_ids),
        ):
            trajectory[branch]["cost"].append(_stability_cost(trend[ids]))
            trajectory[branch]["bad_orientation"].append(bad_orientation[ids])
            trajectory[branch]["base_contact"].append(base_contact[ids])
            trajectory[branch]["root_state"].append(_local_root_state(env, ids))
            trajectory[branch]["joint_state"].append(
                env.scene["robot"].data.joint_pos[ids].clone()
            )
            trajectory[branch]["action"].append(action[ids].clone())

    result = {}
    time_indices = torch.arange(horizon_steps, device=env.device)[None, :]
    for branch, first_failure in (
        ("baseline", baseline_first_failure),
        ("candidate", candidate_first_failure),
    ):
        branch_trajectory = {
            key: torch.stack(values, dim=1)
            for key, values in trajectory[branch].items()
        }
        failure = (
            branch_trajectory["bad_orientation"]
            | branch_trajectory["base_contact"]
        )
        pre_failure = time_indices < first_failure[:, None]
        valid_count = pre_failure.sum(dim=1).clamp_min(1)
        pre_failure_mean = (
            branch_trajectory["cost"] * pre_failure
        ).sum(dim=1) / valid_count
        result[branch] = {
            "cost_trajectory": branch_trajectory["cost"],
            "bad_orientation_trajectory": branch_trajectory["bad_orientation"],
            "base_contact_trajectory": branch_trajectory["base_contact"],
            "root_state_trajectory": branch_trajectory["root_state"],
            "joint_state_trajectory": branch_trajectory["joint_state"],
            "action_trajectory": branch_trajectory["action"],
            "first_failure_step": first_failure,
            "failure": failure.any(dim=1),
            "pre_failure_cost_mean": pre_failure_mean,
            "score": pre_failure_mean
            + args.failure_penalty * failure.any(dim=1).float(),
        }
    return observations, result


# new change
def main() -> None:
    """Collect strict pairs with simultaneous, synchronized physics twins."""

    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils import parse_env_cfg

    import softmimic_gym.tasks  # noqa: F401

    if not 2.0 <= args.outcome_horizon_s <= 5.0:
        raise ValueError("--outcome_horizon_s must be inside [2, 5] seconds.")
    if args.snapshot_batches != 1 or args.num_envs < 1:
        raise ValueError(
            "Strict PhysX pairing requires --snapshot_batches 1. Collect "
            "additional batches in fresh processes with --phase_offset."
        )
    if not 0.0 <= args.phase_offset < 1.0:
        raise ValueError("--phase_offset must be in [0, 1).")
    if args.teacher_gain <= 0.0 or args.teacher_limit_rad <= 0.0:
        raise ValueError("Teacher gain and limit must be positive.")

    # `num_envs` means number of causal pairs at the CLI.  Internally each
    # state is represented by one baseline and one candidate physics twin.
    pair_count = args.num_envs
    total_envs = pair_count * 2
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=total_envs)
    env_cfg.seed = args.seed
    env_cfg.recorders = None
    disabled_terminations = _disable_automatic_terminations(env_cfg)
    _configure_first_step_pairing(env_cfg)
    if hasattr(env_cfg.observations.policy, "enable_corruption"):
        env_cfg.observations.policy.enable_corruption = False
    env = gym.make(args.task, cfg=env_cfg)
    try:
        unwrapped = env.unwrapped
        device = unwrapped.device
        policy = torch.jit.load(args.jit_policy, map_location=device).eval()
        command = unwrapped.command_manager.get_term("upper_body_joints")
        motion_file_count = len(command.cfg.demo_recording_path)
        if pair_count % motion_file_count != 0:
            raise ValueError(
                f"--num_envs (pair count) must be divisible by {motion_file_count} "
                "so every twin uses the same reference file."
            )
        source_ids = torch.arange(pair_count, device=device, dtype=torch.long)
        candidate_ids = source_ids + pair_count
        support_ids = torch.tensor(SUPPORT_ACTION_IDS, device=device, dtype=torch.long)
        horizon_steps = int(round(args.outcome_horizon_s / unwrapped.step_dt))
        actual_horizon_s = horizon_steps * unwrapped.step_dt
        if not 0 < args.pulse_steps <= horizon_steps:
            raise ValueError("--pulse_steps must be inside the horizon.")
        observations = _initialize_stratified_pair_states(
            unwrapped,
            source_ids,
            candidate_ids,
            motion_file_count,
            args.audit_steps + args.warmup_steps + horizon_steps + 3,
            args.phase_offset,
        )

        print(
            {
                "mode": "simultaneous_synchronized_physics_twins",
                "strict_pairing": True,
                "frozen_policy": str(pathlib.Path(args.jit_policy)),
                "uses_v3_checkpoint": False,
                "ppo_updates": False,
                "pair_count": pair_count,
                "physics_env_count": total_envs,
                "step_dt": float(unwrapped.step_dt),
                "horizon_steps": horizon_steps,
                "horizon_s": float(actual_horizon_s),
                "phase_offset": args.phase_offset,
                "disabled_automatic_terminations": disabled_terminations,
            },
            flush=True,
        )

        features_all = []
        candidate_target_all = []
        valid_all = []
        motion_all = []
        source_time_all = []
        baseline_all = []
        candidate_all = []
        audit_all = []

        for batch_index in range(args.snapshot_batches):
            observations, audit = _audit_physics_twins(
                unwrapped,
                observations,
                policy,
                source_ids,
                candidate_ids,
                args.audit_steps,
            )
            audit_all.append(audit)

            for _ in range(args.warmup_steps):
                with torch.inference_mode():
                    action = policy(observations["policy"]).detach()
                observations, _, _, _, _ = unwrapped.step(action)

            pair_feature_delta = float(
                (
                    observations["support_adapter"][source_ids]
                    - observations["support_adapter"][candidate_ids]
                )
                .abs()
                .max()
            )
            if pair_feature_delta > args.audit_joint_tol:
                raise RuntimeError(
                    "Twin support features diverged during common warmup: "
                    f"max={pair_feature_delta:.6g}."
                )
            features = observations["support_adapter"][source_ids].detach().clone()
            # new change
            explicit_candidate = _candidate_vector(device)
            if explicit_candidate is None:
                candidate_target = torch.clamp(
                    -args.teacher_gain * features[:, 20:30],
                    min=-args.teacher_limit_rad,
                    max=args.teacher_limit_rad,
                )
                candidate_mode = "q_error_teacher"
            else:
                candidate_target = explicit_candidate.unsqueeze(0).expand(
                    pair_count, -1
                ).clone()
                candidate_mode = "explicit_finite_difference"
            start_bad_orientation, start_base_contact = _failure_terms(unwrapped)
            remaining_motion_steps = (
                command.motion_durations[source_ids] / command.motion_dt
                - command.motion_count[source_ids]
            )
            valid = (
                ~start_bad_orientation[source_ids]
                & ~start_base_contact[source_ids]
                & ~start_bad_orientation[candidate_ids]
                & ~start_base_contact[candidate_ids]
                & (remaining_motion_steps > horizon_steps + 2)
            )
            motion_id = (
                command.motion_ids[source_ids] % motion_file_count
            ).detach().clone()
            source_time = (
                command.motion_count[source_ids] * command.motion_dt
            ).detach().clone()

            observations, paired = _run_simultaneous_paired_branches(
                unwrapped,
                observations,
                policy,
                source_ids,
                candidate_ids,
                support_ids,
                candidate_target,
                horizon_steps,
                args.pulse_steps,
            )
            features_all.append(features.cpu())
            candidate_target_all.append(candidate_target.cpu())
            valid_all.append(valid.cpu())
            motion_all.append(motion_id.cpu())
            source_time_all.append(source_time.cpu())
            baseline_all.append(
                {key: value.cpu() for key, value in paired["baseline"].items()}
            )
            candidate_all.append(
                {key: value.cpu() for key, value in paired["candidate"].items()}
            )
            score_delta = paired["candidate"]["score"] - paired["baseline"]["score"]
            print(
                {
                    "snapshot_batch": batch_index + 1,
                    "valid": int(valid.sum()),
                    "baseline_failures": int(
                        (paired["baseline"]["failure"] & valid).sum()
                    ),
                    "candidate_failures": int(
                        (paired["candidate"]["failure"] & valid).sum()
                    ),
                    "paired_score_delta_mean": float(score_delta[valid].mean())
                    if valid.any()
                    else float("nan"),
                },
                flush=True,
            )

        def concatenate_branch(batches):
            return {
                key: torch.cat([batch[key] for batch in batches], dim=0)
                for key in batches[0]
            }

        valid = torch.cat(valid_all, dim=0)
        baseline = concatenate_branch(baseline_all)
        candidate_branch = concatenate_branch(candidate_all)
        score_delta = candidate_branch["score"] - baseline["score"]
        failure_delta = candidate_branch["failure"].to(torch.int8) - baseline[
            "failure"
        ].to(torch.int8)
        beneficial = (failure_delta < 0) | (
            (failure_delta == 0) & (score_delta < 0.0)
        )
        dataset = {
            "support_recovery_features": torch.cat(features_all, dim=0),
            "candidate_target_rad": torch.cat(candidate_target_all, dim=0),
            "valid": valid,
            "baseline": baseline,
            "candidate": candidate_branch,
            "paired_score_delta": score_delta,
            "paired_failure_delta": failure_delta,
            "candidate_beneficial": beneficial,
            "motion_id": torch.cat(motion_all, dim=0),
            "source_time_s": torch.cat(source_time_all, dim=0),
            "support_action_ids": SUPPORT_ACTION_IDS,
            "support_joint_names": SUPPORT_JOINT_NAMES,
            "teacher_gain": args.teacher_gain,
            "teacher_limit_rad": args.teacher_limit_rad,
            # new change
            "candidate_mode": candidate_mode,
            "explicit_candidate_vector_rad": (
                None
                if explicit_candidate is None
                else explicit_candidate.detach().cpu()
            ),
            "pulse_steps": args.pulse_steps,
            "outcome_horizon_steps": horizon_steps,
            "outcome_horizon_s": float(actual_horizon_s),
            "failure_penalty": args.failure_penalty,
            "snapshot_batches": args.snapshot_batches,
            "pair_count": pair_count,
            "physics_env_count": total_envs,
            "seed": args.seed,
            "strict_pairing": True,
            "same_simulator_snapshot": True,
            "simultaneous_physics_twins": True,
            "snapshot_replay_audits": audit_all,
            "ppo_updates": False,
            "source_checkpoint_role": "frozen model_1375 JIT only",
            "feature_order": (
                "support_v2[79],com_distance,com_outward_velocity,torso_tilt,"
                "tilt_growth,root_xy_speed,max_foot_xy_speed,min_support_load,"
                "support_load_imbalance"
            ),
        }
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, output)
        valid_score_delta = score_delta[valid]
        print(
            {
                "output": str(output),
                "samples_total": int(valid.numel()),
                "samples_valid": int(valid.sum()),
                "baseline_failures": int((baseline["failure"] & valid).sum()),
                "candidate_failures": int(
                    (candidate_branch["failure"] & valid).sum()
                ),
                "candidate_beneficial_fraction": float(
                    beneficial[valid].float().mean()
                ),
                "paired_score_delta_mean": float(valid_score_delta.mean()),
                "paired_score_delta_median": float(valid_score_delta.median()),
                "horizon_s": float(actual_horizon_s),
            },
            flush=True,
        )
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
