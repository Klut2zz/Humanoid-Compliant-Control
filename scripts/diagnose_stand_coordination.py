"""Phase-resolved diagnostics for the 100% StandCoordinationV1 controller."""

# new change
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys


# new change
_PROJECT_ROOT = pathlib.Path(__file__).absolute().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "softmimic_gym"))

from isaaclab.app import AppLauncher


# new change
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-G1-StandCoordinationV1-100Pct-29DOF-v0",
)
parser.add_argument("--jit_policy", required=True)
# new change
parser.add_argument(
    "--jit_extra_obs_groups",
    nargs="*",
    default=(),
    help="Append named structured observation groups after the policy tensor.",
)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--phase_bins", type=int, default=48)
# new change
parser.add_argument(
    "--support_contact_threshold_n",
    type=float,
    default=20.0,
    help=(
        "Vertical-force threshold for the empirically resolved left/right G1 "
        "support channels. The official asset reports sole load on knee-link "
        "sensor rows, so ankle rows are retained only as a legacy audit."
    ),
)
# new change
parser.add_argument(
    "--slip_speed_threshold_mps",
    type=float,
    default=0.02,
    help="Contact-gated foot XY-speed threshold used to count slip violations.",
)
# new change
parser.add_argument(
    "--root_speed_threshold_mps",
    type=float,
    default=0.05,
    help="Root XY-speed threshold used to count stationary-support violations.",
)
# new change
parser.add_argument(
    "--support_feedback_gain",
    type=float,
    default=0.0,
    help=(
        "Inference-only outer-loop correction gain for bilateral hip/ankle "
        "pitch/roll targets: delta_target=-gain*(q-q_ref)."
    ),
)
# new change
parser.add_argument(
    "--support_feedback_limit_rad",
    type=float,
    default=0.03,
    help="Per-axis physical target limit for the support-chain probe.",
)
# new change
parser.add_argument(
    "--support_feedback_excluded_motion_ids",
    nargs="*",
    type=int,
    default=(),
    help=(
        "Optional zero-based StandCoordination motion ids on which the analytic "
        "support feedback is disabled. This isolates motion-specific safety "
        "failures without changing the loaded policy or other motion classes."
    ),
)
# new change
parser.add_argument(
    "--support_feedback_enabled_joint_names",
    nargs="*",
    default=(),
    help=(
        "Optional subset of the canonical ten support joints on which analytic "
        "feedback is enabled. Empty enables all ten. This is an inference-only "
        "credit-isolation probe."
    ),
)
# new change
parser.add_argument(
    "--support_feedback_require_double_support",
    action="store_true",
    help="Enable support residual only while both resolved support channels are in contact.",
)
# new change
parser.add_argument(
    "--support_feedback_max_com_distance_m",
    type=float,
    default=0.0,
    help="If positive, disable support residual when COM-to-foot-midpoint XY distance exceeds this value.",
)
# new change
parser.add_argument(
    "--support_feedback_max_tilt_rad",
    type=float,
    default=0.0,
    help="If positive, disable support residual when root roll/pitch norm exceeds this value.",
)
# new change
parser.add_argument(
    "--support_feedback_min_episode_age",
    type=int,
    default=0,
    help="Disable support residual for this many control steps after every reset.",
)
# new change
parser.add_argument(
    "--support_feedback_smooth_gate",
    action="store_true",
    help=(
        "Use continuous contact/COM/tilt/age margins instead of a hard on/off "
        "gate, avoiding target discontinuities near safety thresholds."
    ),
)
# new change
parser.add_argument(
    "--support_policy_baseline_jit",
    default=None,
    help=(
        "Optional frozen baseline JIT used to isolate the learned support-head "
        "residual of a structured candidate policy."
    ),
)
# new change
parser.add_argument(
    "--support_policy_baseline_extra_obs_groups",
    nargs="*",
    default=(),
    help=(
        "Append structured observation groups for a baseline JIT that was "
        "exported from the new adapter-capable model. Leave empty for legacy "
        "705D baseline JIT files."
    ),
)
# new change
parser.add_argument(
    "--support_policy_residual_scale",
    type=float,
    default=1.0,
    help=(
        "Inference-only multiplier for candidate-minus-baseline action on the "
        "complete ten-DOF support rows. Requires --support_policy_baseline_jit."
    ),
)
# new change
parser.add_argument(
    "--support_actor_residual_scale",
    type=float,
    default=1.0,
    help=(
        "Inference-only multiplier on the complete ten-DOF support action "
        "rows after policy evaluation. StandCoordination actions are already "
        "reference-relative, so zero means exact q_ref targets and one is the "
        "unchanged actor."
    ),
)
# new change
parser.add_argument(
    "--global_support_feedback_limit_rad",
    type=float,
    default=0.0,
    help=(
        "Inference-only coordinated support feedback authority. Zero disables "
        "the probe; positive values bound every hip/knee/ankle correction."
    ),
)
# new change
parser.add_argument(
    "--global_support_feedback_velocity_horizon_s",
    type=float,
    default=0.30,
    help="Capture-point look-ahead multiplying root XY velocity.",
)
# new change
parser.add_argument(
    "--global_support_feedback_com_weight",
    type=float,
    default=0.50,
    help="Weight of signed COM-to-support-center offset in the capture error.",
)
# new change
parser.add_argument(
    "--global_support_feedback_error_scale_m",
    type=float,
    default=0.10,
    help="Tanh transition scale for the signed global support error.",
)
# new change
parser.add_argument(
    "--global_support_feedback_min_episode_age",
    type=int,
    default=25,
    help="Start age of a 50-step smooth ramp for coordinated feedback.",
)
# new change
parser.add_argument(
    "--knee_target_bias_rad",
    nargs=2,
    type=float,
    metavar=("LEFT", "RIGHT"),
    default=(0.0, 0.0),
    help=(
        "Inference-only bilateral knee-pitch target bias in radians. This is "
        "used for finite-difference support controllability diagnosis."
    ),
)
# new change
parser.add_argument(
    "--right_wrist_feedback_gain",
    type=float,
    default=0.0,
    help=(
        "Inference-only outer-loop correction gain applied to the three right-wrist "
        "targets: delta_q_target=-gain*(q-q_ref)."
    ),
)
# new change
parser.add_argument(
    "--bilateral_arm_feedback_gain",
    type=float,
    default=0.0,
    help=(
        "Inference-only outer-loop correction gain for all fourteen arm joints: "
        "delta_q_target=-gain*(q-q_ref)."
    ),
)
# new change
parser.add_argument(
    "--bilateral_arm_feedback_limit_rad",
    type=float,
    default=0.08,
    help="Per-axis absolute limit of the bilateral arm target correction.",
)
# new change
parser.add_argument(
    "--bilateral_proximal_feedback_gain",
    type=float,
    default=0.0,
    help=(
        "Inference-only outer-loop correction gain for bilateral shoulder "
        "pitch/roll/yaw and elbow joints. Wrist, waist, legs, actuators, and "
        "the loaded actor remain unchanged."
    ),
)
# new change
parser.add_argument(
    "--bilateral_proximal_feedback_limit_rad",
    type=float,
    default=0.04,
    help="Per-axis absolute limit of the bilateral proximal-arm target correction.",
)
# new change
parser.add_argument(
    "--bilateral_proximal_feedback_motion_ids",
    nargs="*",
    type=int,
    default=(),
    help=(
        "Optional zero-based motion ids on which to enable the proximal probe. "
        "Empty means all motions; StandCoordination bilateral sagittal is id 1."
    ),
)
# new change
parser.add_argument(
    "--right_wrist_feedback_limit_rad",
    type=float,
    default=0.08,
    help="Per-axis absolute limit of the inference-only wrist target correction.",
)
parser.add_argument(
    "--episode_warmup_steps",
    type=int,
    default=50,
    help="Exclude reset transients from accuracy statistics, but not failures.",
)
# new change
parser.add_argument(
    "--waist_target_bias_rad",
    nargs=3,
    type=float,
    metavar=("YAW", "ROLL", "PITCH"),
    default=(0.0, 0.0, 0.0),
    help=(
        "Inference-only fixed target-space waist bias in radians. The order is "
        "yaw roll pitch; the loaded actor, actuator gains, and all other action "
        "rows remain unchanged."
    ),
)
# new change
parser.add_argument(
    "--waist_error_feedback_gain",
    nargs=3,
    type=float,
    metavar=("YAW", "ROLL", "PITCH"),
    default=(0.0, 0.0, 0.0),
    help=(
        "Inference-only waist outer-loop gains. Applies "
        "delta_q_target=-gain*(q-q_ref) independently to yaw, roll, pitch."
    ),
)
# new change
parser.add_argument(
    "--waist_error_feedback_limit_rad",
    nargs=3,
    type=float,
    metavar=("YAW", "ROLL", "PITCH"),
    default=(0.005, 0.01, 0.01),
    help="Per-axis absolute target-space limits for waist error feedback.",
)
parser.add_argument("--output_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# new change
def _safe_divide(numerator, denominator):
    import torch

    return numerator / torch.clamp(denominator, min=torch.finfo(numerator.dtype).eps)


# new change
def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


# new change
def main() -> None:
    import gymnasium as gym
    import torch

    from isaaclab.utils import math as math_utils
    from isaaclab_tasks.utils import parse_env_cfg

    import softmimic_gym.tasks  # noqa: F401
    from softmimic_gym.assets.robots.g1_compat import (
        G1_SOFTMIMIC_29_JOINT_NAMES,
        # new change
        G1_SOFTMIMIC_BILATERAL_ARM_ACTION_IDS,
        G1_SOFTMIMIC_KEYPOINT_BODY_NAMES,
    )

    if args.num_envs % 8 != 0:
        raise ValueError("--num_envs must be divisible by eight for balanced motion coverage.")
    if args.phase_bins <= 0:
        raise ValueError("--phase_bins must be positive.")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.recorders = None
    env = gym.make(args.task, cfg=env_cfg)

    try:
        print("[STAND_DIAG] environment_constructed", flush=True)
        observations, _ = env.reset()
        print("[STAND_DIAG] environment_reset", flush=True)
        unwrapped = env.unwrapped
        device = unwrapped.device
        robot = unwrapped.scene["robot"]
        command = unwrapped.command_manager.get_term("upper_body_joints")
        policy = torch.jit.load(args.jit_policy, map_location=device)
        policy.eval()
# new change
        support_baseline_policy = None
        if args.support_policy_baseline_jit is not None:
            support_baseline_policy = torch.jit.load(
                args.support_policy_baseline_jit, map_location=device
            )
            support_baseline_policy.eval()
        elif args.support_policy_residual_scale != 1.0:
            raise ValueError(
                "--support_policy_residual_scale requires "
                "--support_policy_baseline_jit."
            )
        print("[STAND_DIAG] policy_loaded", flush=True)

        motion_names = tuple(
            pathlib.Path(path).stem for path in command.cfg.demo_recording_path
        )
        num_motions = len(motion_names)
        if num_motions != 8:
            raise RuntimeError(f"Expected eight StandCoordination motions, got {num_motions}.")
        # new change
        invalid_probe_motion_ids = sorted(
            set(args.bilateral_proximal_feedback_motion_ids) - set(range(num_motions))
        )
        if invalid_probe_motion_ids:
            raise ValueError(
                "Invalid --bilateral_proximal_feedback_motion_ids: "
                f"{invalid_probe_motion_ids}; expected ids in [0, {num_motions - 1}]."
            )
        motion_ids = torch.arange(args.num_envs, device=device) % num_motions
        source_duration_s = float(command.motion_lib.get_max_times(command.motion_ids).max())
        phase_width_s = source_duration_s / args.phase_bins

        control_ids = torch.as_tensor(command.control_joint_ids, device=device, dtype=torch.long)
        waist_ids = torch.tensor([2, 5, 8], device=device)
        # new change
        support_ids = torch.tensor(
            [0, 1, 9, 10, 13, 14, 3, 4, 17, 18],
            device=device,
            dtype=torch.long,
        )
# new change
        knee_ids = torch.tensor([9, 10], device=device, dtype=torch.long)
        support_joint_names = tuple(
            G1_SOFTMIMIC_29_JOINT_NAMES[index]
            for index in support_ids.tolist()
        )
        # new change
        invalid_support_feedback_joint_names = sorted(
            set(args.support_feedback_enabled_joint_names)
            - set(support_joint_names)
        )
        if invalid_support_feedback_joint_names:
            raise ValueError(
                "Invalid --support_feedback_enabled_joint_names: "
                f"{invalid_support_feedback_joint_names}; expected a subset of "
                f"{list(support_joint_names)}."
            )
        support_feedback_joint_mask = torch.tensor(
            [
                not args.support_feedback_enabled_joint_names
                or name in args.support_feedback_enabled_joint_names
                for name in support_joint_names
            ],
            device=device,
            dtype=torch.bool,
        )
        right_wrist_ids = torch.tensor([24, 26, 28], device=device)
        # new change
        arm_ids = torch.tensor(
            G1_SOFTMIMIC_BILATERAL_ARM_ACTION_IDS, device=device, dtype=torch.long
        )
        arm_joint_names = tuple(G1_SOFTMIMIC_29_JOINT_NAMES[index] for index in arm_ids.tolist())
        # new change
        # Strict-29DOF action rows: bilateral shoulder pitch/roll/yaw and
        # elbow only.  Wrist rows are deliberately excluded from this probe.
        proximal_ids = torch.tensor(
            [11, 12, 15, 16, 19, 20, 21, 22], device=device, dtype=torch.long
        )
        proximal_joint_names = tuple(
            G1_SOFTMIMIC_29_JOINT_NAMES[index] for index in proximal_ids.tolist()
        )
        waist_names = ("waist_yaw", "waist_roll", "waist_pitch")
        right_wrist_names = ("right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw")

        # Shoulder, elbow, and wrist reference keypoints, left then right.
        arm_keypoint_reference_ids = torch.tensor([8, 9, 10, 11, 12, 13], device=device)
        arm_keypoint_names = (
            "left_shoulder",
            "left_elbow",
            "left_wrist",
            "right_shoulder",
            "right_elbow",
            "right_wrist",
        )
        if tuple(command.cfg.keypoint_body_ids) != tuple(G1_SOFTMIMIC_KEYPOINT_BODY_NAMES):
            raise RuntimeError("Unexpected keypoint ordering; refusing to produce a mislabeled report.")

        left_foot_ids, left_foot_names = robot.find_bodies(
            ["left_ankle_roll_link"], preserve_order=True
        )
        right_foot_ids, right_foot_names = robot.find_bodies(
            ["right_ankle_roll_link"], preserve_order=True
        )
        if left_foot_names != ["left_ankle_roll_link"] or right_foot_names != [
            "right_ankle_roll_link"
        ]:
            raise RuntimeError("Could not resolve the two G1 support feet.")
        foot_body_ids = torch.tensor(
            [left_foot_ids[0], right_foot_ids[0]], device=device, dtype=torch.long
        )

        contact_sensor = unwrapped.scene.sensors["contact_forces"]
        contact_ids, contact_names = contact_sensor.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )
        if contact_names != ["left_ankle_roll_link", "right_ankle_roll_link"]:
            raise RuntimeError("Could not resolve support-foot contact sensor entries.")
        contact_ids = torch.tensor(contact_ids, device=device, dtype=torch.long)

        # new change
        # The official IsaacLab G1 USD reports the dominant left/right sole
        # loads on the corresponding knee-link rows of ContactSensor data.
        # Keep the ankle rows above as a legacy audit, but resolve support from
        # these empirically verified channels so contact gating is meaningful.
        support_contact_ids_list, support_contact_names = contact_sensor.find_bodies(
            ["left_knee_link", "right_knee_link"], preserve_order=True
        )
        if support_contact_names != ["left_knee_link", "right_knee_link"]:
            raise RuntimeError("Could not resolve the two G1 support-force channels.")
        support_contact_ids = torch.tensor(
            support_contact_ids_list, device=device, dtype=torch.long
        )

        flat_bins = num_motions * args.phase_bins
        dtype = torch.float64
        sample_count = torch.zeros(flat_bins, device=device, dtype=dtype)
        keypoint_sum = torch.zeros((flat_bins, 6), device=device, dtype=dtype)
        keypoint_sq_sum = torch.zeros_like(keypoint_sum)
        waist_q_abs_sum = torch.zeros((flat_bins, 3), device=device, dtype=dtype)
        # new change
        waist_q_signed_sum = torch.zeros_like(waist_q_abs_sum)
        waist_dq_abs_sum = torch.zeros_like(waist_q_abs_sum)
        # new change
        support_q_abs_sum = torch.zeros(
            (flat_bins, len(support_joint_names)), device=device, dtype=dtype
        )
        support_q_signed_sum = torch.zeros_like(support_q_abs_sum)
        support_target_abs_sum = torch.zeros_like(support_q_abs_sum)
        # new change
        # Measure the learned V2 residual in physical target radians before
        # applying any inference-only multiplier.  This distinguishes a wrong
        # direction from a correct but ineffective residual magnitude.
        support_policy_residual_abs_sum = torch.zeros(
            len(support_joint_names), device=device, dtype=dtype
        )
        support_policy_residual_max = torch.zeros_like(
            support_policy_residual_abs_sum
        )
        support_policy_residual_sample_count = torch.zeros(
            (), device=device, dtype=dtype
        )
        # new change
        support_feedback_gate_active_count = torch.zeros(
            (), device=device, dtype=dtype
        )
        support_feedback_gate_sample_count = torch.zeros(
            (), device=device, dtype=dtype
        )
        # new change
        global_support_feedback_abs_sum = torch.zeros(
            len(support_joint_names), device=device, dtype=dtype
        )
        global_support_feedback_max = torch.zeros_like(
            global_support_feedback_abs_sum
        )
        global_support_feedback_saturated_count = torch.zeros_like(
            global_support_feedback_abs_sum
        )
        global_support_feedback_sample_count = torch.zeros(
            (), device=device, dtype=dtype
        )
        # new change
        arm_q_abs_sum = torch.zeros((flat_bins, len(arm_joint_names)), device=device, dtype=dtype)
        arm_q_signed_sum = torch.zeros_like(arm_q_abs_sum)
        arm_target_abs_sum = torch.zeros_like(arm_q_abs_sum)
        # new change
        proximal_feedback_abs_sum = torch.zeros(
            len(proximal_joint_names), device=device, dtype=dtype
        )
        proximal_feedback_max = torch.zeros_like(proximal_feedback_abs_sum)
        proximal_feedback_saturated_count = torch.zeros_like(proximal_feedback_abs_sum)
        proximal_feedback_sample_count = torch.zeros((), device=device, dtype=dtype)
        com_support_abs_sum = torch.zeros((flat_bins, 3), device=device, dtype=dtype)
        foot_slip_sum = torch.zeros((flat_bins, 2), device=device, dtype=dtype)
        foot_slip_sq_sum = torch.zeros_like(foot_slip_sum)
        foot_contact_count = torch.zeros_like(foot_slip_sum)
        # new change
        resolved_contact_count = torch.zeros_like(foot_slip_sum)
        resolved_force_sum = torch.zeros_like(foot_slip_sum)
        contact_slip_speed_sum = torch.zeros_like(foot_slip_sum)
        contact_slip_speed_sq_sum = torch.zeros_like(foot_slip_sum)
        contact_slip_sample_count = torch.zeros_like(foot_slip_sum)
        contact_slip_violation_count = torch.zeros_like(foot_slip_sum)
        contact_anchor_displacement_sum = torch.zeros_like(foot_slip_sum)
        contact_anchor_sample_count = torch.zeros_like(foot_slip_sum)
        double_support_count = torch.zeros(flat_bins, device=device, dtype=dtype)
        no_support_count = torch.zeros_like(double_support_count)
        root_xy_abs_velocity_sum = torch.zeros((flat_bins, 2), device=device, dtype=dtype)
        root_xy_speed_sum = torch.zeros(flat_bins, device=device, dtype=dtype)
        root_xy_speed_sq_sum = torch.zeros_like(root_xy_speed_sum)
        root_xy_speed_violation_count = torch.zeros_like(root_xy_speed_sum)
        root_xy_drift_sum = torch.zeros_like(root_xy_speed_sum)
        root_xy_drift_sq_sum = torch.zeros_like(root_xy_speed_sum)

        # Maximums are retained per motion rather than averaged across phase
        # bins, making short high-speed skating events visible.
        motion_root_speed_max = torch.zeros(num_motions, device=device, dtype=dtype)
        motion_root_drift_max = torch.zeros_like(motion_root_speed_max)
        motion_contact_slip_speed_max = torch.zeros(
            (num_motions, 2), device=device, dtype=dtype
        )
        motion_contact_anchor_displacement_max = torch.zeros_like(
            motion_contact_slip_speed_max
        )

        # Sufficient statistics for e = bias - lag * q_ref_dot.
        wrist_count = torch.zeros((num_motions, 3), device=device, dtype=dtype)
        wrist_error_sum = torch.zeros_like(wrist_count)
        wrist_abs_error_sum = torch.zeros_like(wrist_count)
        wrist_error_sq_sum = torch.zeros_like(wrist_count)
        wrist_ref_vel_sum = torch.zeros_like(wrist_count)
        wrist_ref_vel_sq_sum = torch.zeros_like(wrist_count)
        wrist_error_ref_vel_sum = torch.zeros_like(wrist_count)

        first_bad_step = torch.full(
            (args.num_envs,), args.steps + 1, device=device, dtype=torch.long
        )
        first_bad_relative_time = torch.full((args.num_envs,), float("nan"), device=device)
        first_bad_source_time = torch.full((args.num_envs,), float("nan"), device=device)
        # new change
        first_failure_step = torch.full(
            (args.num_envs,), args.steps + 1, device=device, dtype=torch.long
        )
        first_failure_relative_time = torch.full(
            (args.num_envs,), float("nan"), device=device
        )
        first_failure_source_time = torch.full_like(
            first_failure_relative_time, float("nan")
        )
        first_failure_bad_orientation = torch.zeros(
            args.num_envs, device=device, dtype=torch.bool
        )
        first_failure_base_contact = torch.zeros_like(first_failure_bad_orientation)
        episode_age = torch.zeros(args.num_envs, device=device, dtype=torch.long)

        # new change
        initial_foot_xy = robot.data.body_com_pos_w.index_select(1, foot_body_ids)[
            :, :, :2
        ].to(dtype)
        contact_anchor_xy = initial_foot_xy.clone()
        previous_resolved_contact = torch.zeros(
            (args.num_envs, 2), device=device, dtype=torch.bool
        )

        masses = robot.data.default_mass.to(device=device, dtype=dtype)
        total_mass = masses.sum(dim=1, keepdim=True)
        print(
            f"[STAND_DIAG] collectors_ready motions={num_motions} "
            f"source_duration_s={source_duration_s:.3f}",
            flush=True,
        )
        print(f"[STAND_DIAG] robot_body_names={robot.body_names}", flush=True)
        print(
            f"[STAND_DIAG] contact_sensor_body_names={contact_sensor.body_names}",
            flush=True,
        )

        for step_index in range(args.steps):
            phase_before_step = command.relative_timestamp.clone()
            source_time_before_step = command.motion_count.clone() * command.motion_dt
            with torch.inference_mode():
                # new change
                policy_input = observations["policy"]
                if args.jit_extra_obs_groups:
                    policy_input = torch.cat(
                        [policy_input]
                        + [observations[name] for name in args.jit_extra_obs_groups],
                        dim=-1,
                    )
                action = policy(policy_input).detach()
                # new change
                # The environment composes q_target=q_ref+0.30*action on all
                # support rows. Scaling these raw rows therefore isolates the
                # frozen actor's support residual without changing the motion
                # reference, actuators, arms, waist, or environment state.
                action[:, support_ids] *= args.support_actor_residual_scale
                # new change
                # Isolate and scale only the learned structured support residual.
                # A scale of 1.0 exactly preserves the candidate action.  The
                # original actor and every arm/waist/non-support row always come
                # from the candidate unchanged.
                if support_baseline_policy is not None:
                    # new change
                    baseline_policy_input = observations["policy"]
                    if args.support_policy_baseline_extra_obs_groups:
                        baseline_policy_input = torch.cat(
                            [baseline_policy_input]
                            + [
                                observations[name]
                                for name in args.support_policy_baseline_extra_obs_groups
                            ],
                            dim=-1,
                        )
                    baseline_action = support_baseline_policy(
                        baseline_policy_input
                    ).detach()
                    support_delta = action.index_select(1, support_ids) - baseline_action.index_select(
                        1, support_ids
                    )
                    # new change
                    physical_support_delta = 0.30 * support_delta
                    support_policy_residual_abs_sum += (
                        physical_support_delta.abs().sum(dim=0).to(dtype)
                    )
                    support_policy_residual_max = torch.maximum(
                        support_policy_residual_max,
                        physical_support_delta.abs().amax(dim=0).to(dtype),
                    )
                    support_policy_residual_sample_count += args.num_envs
                    action[:, support_ids] = baseline_action.index_select(
                        1, support_ids
                    ) + args.support_policy_residual_scale * support_delta
                # new change
                # Both knee rows use the strict lower-body 0.30-rad action
                # scale.  Zero bias is exactly neutral.  Keeping left and
                # right independent exposes symmetric and antisymmetric
                # controllability without changing the loaded policy.
                knee_target_bias = torch.as_tensor(
                    args.knee_target_bias_rad,
                    device=device,
                    dtype=action.dtype,
                )
                action[:, knee_ids] += knee_target_bias / 0.30
                # new change
                proximal_target_correction = torch.zeros(
                    (args.num_envs, len(proximal_joint_names)),
                    device=device,
                    dtype=action.dtype,
                )
                proximal_feedback_active_mask = torch.ones(
                    args.num_envs, device=device, dtype=torch.bool
                )
                if args.bilateral_proximal_feedback_motion_ids:
                    proximal_feedback_active_mask = torch.zeros_like(
                        proximal_feedback_active_mask
                    )
                    for probe_motion_id in args.bilateral_proximal_feedback_motion_ids:
                        proximal_feedback_active_mask |= motion_ids == probe_motion_id
                if args.bilateral_proximal_feedback_gain != 0.0:
                    current_q = robot.data.joint_pos.index_select(1, control_ids)
                    reference_q_now = command.dof_pos[:, 1]
                    proximal_target_correction = torch.clamp(
                        -args.bilateral_proximal_feedback_gain
                        * (
                            current_q.index_select(1, proximal_ids)
                            - reference_q_now.index_select(1, proximal_ids)
                        ),
                        min=-args.bilateral_proximal_feedback_limit_rad,
                        max=args.bilateral_proximal_feedback_limit_rad,
                    )
                    proximal_target_correction = torch.where(
                        proximal_feedback_active_mask.unsqueeze(1),
                        proximal_target_correction,
                        torch.zeros_like(proximal_target_correction),
                    )
                    action[:, proximal_ids] += proximal_target_correction / 0.15
                # new change
                if args.bilateral_arm_feedback_gain != 0.0:
                    current_q = robot.data.joint_pos.index_select(1, control_ids)
                    reference_q_now = command.dof_pos[:, 1]
                    arm_target_correction = torch.clamp(
                        -args.bilateral_arm_feedback_gain
                        * (
                            current_q.index_select(1, arm_ids)
                            - reference_q_now.index_select(1, arm_ids)
                        ),
                        min=-args.bilateral_arm_feedback_limit_rad,
                        max=args.bilateral_arm_feedback_limit_rad,
                    )
                    action[:, arm_ids] += arm_target_correction / 0.15
                # new change
                # Diagnostic only: close a small outer position-error loop in
                # target space.  StandCoordination uses a 0.15-rad arm action
                # scale, so convert the bounded target correction back to raw
                # actor units before env.step().  Zero gain is neutral.
                if args.right_wrist_feedback_gain != 0.0:
                    current_q = robot.data.joint_pos.index_select(1, control_ids)
                    reference_q_now = command.dof_pos[:, 1]
                    wrist_target_correction = torch.clamp(
                        -args.right_wrist_feedback_gain
                        * (
                            current_q.index_select(1, right_wrist_ids)
                            - reference_q_now.index_select(1, right_wrist_ids)
                        ),
                        min=-args.right_wrist_feedback_limit_rad,
                        max=args.right_wrist_feedback_limit_rad,
                    )
                    action[:, right_wrist_ids] += wrist_target_correction / 0.15
                # new change
                # Finite-difference controllability probe. StandCoordination's
                # waist action scale is 0.20 rad, so convert the requested
                # physical target bias back to actor units. A zero vector is
                # exactly neutral and leaves the loaded policy unchanged.
                waist_target_bias = torch.as_tensor(
                    args.waist_target_bias_rad,
                    device=device,
                    dtype=action.dtype,
                )
                action[:, waist_ids] += waist_target_bias / 0.20
                # new change
                # State-conditioned alternative to a fixed bias. This probe
                # closes only a bounded outer position-error loop and leaves
                # the actor, actuator parameters, and reference unchanged.
                waist_feedback_gain = torch.as_tensor(
                    args.waist_error_feedback_gain,
                    device=device,
                    dtype=action.dtype,
                )
                waist_feedback_limit = torch.as_tensor(
                    args.waist_error_feedback_limit_rad,
                    device=device,
                    dtype=action.dtype,
                )
                if torch.any(waist_feedback_gain != 0.0):
                    current_q = robot.data.joint_pos.index_select(1, control_ids)
                    reference_q_now = command.dof_pos[:, 1]
                    waist_error = (
                        current_q.index_select(1, waist_ids)
                        - reference_q_now.index_select(1, waist_ids)
                    )
                    waist_feedback = torch.clamp(
                        -waist_feedback_gain * waist_error,
                        min=-waist_feedback_limit,
                        max=waist_feedback_limit,
                    )
                    action[:, waist_ids] += waist_feedback / 0.20
                # new change
                # Read-only support controllability probe.  All eight rows
                # use the strict-29DOF 0.30-rad action scale.  The zero-gain
                # default is exactly equivalent to the accepted checkpoint.
                if args.support_feedback_gain != 0.0:
                    current_q = robot.data.joint_pos.index_select(1, control_ids)
                    reference_q_now = command.dof_pos[:, 1]
                    support_feedback = torch.clamp(
                        -args.support_feedback_gain
                        * (
                            current_q.index_select(1, support_ids)
                            - reference_q_now.index_select(1, support_ids)
                        ),
                        min=-args.support_feedback_limit_rad,
                        max=args.support_feedback_limit_rad,
                    )
                    # new change
                    # Isolate which support-chain outputs are controllable
                    # without altering the common gain, limit, or base actor.
                    support_feedback = support_feedback * (
                        support_feedback_joint_mask[None, :].to(
                            support_feedback.dtype
                        )
                    )
                    # new change
                    # The outer loop must yield to the frozen actor whenever
                    # support is marginal. This prevents q-error correction
                    # from fighting recovery actions near contact loss/falls.
                    support_feedback_gate = torch.ones(
                        args.num_envs,
                        device=device,
                        dtype=support_feedback.dtype
                        if args.support_feedback_smooth_gate
                        else torch.bool,
                    )
                    if args.support_feedback_require_double_support:
                        pre_action_support_force = (
                            contact_sensor.data.net_forces_w_history.index_select(
                                2, support_contact_ids
                            )[..., 2]
                            .abs()
                            .max(dim=1)
                            .values
                        )
                        if args.support_feedback_smooth_gate:
                            minimum_support_force = pre_action_support_force.min(dim=1).values
                            force_full_scale_n = max(
                                100.0,
                                args.support_contact_threshold_n + 1.0,
                            )
                            support_feedback_gate *= torch.clamp(
                                (
                                    minimum_support_force
                                    - args.support_contact_threshold_n
                                )
                                / (
                                    force_full_scale_n
                                    - args.support_contact_threshold_n
                                ),
                                min=0.0,
                                max=1.0,
                            ).to(support_feedback_gate.dtype)
                        else:
                            support_feedback_gate &= (
                                pre_action_support_force
                                >= args.support_contact_threshold_n
                            ).all(dim=1)
                    if args.support_feedback_max_com_distance_m > 0.0:
                        pre_action_body_com = robot.data.body_com_pos_w.to(dtype)
                        pre_action_whole_body_com = (
                            pre_action_body_com * masses.unsqueeze(-1)
                        ).sum(dim=1) / total_mass
                        pre_action_foot_xy = robot.data.body_com_pos_w.index_select(
                            1, foot_body_ids
                        )[:, :, :2].to(dtype)
                        pre_action_support_center = pre_action_foot_xy.mean(dim=1)
                        pre_action_com_distance = torch.linalg.vector_norm(
                            pre_action_whole_body_com[:, :2]
                            - pre_action_support_center,
                            dim=1,
                        )
                        if args.support_feedback_smooth_gate:
                            support_feedback_gate *= torch.clamp(
                                1.0
                                - pre_action_com_distance
                                / args.support_feedback_max_com_distance_m,
                                min=0.0,
                                max=1.0,
                            ).to(support_feedback_gate.dtype)
                        else:
                            support_feedback_gate &= (
                                pre_action_com_distance
                                <= args.support_feedback_max_com_distance_m
                            )
                    if args.support_feedback_max_tilt_rad > 0.0:
                        root_roll, root_pitch, _ = math_utils.euler_xyz_from_quat(
                            robot.data.root_quat_w
                        )
                        root_tilt = torch.linalg.vector_norm(
                            torch.stack((root_roll, root_pitch), dim=1),
                            dim=1,
                        )
                        if args.support_feedback_smooth_gate:
                            support_feedback_gate *= torch.clamp(
                                1.0
                                - root_tilt
                                / args.support_feedback_max_tilt_rad,
                                min=0.0,
                                max=1.0,
                            ).to(support_feedback_gate.dtype)
                        else:
                            support_feedback_gate &= (
                                root_tilt <= args.support_feedback_max_tilt_rad
                            )
                    if args.support_feedback_smooth_gate:
                        support_feedback_gate *= torch.clamp(
                            (
                                episode_age
                                - args.support_feedback_min_episode_age
                            ).to(support_feedback_gate.dtype)
                            / 50.0,
                            min=0.0,
                            max=1.0,
                        )
                    else:
                        support_feedback_gate &= (
                            episode_age >= args.support_feedback_min_episode_age
                        )
                    support_feedback_gate_active_count += (
                        support_feedback_gate.sum().to(dtype)
                    )
                    support_feedback_gate_sample_count += args.num_envs
                    support_feedback = support_feedback * (
                        support_feedback_gate[:, None].to(support_feedback.dtype)
                    )
                    # new change
                    # A failed motion class must remain exactly on the frozen
                    # model_1375 action. Mask after clipping so disabled rows
                    # are exactly zero and cannot receive a residual target.
                    if args.support_feedback_excluded_motion_ids:
                        support_feedback_enabled = torch.ones(
                            args.num_envs,
                            device=device,
                            dtype=torch.bool,
                        )
                        for excluded_motion_id in (
                            args.support_feedback_excluded_motion_ids
                        ):
                            support_feedback_enabled &= motion_ids != excluded_motion_id
                        support_feedback = support_feedback * (
                            support_feedback_enabled[:, None].to(support_feedback.dtype)
                        )
                    action[:, support_ids] += support_feedback / 0.30
                # new change
                # Global-drift probe derived from the corrected strict paired
                # response matrix.  This is deliberately not q-error
                # feedback: model_1375 lacks translational state, while V4/V5
                # showed that q-ref correction alone cannot contain root
                # skating.  Positive sagittal error uses hip/ankle positive
                # and knee negative; positive lateral error uses hip-roll
                # negative and ankle-roll positive.  These opposing pairs
                # preserve a support-chain shape instead of translating every
                # lower-body target in the same direction.
                if args.global_support_feedback_limit_rad > 0.0:
                    if args.global_support_feedback_error_scale_m <= 0.0:
                        raise ValueError(
                            "--global_support_feedback_error_scale_m must be positive."
                        )
                    reference_root_xy = command.root_state["root_pos"][:, 1, :2]
                    root_error_w = torch.cat(
                        (
                            robot.data.root_pos_w[:, :2] - reference_root_xy,
                            torch.zeros_like(reference_root_xy[:, :1]),
                        ),
                        dim=1,
                    )
                    _, _, root_yaw = math_utils.euler_xyz_from_quat(
                        robot.data.root_quat_w
                    )
                    yaw_quat = math_utils.quat_from_euler_xyz(
                        torch.zeros_like(root_yaw),
                        torch.zeros_like(root_yaw),
                        root_yaw,
                    )
                    root_error_b = math_utils.quat_rotate_inverse(
                        yaw_quat, root_error_w
                    )[:, :2]
                    root_velocity_b = robot.data.root_lin_vel_b[:, :2]
                    pre_action_body_com = robot.data.body_com_pos_w
                    pre_action_whole_body_com = (
                        pre_action_body_com * masses.to(action.dtype).unsqueeze(-1)
                    ).sum(dim=1) / total_mass.to(action.dtype)
                    pre_action_support_xy = robot.data.body_com_pos_w.index_select(
                        1, foot_body_ids
                    )[:, :, :2].mean(dim=1)
                    com_offset_w = torch.cat(
                        (
                            pre_action_whole_body_com[:, :2] - pre_action_support_xy,
                            torch.zeros_like(pre_action_support_xy[:, :1]),
                        ),
                        dim=1,
                    )
                    com_offset_b = math_utils.quat_rotate_inverse(
                        yaw_quat, com_offset_w
                    )[:, :2]
                    capture_error_b = (
                        root_error_b
                        + args.global_support_feedback_velocity_horizon_s
                        * root_velocity_b
                        + args.global_support_feedback_com_weight * com_offset_b
                    )
                    bounded_axis = args.global_support_feedback_limit_rad * torch.tanh(
                        capture_error_b
                        / args.global_support_feedback_error_scale_m
                    )
                    sagittal = bounded_axis[:, 0]
                    lateral = bounded_axis[:, 1]
                    global_support_feedback = torch.stack(
                        (
                            sagittal,
                            sagittal,
                            -sagittal,
                            -sagittal,
                            sagittal,
                            sagittal,
                            -lateral,
                            -lateral,
                            lateral,
                            lateral,
                        ),
                        dim=1,
                    )
                    smooth_age_gate = torch.clamp(
                        (
                            episode_age
                            - args.global_support_feedback_min_episode_age
                        ).to(action.dtype)
                        / 50.0,
                        min=0.0,
                        max=1.0,
                    )
                    global_support_feedback *= smooth_age_gate[:, None]
                    global_support_feedback_abs_sum += (
                        global_support_feedback.abs().sum(dim=0).to(dtype)
                    )
                    global_support_feedback_max = torch.maximum(
                        global_support_feedback_max,
                        global_support_feedback.abs().amax(dim=0).to(dtype),
                    )
                    global_support_feedback_saturated_count += (
                        global_support_feedback.abs()
                        >= 0.98 * args.global_support_feedback_limit_rad
                    ).sum(dim=0).to(dtype)
                    global_support_feedback_sample_count += args.num_envs
                    action[:, support_ids] += global_support_feedback / 0.30
                # new change
                # All fourteen arm rows use the task's 0.15-rad residual
                # scale.  Retain the exact target sent this step so q-target
                # dynamics can be separated from policy/reference bias.
                arm_target_before_step = (
                    command.dof_pos[:, 1].index_select(1, arm_ids)
                    + 0.15 * action.index_select(1, arm_ids)
                )
                # new change
                # Hip/ankle action scale is 0.30 rad per actor unit. Retain
                # the physical target to distinguish reference tracking from
                # actuator execution in V3 support-chain checkpoints.
                support_target_before_step = (
                    command.dof_pos[:, 1].index_select(1, support_ids)
                    + 0.30 * action.index_select(1, support_ids)
                )
            observations, _, terminated, truncated, _ = env.step(action)

            bad_orientation = unwrapped.termination_manager.get_term(
                "bad_orientation"
            ).bool()
            # new change
            base_contact = unwrapped.termination_manager.get_term("base_contact").bool()
            first_bad_mask = bad_orientation & (first_bad_step > args.steps)
            first_bad_step[first_bad_mask] = step_index + 1
            first_bad_relative_time[first_bad_mask] = phase_before_step[first_bad_mask]
            first_bad_source_time[first_bad_mask] = source_time_before_step[first_bad_mask]

            # new change
            # Record the first non-timeout failure against the pre-step phase;
            # ManagerBasedRLEnv may already have reset the physical state by
            # the time observations are returned.
            first_failure_mask = terminated.bool() & (first_failure_step > args.steps)
            first_failure_step[first_failure_mask] = step_index + 1
            first_failure_relative_time[first_failure_mask] = phase_before_step[
                first_failure_mask
            ]
            first_failure_source_time[first_failure_mask] = source_time_before_step[
                first_failure_mask
            ]
            first_failure_bad_orientation[first_failure_mask] = bad_orientation[
                first_failure_mask
            ]
            first_failure_base_contact[first_failure_mask] = base_contact[
                first_failure_mask
            ]

            done = terminated | truncated
            episode_age += 1
            episode_age[done] = 0
            eligible = episode_age >= args.episode_warmup_steps
            if not torch.any(eligible):
                continue

            source_time = command.motion_count * command.motion_dt
            phase_bin = torch.clamp(
                torch.floor(source_time / phase_width_s).long(),
                min=0,
                max=args.phase_bins - 1,
            )
            flat_index = motion_ids * args.phase_bins + phase_bin
            selected = torch.nonzero(eligible, as_tuple=False).squeeze(-1)
            selected_index = flat_index[selected]
            ones = torch.ones(selected.shape[0], device=device, dtype=dtype)
            # new change
            # Track how much target authority the probe consumes.  A lower
            # correction with similar accuracy is preferred because later
            # compliant training needs residual/control margin.
            selected_active = selected[proximal_feedback_active_mask[selected]]
            if selected_active.numel() > 0:
                selected_proximal_feedback = (
                    proximal_target_correction[selected_active].abs().to(dtype)
                )
                proximal_feedback_abs_sum += selected_proximal_feedback.sum(dim=0)
                proximal_feedback_max = torch.maximum(
                    proximal_feedback_max, selected_proximal_feedback.max(dim=0).values
                )
                proximal_feedback_saturated_count += (
                    selected_proximal_feedback
                    >= args.bilateral_proximal_feedback_limit_rad - 1.0e-6
                ).to(dtype).sum(dim=0)
                proximal_feedback_sample_count += selected_active.shape[0]
            sample_count.index_add_(0, selected_index, ones)

            actual_q = robot.data.joint_pos.index_select(1, control_ids)
            actual_dq = robot.data.joint_vel.index_select(1, control_ids)
            reference_q = command.dof_pos[:, 1]
            reference_dq = command.dof_vel[:, 1]

            # new change
            waist_q_signed_error = (
                actual_q.index_select(1, waist_ids) - reference_q.index_select(1, waist_ids)
            ).to(dtype)
            waist_q_error = waist_q_signed_error.abs()
            waist_dq_error = torch.abs(
                actual_dq.index_select(1, waist_ids) - reference_dq.index_select(1, waist_ids)
            ).to(dtype)
            waist_q_abs_sum.index_add_(0, selected_index, waist_q_error[selected])
            # new change
            waist_q_signed_sum.index_add_(
                0, selected_index, waist_q_signed_error[selected]
            )
            waist_dq_abs_sum.index_add_(0, selected_index, waist_dq_error[selected])
            # new change
            support_actual_q = actual_q.index_select(1, support_ids)
            support_reference_q = reference_q.index_select(1, support_ids)
            support_q_signed_error = (
                support_actual_q - support_reference_q
            ).to(dtype)
            support_q_abs_sum.index_add_(
                0, selected_index, support_q_signed_error[selected].abs()
            )
            support_q_signed_sum.index_add_(
                0, selected_index, support_q_signed_error[selected]
            )
            support_target_abs_sum.index_add_(
                0,
                selected_index,
                (support_actual_q - support_target_before_step)
                .to(dtype)[selected]
                .abs(),
            )
            # new change
            arm_actual_q = actual_q.index_select(1, arm_ids)
            arm_reference_q = reference_q.index_select(1, arm_ids)
            arm_q_signed_error = (arm_actual_q - arm_reference_q).to(dtype)
            arm_q_abs_sum.index_add_(0, selected_index, arm_q_signed_error[selected].abs())
            arm_q_signed_sum.index_add_(0, selected_index, arm_q_signed_error[selected])
            arm_target_abs_sum.index_add_(
                0,
                selected_index,
                (arm_actual_q - arm_target_before_step).to(dtype)[selected].abs(),
            )

            keypoint_ids = command.keypoint_body_ids_tensor
            desired_keypoints = command.keypoints[:, 1]
            current_keypoints = robot.data.body_pos_w[:, keypoint_ids, :3]
            current_root = robot.data.root_pos_w[:, :3].clone()
            current_root[:, 2] = 0.0
            desired_root = command.root_state["root_pos"][:, 1, :3].clone()
            desired_root[:, 2] = 0.0
            _, _, current_yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
            desired_quat = command.root_state["root_rot"][:, 1, [3, 0, 1, 2]]
            _, _, desired_yaw = math_utils.euler_xyz_from_quat(desired_quat)
            yaw_delta = math_utils.wrap_to_pi(current_yaw - desired_yaw)
            yaw_rotation = math_utils.quat_from_euler_xyz(
                torch.zeros_like(yaw_delta), torch.zeros_like(yaw_delta), yaw_delta
            )
            desired_local = math_utils.quat_rotate(
                yaw_rotation.unsqueeze(1).expand(-1, desired_keypoints.shape[1], -1),
                desired_keypoints - desired_root.unsqueeze(1),
            )
            current_local = current_keypoints - current_root.unsqueeze(1)
            all_keypoint_distance = torch.linalg.vector_norm(
                current_local - desired_local, dim=-1
            )
            arm_distance = all_keypoint_distance.index_select(
                1, arm_keypoint_reference_ids
            ).to(dtype)
            keypoint_sum.index_add_(0, selected_index, arm_distance[selected])
            keypoint_sq_sum.index_add_(0, selected_index, arm_distance[selected].square())

            body_com = robot.data.body_com_pos_w.to(device=device, dtype=dtype)
            whole_body_com = (body_com * masses.unsqueeze(-1)).sum(dim=1) / total_mass
            foot_pos = robot.data.body_com_pos_w.index_select(1, foot_body_ids).to(dtype)
            # new change
            # Match the task's own foot-contact observation: use the peak
            # force norm over the contact-sensor history.  The instantaneous
            # z component is often zero between sensor updates and falsely
            # labels a planted foot as airborne.
            contact_force = (
                contact_sensor.data.net_forces_w_history.index_select(2, contact_ids)
                .norm(dim=-1)
                .max(dim=1)
                .values.to(dtype)
            )
            # new change
            resolved_support_force = (
                contact_sensor.data.net_forces_w_history.index_select(
                    2, support_contact_ids
                )[..., 2]
                .abs()
                .max(dim=1)
                .values.to(dtype)
            )
            resolved_contact_mask = (
                resolved_support_force >= args.support_contact_threshold_n
            )
            midpoint = foot_pos[:, :, :2].mean(dim=1)
            # new change
            # All StandCoordination references command double support.  Use
            # the geometric midpoint because the legacy ankle contact signal
            # is demonstrably mis-attributed on the official G1 USD.
            support_center = midpoint
            com_delta_world = torch.cat(
                (whole_body_com[:, :2] - support_center, torch.zeros_like(whole_body_com[:, :1])),
                dim=1,
            ).to(robot.data.root_quat_w.dtype)
            _, _, robot_yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
            yaw_quat = math_utils.quat_from_euler_xyz(
                torch.zeros_like(robot_yaw), torch.zeros_like(robot_yaw), robot_yaw
            )
            com_delta_local = math_utils.quat_rotate_inverse(yaw_quat, com_delta_world).to(dtype)
            com_metrics = torch.stack(
                (
                    torch.abs(com_delta_local[:, 0]),
                    torch.abs(com_delta_local[:, 1]),
                    torch.linalg.vector_norm(com_delta_local[:, :2], dim=1),
                ),
                dim=1,
            )
            com_support_abs_sum.index_add_(0, selected_index, com_metrics[selected])

            foot_speed = torch.linalg.vector_norm(
                robot.data.body_com_lin_vel_w.index_select(1, foot_body_ids)[:, :, :2], dim=-1
            ).to(dtype)
            contact_mask = contact_force > 1.0
            selected_contact = contact_mask[selected].to(dtype)
            # new change
            # Foot XY speed is accumulated unconditionally: both feet should
            # remain planted in every reference, and contact gating would hide
            # slip because of the legacy sensor/body mismatch above.
            foot_slip_sum.index_add_(0, selected_index, foot_speed[selected])
            foot_slip_sq_sum.index_add_(0, selected_index, foot_speed[selected].square())
            foot_contact_count.index_add_(0, selected_index, selected_contact)

            # new change
            # A contact anchor is reset only when a support channel first
            # becomes loaded.  Its displacement therefore measures actual
            # stance-phase skating rather than ordinary airborne foot motion.
            foot_xy = foot_pos[:, :, :2]
            new_contact = resolved_contact_mask & ~previous_resolved_contact
            reset_anchor = new_contact | done.unsqueeze(1)
            contact_anchor_xy = torch.where(
                reset_anchor.unsqueeze(-1), foot_xy, contact_anchor_xy
            )
            contact_anchor_displacement = torch.linalg.vector_norm(
                foot_xy - contact_anchor_xy, dim=-1
            )
            previous_resolved_contact = resolved_contact_mask.clone()
            previous_resolved_contact[done] = False

            selected_resolved_contact = resolved_contact_mask[selected].to(dtype)
            selected_contact_slip_speed = foot_speed[selected] * selected_resolved_contact
            selected_anchor_displacement = (
                contact_anchor_displacement[selected] * selected_resolved_contact
            )
            resolved_contact_count.index_add_(
                0, selected_index, selected_resolved_contact
            )
            resolved_force_sum.index_add_(
                0, selected_index, resolved_support_force[selected]
            )
            contact_slip_speed_sum.index_add_(
                0, selected_index, selected_contact_slip_speed
            )
            contact_slip_speed_sq_sum.index_add_(
                0, selected_index, selected_contact_slip_speed.square()
            )
            contact_slip_sample_count.index_add_(
                0, selected_index, selected_resolved_contact
            )
            contact_slip_violation_count.index_add_(
                0,
                selected_index,
                (
                    (foot_speed[selected] >= args.slip_speed_threshold_mps)
                    & resolved_contact_mask[selected]
                ).to(dtype),
            )
            contact_anchor_displacement_sum.index_add_(
                0, selected_index, selected_anchor_displacement
            )
            contact_anchor_sample_count.index_add_(
                0, selected_index, selected_resolved_contact
            )
            double_support_count.index_add_(
                0,
                selected_index,
                resolved_contact_mask[selected].all(dim=1).to(dtype),
            )
            no_support_count.index_add_(
                0,
                selected_index,
                (~resolved_contact_mask[selected].any(dim=1)).to(dtype),
            )

            root_xy_velocity = robot.data.root_lin_vel_w[:, :2].to(dtype)
            root_xy_speed = torch.linalg.vector_norm(root_xy_velocity, dim=1)
            root_xy_drift = torch.linalg.vector_norm(
                (
                    robot.data.root_pos_w[:, :2]
                    - unwrapped.scene.env_origins[:, :2]
                ).to(dtype),
                dim=1,
            )
            root_xy_abs_velocity_sum.index_add_(
                0, selected_index, root_xy_velocity[selected].abs()
            )
            root_xy_speed_sum.index_add_(0, selected_index, root_xy_speed[selected])
            root_xy_speed_sq_sum.index_add_(
                0, selected_index, root_xy_speed[selected].square()
            )
            root_xy_speed_violation_count.index_add_(
                0,
                selected_index,
                (root_xy_speed[selected] >= args.root_speed_threshold_mps).to(dtype),
            )
            root_xy_drift_sum.index_add_(0, selected_index, root_xy_drift[selected])
            root_xy_drift_sq_sum.index_add_(
                0, selected_index, root_xy_drift[selected].square()
            )

            wrist_error = (
                actual_q.index_select(1, right_wrist_ids)
                - reference_q.index_select(1, right_wrist_ids)
            ).to(dtype)
            wrist_ref_vel = reference_dq.index_select(1, right_wrist_ids).to(dtype)
            for motion_id in range(num_motions):
                mask = eligible & (motion_ids == motion_id)
                if not torch.any(mask):
                    continue
                # new change
                motion_root_speed_max[motion_id] = torch.maximum(
                    motion_root_speed_max[motion_id], root_xy_speed[mask].max()
                )
                motion_root_drift_max[motion_id] = torch.maximum(
                    motion_root_drift_max[motion_id], root_xy_drift[mask].max()
                )
                for foot_id in range(2):
                    contact_for_motion = mask & resolved_contact_mask[:, foot_id]
                    if torch.any(contact_for_motion):
                        motion_contact_slip_speed_max[motion_id, foot_id] = torch.maximum(
                            motion_contact_slip_speed_max[motion_id, foot_id],
                            foot_speed[contact_for_motion, foot_id].max(),
                        )
                        motion_contact_anchor_displacement_max[
                            motion_id, foot_id
                        ] = torch.maximum(
                            motion_contact_anchor_displacement_max[motion_id, foot_id],
                            contact_anchor_displacement[
                                contact_for_motion, foot_id
                            ].max(),
                        )
                count = mask.sum().to(dtype)
                wrist_count[motion_id] += count
                wrist_error_sum[motion_id] += wrist_error[mask].sum(dim=0)
                wrist_abs_error_sum[motion_id] += wrist_error[mask].abs().sum(dim=0)
                wrist_error_sq_sum[motion_id] += wrist_error[mask].square().sum(dim=0)
                wrist_ref_vel_sum[motion_id] += wrist_ref_vel[mask].sum(dim=0)
                wrist_ref_vel_sq_sum[motion_id] += wrist_ref_vel[mask].square().sum(dim=0)
                wrist_error_ref_vel_sum[motion_id] += (
                    wrist_error[mask] * wrist_ref_vel[mask]
                ).sum(dim=0)

            if step_index + 1 == 250:
                # new change
                # Report where PhysX actually attributes contacts.  The
                # official G1 asset may not attach sole collisions to the
                # ankle-roll body assumed by older SoftMimic configurations.
                all_contact_force = (
                    contact_sensor.data.net_forces_w_history.norm(dim=-1)
                    .max(dim=1)
                    .values.mean(dim=0)
                )
                top_values, top_ids = torch.topk(
                    all_contact_force, k=min(6, all_contact_force.numel())
                )
                top_contacts = [
                    (contact_sensor.body_names[int(body_id)], float(value))
                    for body_id, value in zip(top_ids.cpu().tolist(), top_values.cpu().tolist())
                ]
                print(f"[STAND_DIAG] top_contact_bodies={top_contacts}", flush=True)
            if (step_index + 1) % 250 == 0:
                print(f"[STAND_DIAG] stepped={step_index + 1}/{args.steps}", flush=True)

        phase_count = sample_count.reshape(num_motions, args.phase_bins)
        phase_keypoint = _safe_divide(
            keypoint_sum, sample_count.unsqueeze(1)
        ).reshape(num_motions, args.phase_bins, 6)
        phase_keypoint_rms = torch.sqrt(
            _safe_divide(keypoint_sq_sum, sample_count.unsqueeze(1))
        ).reshape(num_motions, args.phase_bins, 6)
        phase_waist_q = _safe_divide(
            waist_q_abs_sum, sample_count.unsqueeze(1)
        ).reshape(num_motions, args.phase_bins, 3)
        # new change
        phase_waist_q_signed = _safe_divide(
            waist_q_signed_sum, sample_count.unsqueeze(1)
        ).reshape(num_motions, args.phase_bins, 3)
        phase_waist_dq = _safe_divide(
            waist_dq_abs_sum, sample_count.unsqueeze(1)
        ).reshape(num_motions, args.phase_bins, 3)
        # new change
        phase_support_q_abs = _safe_divide(
            support_q_abs_sum, sample_count.unsqueeze(1)
        ).reshape(num_motions, args.phase_bins, len(support_joint_names))
        phase_support_q_signed = _safe_divide(
            support_q_signed_sum, sample_count.unsqueeze(1)
        ).reshape(num_motions, args.phase_bins, len(support_joint_names))
        phase_support_target_abs = _safe_divide(
            support_target_abs_sum, sample_count.unsqueeze(1)
        ).reshape(num_motions, args.phase_bins, len(support_joint_names))
        # new change
        phase_arm_q_abs = _safe_divide(
            arm_q_abs_sum, sample_count.unsqueeze(1)
        ).reshape(num_motions, args.phase_bins, len(arm_joint_names))
        phase_arm_q_signed = _safe_divide(
            arm_q_signed_sum, sample_count.unsqueeze(1)
        ).reshape(num_motions, args.phase_bins, len(arm_joint_names))
        phase_arm_target_abs = _safe_divide(
            arm_target_abs_sum, sample_count.unsqueeze(1)
        ).reshape(num_motions, args.phase_bins, len(arm_joint_names))
        phase_com = _safe_divide(
            com_support_abs_sum, sample_count.unsqueeze(1)
        ).reshape(num_motions, args.phase_bins, 3)
        phase_slip = _safe_divide(foot_slip_sum, sample_count.unsqueeze(1)).reshape(
            num_motions, args.phase_bins, 2
        )
        phase_slip_rms = torch.sqrt(
            _safe_divide(foot_slip_sq_sum, sample_count.unsqueeze(1))
        ).reshape(num_motions, args.phase_bins, 2)
        phase_contact_ratio = _safe_divide(
            foot_contact_count, sample_count.unsqueeze(1)
        ).reshape(num_motions, args.phase_bins, 2)
        # new change
        phase_resolved_contact_ratio = _safe_divide(
            resolved_contact_count, sample_count.unsqueeze(1)
        ).reshape(num_motions, args.phase_bins, 2)
        phase_resolved_force = _safe_divide(
            resolved_force_sum, sample_count.unsqueeze(1)
        ).reshape(num_motions, args.phase_bins, 2)
        phase_contact_slip_speed = _safe_divide(
            contact_slip_speed_sum, contact_slip_sample_count
        ).reshape(num_motions, args.phase_bins, 2)
        phase_contact_slip_speed_rms = torch.sqrt(
            _safe_divide(contact_slip_speed_sq_sum, contact_slip_sample_count)
        ).reshape(num_motions, args.phase_bins, 2)
        phase_contact_slip_violation = _safe_divide(
            contact_slip_violation_count, contact_slip_sample_count
        ).reshape(num_motions, args.phase_bins, 2)
        phase_contact_anchor_displacement = _safe_divide(
            contact_anchor_displacement_sum, contact_anchor_sample_count
        ).reshape(num_motions, args.phase_bins, 2)
        phase_double_support_ratio = _safe_divide(
            double_support_count, sample_count
        ).reshape(num_motions, args.phase_bins)
        phase_no_support_ratio = _safe_divide(
            no_support_count, sample_count
        ).reshape(num_motions, args.phase_bins)
        phase_root_xy_abs_velocity = _safe_divide(
            root_xy_abs_velocity_sum, sample_count.unsqueeze(1)
        ).reshape(num_motions, args.phase_bins, 2)
        phase_root_xy_speed = _safe_divide(
            root_xy_speed_sum, sample_count
        ).reshape(num_motions, args.phase_bins)
        phase_root_xy_speed_rms = torch.sqrt(
            _safe_divide(root_xy_speed_sq_sum, sample_count)
        ).reshape(num_motions, args.phase_bins)
        phase_root_speed_violation = _safe_divide(
            root_xy_speed_violation_count, sample_count
        ).reshape(num_motions, args.phase_bins)
        phase_root_xy_drift = _safe_divide(
            root_xy_drift_sum, sample_count
        ).reshape(num_motions, args.phase_bins)
        phase_root_xy_drift_rms = torch.sqrt(
            _safe_divide(root_xy_drift_sq_sum, sample_count)
        ).reshape(num_motions, args.phase_bins)

        phase_csv = output_dir / "phase_bins.csv"
        with phase_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "motion",
                    "phase_bin",
                    "source_time_start_s",
                    "source_time_end_s",
                    "samples",
                    *[f"{name}_mae_m" for name in arm_keypoint_names],
                    *[f"{name}_rms_m" for name in arm_keypoint_names],
                    *[f"{name}_q_mae_deg" for name in waist_names],
                    *[f"{name}_dq_mae_deg_s" for name in waist_names],
                    # new change
                    *[f"{name}_q_ref_mae_deg" for name in arm_joint_names],
                    *[f"{name}_q_ref_bias_deg" for name in arm_joint_names],
                    *[f"{name}_q_target_mae_deg" for name in arm_joint_names],
                    "com_support_ap_mae_m",
                    "com_support_lateral_mae_m",
                    "com_support_distance_mae_m",
                    "left_foot_slip_mae_mps",
                    "right_foot_slip_mae_mps",
                    "left_foot_slip_rms_mps",
                    "right_foot_slip_rms_mps",
                    "legacy_left_ankle_sensor_contact_ratio",
                    "legacy_right_ankle_sensor_contact_ratio",
                    # new change
                    "resolved_left_support_contact_ratio",
                    "resolved_right_support_contact_ratio",
                    "resolved_left_support_force_mean_n",
                    "resolved_right_support_force_mean_n",
                    "left_contact_slip_speed_mae_mps",
                    "right_contact_slip_speed_mae_mps",
                    "left_contact_slip_speed_rms_mps",
                    "right_contact_slip_speed_rms_mps",
                    "left_contact_slip_violation_ratio",
                    "right_contact_slip_violation_ratio",
                    "left_contact_anchor_displacement_mae_m",
                    "right_contact_anchor_displacement_mae_m",
                    "double_support_ratio",
                    "no_support_ratio",
                    "root_abs_vx_mae_mps",
                    "root_abs_vy_mae_mps",
                    "root_xy_speed_mae_mps",
                    "root_xy_speed_rms_mps",
                    "root_speed_violation_ratio",
                    "root_xy_drift_mae_m",
                    "root_xy_drift_rms_m",
                ]
            )
            for motion_id, motion_name in enumerate(motion_names):
                for phase_id in range(args.phase_bins):
                    writer.writerow(
                        [
                            motion_name,
                            phase_id,
                            phase_id * phase_width_s,
                            (phase_id + 1) * phase_width_s,
                            int(phase_count[motion_id, phase_id].item()),
                            *phase_keypoint[motion_id, phase_id].cpu().tolist(),
                            *phase_keypoint_rms[motion_id, phase_id].cpu().tolist(),
                            *torch.rad2deg(phase_waist_q[motion_id, phase_id]).cpu().tolist(),
                            *torch.rad2deg(phase_waist_dq[motion_id, phase_id]).cpu().tolist(),
                            # new change
                            *torch.rad2deg(phase_arm_q_abs[motion_id, phase_id]).cpu().tolist(),
                            *torch.rad2deg(phase_arm_q_signed[motion_id, phase_id]).cpu().tolist(),
                            *torch.rad2deg(phase_arm_target_abs[motion_id, phase_id]).cpu().tolist(),
                            *phase_com[motion_id, phase_id].cpu().tolist(),
                            *phase_slip[motion_id, phase_id].cpu().tolist(),
                            *phase_slip_rms[motion_id, phase_id].cpu().tolist(),
                            *phase_contact_ratio[motion_id, phase_id].cpu().tolist(),
                            # new change
                            *phase_resolved_contact_ratio[
                                motion_id, phase_id
                            ].cpu().tolist(),
                            *phase_resolved_force[motion_id, phase_id].cpu().tolist(),
                            *phase_contact_slip_speed[
                                motion_id, phase_id
                            ].cpu().tolist(),
                            *phase_contact_slip_speed_rms[
                                motion_id, phase_id
                            ].cpu().tolist(),
                            *phase_contact_slip_violation[
                                motion_id, phase_id
                            ].cpu().tolist(),
                            *phase_contact_anchor_displacement[
                                motion_id, phase_id
                            ].cpu().tolist(),
                            float(phase_double_support_ratio[motion_id, phase_id]),
                            float(phase_no_support_ratio[motion_id, phase_id]),
                            *phase_root_xy_abs_velocity[
                                motion_id, phase_id
                            ].cpu().tolist(),
                            float(phase_root_xy_speed[motion_id, phase_id]),
                            float(phase_root_xy_speed_rms[motion_id, phase_id]),
                            float(phase_root_speed_violation[motion_id, phase_id]),
                            float(phase_root_xy_drift[motion_id, phase_id]),
                            float(phase_root_xy_drift_rms[motion_id, phase_id]),
                        ]
                    )

        first_bad = first_bad_step <= args.steps
        failure_csv = output_dir / "failure_events.csv"
        failure_rows: list[dict[str, object]] = []
        for env_id in torch.nonzero(first_bad, as_tuple=False).squeeze(-1).cpu().tolist():
            row = {
                "env_id": env_id,
                "motion": motion_names[env_id % num_motions],
                "rollout_step": int(first_bad_step[env_id].item()),
                "relative_episode_time_s": float(first_bad_relative_time[env_id].item()),
                "source_time_s": float(first_bad_source_time[env_id].item()),
            }
            failure_rows.append(row)
        with failure_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "env_id",
                    "motion",
                    "rollout_step",
                    "relative_episode_time_s",
                    "source_time_s",
                ),
            )
            writer.writeheader()
            writer.writerows(failure_rows)

        # new change
        # Keep the legacy bad-orientation file above, and add the complete
        # non-timeout termination ledger requested by StationarySupportDiagnosis.
        first_failure = first_failure_step <= args.steps
        support_failure_rows: list[dict[str, object]] = []
        for env_id in torch.nonzero(
            first_failure, as_tuple=False
        ).squeeze(-1).cpu().tolist():
            support_failure_rows.append(
                {
                    "env_id": env_id,
                    "motion": motion_names[env_id % num_motions],
                    "rollout_step": int(first_failure_step[env_id].item()),
                    "relative_episode_time_s": float(
                        first_failure_relative_time[env_id].item()
                    ),
                    "source_time_s": float(first_failure_source_time[env_id].item()),
                    "phase_bin": min(
                        args.phase_bins - 1,
                        int(first_failure_source_time[env_id].item() / phase_width_s),
                    ),
                    "bad_orientation": bool(
                        first_failure_bad_orientation[env_id].item()
                    ),
                    "base_contact": bool(first_failure_base_contact[env_id].item()),
                }
            )
        support_failure_csv = output_dir / "stationary_support_failure_events.csv"
        with support_failure_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "env_id",
                    "motion",
                    "rollout_step",
                    "relative_episode_time_s",
                    "source_time_s",
                    "phase_bin",
                    "bad_orientation",
                    "base_contact",
                ),
            )
            writer.writeheader()
            writer.writerows(support_failure_rows)

        motion_summaries: dict[str, object] = {}
        for motion_id, motion_name in enumerate(motion_names):
            counts = phase_count[motion_id]
            total = counts.sum()
            keypoint_mean = _safe_divide(
                (phase_keypoint[motion_id] * counts.unsqueeze(1)).sum(dim=0), total
            )
            waist_q_mean = _safe_divide(
                (phase_waist_q[motion_id] * counts.unsqueeze(1)).sum(dim=0), total
            )
            # new change
            waist_q_signed_mean = _safe_divide(
                (phase_waist_q_signed[motion_id] * counts.unsqueeze(1)).sum(dim=0),
                total,
            )
            waist_dq_mean = _safe_divide(
                (phase_waist_dq[motion_id] * counts.unsqueeze(1)).sum(dim=0), total
            )
            # new change
            support_q_abs_mean = _safe_divide(
                (phase_support_q_abs[motion_id] * counts.unsqueeze(1)).sum(dim=0),
                total,
            )
            support_q_signed_mean = _safe_divide(
                (
                    phase_support_q_signed[motion_id]
                    * counts.unsqueeze(1)
                ).sum(dim=0),
                total,
            )
            support_target_abs_mean = _safe_divide(
                (
                    phase_support_target_abs[motion_id]
                    * counts.unsqueeze(1)
                ).sum(dim=0),
                total,
            )
            # new change
            arm_q_abs_mean = _safe_divide(
                (phase_arm_q_abs[motion_id] * counts.unsqueeze(1)).sum(dim=0), total
            )
            arm_q_signed_mean = _safe_divide(
                (phase_arm_q_signed[motion_id] * counts.unsqueeze(1)).sum(dim=0), total
            )
            arm_target_abs_mean = _safe_divide(
                (phase_arm_target_abs[motion_id] * counts.unsqueeze(1)).sum(dim=0), total
            )
            com_mean = _safe_divide(
                (phase_com[motion_id] * counts.unsqueeze(1)).sum(dim=0), total
            )
            motion_bin_slice = slice(
                motion_id * args.phase_bins, (motion_id + 1) * args.phase_bins
            )
            motion_slip_mean = _safe_divide(
                foot_slip_sum[motion_bin_slice].sum(dim=0), total
            )
            motion_contact_count = foot_contact_count[motion_bin_slice].sum(dim=0)
            motion_contact_ratio = _safe_divide(motion_contact_count, total)
            # new change
            motion_resolved_contact_count = resolved_contact_count[
                motion_bin_slice
            ].sum(dim=0)
            motion_resolved_contact_ratio = _safe_divide(
                motion_resolved_contact_count, total
            )
            motion_resolved_force_mean = _safe_divide(
                resolved_force_sum[motion_bin_slice].sum(dim=0), total
            )
            motion_contact_slip_count = contact_slip_sample_count[
                motion_bin_slice
            ].sum(dim=0)
            motion_contact_slip_mean = _safe_divide(
                contact_slip_speed_sum[motion_bin_slice].sum(dim=0),
                motion_contact_slip_count,
            )
            motion_contact_slip_rms = torch.sqrt(
                _safe_divide(
                    contact_slip_speed_sq_sum[motion_bin_slice].sum(dim=0),
                    motion_contact_slip_count,
                )
            )
            motion_contact_slip_violation = _safe_divide(
                contact_slip_violation_count[motion_bin_slice].sum(dim=0),
                motion_contact_slip_count,
            )
            motion_anchor_displacement_mean = _safe_divide(
                contact_anchor_displacement_sum[motion_bin_slice].sum(dim=0),
                contact_anchor_sample_count[motion_bin_slice].sum(dim=0),
            )
            motion_double_support_ratio = _safe_divide(
                double_support_count[motion_bin_slice].sum(), total
            )
            motion_no_support_ratio = _safe_divide(
                no_support_count[motion_bin_slice].sum(), total
            )
            motion_root_abs_velocity_mean = _safe_divide(
                root_xy_abs_velocity_sum[motion_bin_slice].sum(dim=0), total
            )
            motion_root_speed_mean = _safe_divide(
                root_xy_speed_sum[motion_bin_slice].sum(), total
            )
            motion_root_speed_rms = torch.sqrt(
                _safe_divide(root_xy_speed_sq_sum[motion_bin_slice].sum(), total)
            )
            motion_root_speed_violation = _safe_divide(
                root_xy_speed_violation_count[motion_bin_slice].sum(), total
            )
            motion_root_drift_mean = _safe_divide(
                root_xy_drift_sum[motion_bin_slice].sum(), total
            )
            motion_root_drift_rms = torch.sqrt(
                _safe_divide(root_xy_drift_sq_sum[motion_bin_slice].sum(), total)
            )

            wrist_analysis: dict[str, object] = {}
            for joint_id, joint_name in enumerate(right_wrist_names):
                count = wrist_count[motion_id, joint_id]
                mean_e = _safe_divide(wrist_error_sum[motion_id, joint_id], count)
                mae_e = _safe_divide(wrist_abs_error_sum[motion_id, joint_id], count)
                mean_v = _safe_divide(wrist_ref_vel_sum[motion_id, joint_id], count)
                var_v = torch.clamp(
                    _safe_divide(wrist_ref_vel_sq_sum[motion_id, joint_id], count)
                    - mean_v.square(),
                    min=0.0,
                )
                cov_ev = (
                    _safe_divide(wrist_error_ref_vel_sum[motion_id, joint_id], count)
                    - mean_e * mean_v
                )
                lag_s = -cov_ev / torch.clamp(var_v, min=1.0e-12)
                var_e = torch.clamp(
                    _safe_divide(wrist_error_sq_sum[motion_id, joint_id], count)
                    - mean_e.square(),
                    min=0.0,
                )
                corr = cov_ev / torch.sqrt(torch.clamp(var_e * var_v, min=1.0e-12))
                ref_vel_rms = torch.sqrt(
                    _safe_divide(wrist_ref_vel_sq_sum[motion_id, joint_id], count)
                )
                residual_var = torch.clamp(
                    var_e - cov_ev.square() / torch.clamp(var_v, min=1.0e-12), min=0.0
                )
                bias_fraction = torch.abs(mean_e) / torch.clamp(mae_e, min=1.0e-12)
                if torch.rad2deg(ref_vel_rms).item() < 0.5:
                    classification = "static_reference_bias"
                elif abs(corr.item()) >= 0.35 and lag_s.item() >= 0.02:
                    classification = "dynamic_lag"
                elif bias_fraction.item() >= 0.5:
                    classification = "fixed_bias_dominant"
                else:
                    classification = "mixed_or_nonlinear"
                wrist_analysis[joint_name] = {
                    "bias_deg": torch.rad2deg(mean_e).item(),
                    "mae_deg": torch.rad2deg(mae_e).item(),
                    "reference_velocity_rms_deg_s": torch.rad2deg(ref_vel_rms).item(),
                    "estimated_lag_s": lag_s.item() if var_v.item() > 1.0e-10 else None,
                    "error_reference_velocity_correlation": corr.item()
                    if var_v.item() > 1.0e-10
                    else None,
                    "residual_rms_after_bias_lag_deg": torch.rad2deg(
                        torch.sqrt(residual_var)
                    ).item(),
                    "classification": classification,
                }

            motion_failures = [row for row in failure_rows if row["motion"] == motion_name]
            source_failure_times = [float(row["source_time_s"]) for row in motion_failures]
            # new change
            support_failures = [
                row for row in support_failure_rows if row["motion"] == motion_name
            ]
            support_failure_times = [
                float(row["source_time_s"]) for row in support_failures
            ]
            motion_summaries[motion_name] = {
                "samples": int(total.item()),
                "first_bad_count": len(motion_failures),
                "failure_source_time_s": {
                    "p10": _percentile(source_failure_times, 0.10),
                    "median": statistics.median(source_failure_times)
                    if source_failure_times
                    else None,
                    "p90": _percentile(source_failure_times, 0.90),
                },
                "keypoint_mae_m": dict(
                    zip(arm_keypoint_names, keypoint_mean.cpu().tolist())
                ),
                "waist_q_mae_deg": dict(
                    zip(waist_names, torch.rad2deg(waist_q_mean).cpu().tolist())
                ),
                # new change
                "waist_q_ref_bias_deg": dict(
                    zip(
                        waist_names,
                        torch.rad2deg(waist_q_signed_mean).cpu().tolist(),
                    )
                ),
                "waist_dq_mae_deg_s": dict(
                    zip(waist_names, torch.rad2deg(waist_dq_mean).cpu().tolist())
                ),
                # new change
                "support_joint_q_ref_mae_deg": dict(
                    zip(
                        support_joint_names,
                        torch.rad2deg(support_q_abs_mean).cpu().tolist(),
                    )
                ),
                "support_joint_q_ref_bias_deg": dict(
                    zip(
                        support_joint_names,
                        torch.rad2deg(support_q_signed_mean).cpu().tolist(),
                    )
                ),
                "support_joint_q_target_mae_deg": dict(
                    zip(
                        support_joint_names,
                        torch.rad2deg(support_target_abs_mean).cpu().tolist(),
                    )
                ),
                # new change
                "arm_joint_q_ref_mae_deg": dict(
                    zip(arm_joint_names, torch.rad2deg(arm_q_abs_mean).cpu().tolist())
                ),
                "arm_joint_q_ref_bias_deg": dict(
                    zip(arm_joint_names, torch.rad2deg(arm_q_signed_mean).cpu().tolist())
                ),
                "arm_joint_q_target_mae_deg": dict(
                    zip(arm_joint_names, torch.rad2deg(arm_target_abs_mean).cpu().tolist())
                ),
                "com_support_mae_m": dict(
                    zip(("ap", "lateral", "distance"), com_mean.cpu().tolist())
                ),
                "unconditional_foot_xy_speed_mae_mps": dict(
                    zip(("left", "right"), motion_slip_mean.cpu().tolist())
                ),
                "legacy_ankle_sensor_contact_ratio": dict(
                    zip(("left", "right"), motion_contact_ratio.cpu().tolist())
                ),
                # new change
                "stationary_support": {
                    "first_failure_count": len(support_failures),
                    "bad_orientation_count": sum(
                        bool(row["bad_orientation"]) for row in support_failures
                    ),
                    "base_contact_count": sum(
                        bool(row["base_contact"]) for row in support_failures
                    ),
                    "failure_source_time_s": {
                        "p10": _percentile(support_failure_times, 0.10),
                        "median": statistics.median(support_failure_times)
                        if support_failure_times
                        else None,
                        "p90": _percentile(support_failure_times, 0.90),
                    },
                    "resolved_contact_ratio": dict(
                        zip(
                            ("left", "right"),
                            motion_resolved_contact_ratio.cpu().tolist(),
                        )
                    ),
                    "resolved_vertical_force_mean_n": dict(
                        zip(
                            ("left", "right"),
                            motion_resolved_force_mean.cpu().tolist(),
                        )
                    ),
                    "double_support_ratio": float(motion_double_support_ratio),
                    "no_support_ratio": float(motion_no_support_ratio),
                    "contact_slip_speed_mae_mps": dict(
                        zip(
                            ("left", "right"),
                            motion_contact_slip_mean.cpu().tolist(),
                        )
                    ),
                    "contact_slip_speed_rms_mps": dict(
                        zip(
                            ("left", "right"),
                            motion_contact_slip_rms.cpu().tolist(),
                        )
                    ),
                    "contact_slip_speed_max_mps": dict(
                        zip(
                            ("left", "right"),
                            motion_contact_slip_speed_max[motion_id].cpu().tolist(),
                        )
                    ),
                    "contact_slip_violation_ratio": dict(
                        zip(
                            ("left", "right"),
                            motion_contact_slip_violation.cpu().tolist(),
                        )
                    ),
                    "contact_anchor_displacement_mae_m": dict(
                        zip(
                            ("left", "right"),
                            motion_anchor_displacement_mean.cpu().tolist(),
                        )
                    ),
                    "contact_anchor_displacement_max_m": dict(
                        zip(
                            ("left", "right"),
                            motion_contact_anchor_displacement_max[
                                motion_id
                            ].cpu().tolist(),
                        )
                    ),
                    "root_abs_velocity_mae_mps": dict(
                        zip(
                            ("x", "y"),
                            motion_root_abs_velocity_mean.cpu().tolist(),
                        )
                    ),
                    "root_xy_speed_mae_mps": float(motion_root_speed_mean),
                    "root_xy_speed_rms_mps": float(motion_root_speed_rms),
                    "root_xy_speed_max_mps": float(
                        motion_root_speed_max[motion_id]
                    ),
                    "root_speed_violation_ratio": float(
                        motion_root_speed_violation
                    ),
                    "root_xy_drift_mae_m": float(motion_root_drift_mean),
                    "root_xy_drift_rms_m": float(motion_root_drift_rms),
                    "root_xy_drift_max_m": float(motion_root_drift_max[motion_id]),
                },
                "right_wrist_error_model": wrist_analysis,
            }

        # new change
        proximal_feedback_mean = _safe_divide(
            proximal_feedback_abs_sum, proximal_feedback_sample_count
        )
        proximal_feedback_saturation = _safe_divide(
            proximal_feedback_saturated_count, proximal_feedback_sample_count
        )
        proximal_feedback_stats = {
            "mean_abs_target_correction_rad": dict(
                zip(proximal_joint_names, proximal_feedback_mean.cpu().tolist())
            ),
            "max_abs_target_correction_rad": dict(
                zip(proximal_joint_names, proximal_feedback_max.cpu().tolist())
            ),
            "saturation_fraction": dict(
                zip(proximal_joint_names, proximal_feedback_saturation.cpu().tolist())
            ),
            "overall_mean_abs_target_correction_rad": float(
                proximal_feedback_mean.mean().item()
            ),
            "overall_max_abs_target_correction_rad": float(
                proximal_feedback_max.max().item()
            ),
            "overall_saturation_fraction": float(
                proximal_feedback_saturation.mean().item()
            ),
        }

        # new change
        support_policy_residual_mean = _safe_divide(
            support_policy_residual_abs_sum,
            support_policy_residual_sample_count,
        )
        support_policy_residual_stats = {
            "mean_abs_target_residual_rad": dict(
                zip(
                    support_joint_names,
                    support_policy_residual_mean.cpu().tolist(),
                )
            ),
            "max_abs_target_residual_rad": dict(
                zip(
                    support_joint_names,
                    support_policy_residual_max.cpu().tolist(),
                )
            ),
            "overall_mean_abs_target_residual_rad": float(
                support_policy_residual_mean.mean().item()
            ),
            "overall_max_abs_target_residual_rad": float(
                support_policy_residual_max.max().item()
            ),
        }

        # new change
        global_support_feedback_mean = _safe_divide(
            global_support_feedback_abs_sum,
            global_support_feedback_sample_count,
        )
        global_support_feedback_stats = {
            "mean_abs_target_correction_rad": dict(
                zip(support_joint_names, global_support_feedback_mean.cpu().tolist())
            ),
            "max_abs_target_correction_rad": dict(
                zip(support_joint_names, global_support_feedback_max.cpu().tolist())
            ),
            "saturation_ratio": dict(
                zip(
                    support_joint_names,
                    _safe_divide(
                        global_support_feedback_saturated_count,
                        global_support_feedback_sample_count,
                    ).cpu().tolist(),
                )
            ),
            "overall_mean_abs_target_correction_rad": float(
                global_support_feedback_mean.mean().item()
            ),
            "overall_max_abs_target_correction_rad": float(
                global_support_feedback_max.max().item()
            ),
        }

        summary = {
            "task": args.task,
            "jit_policy": str(pathlib.Path(args.jit_policy).resolve()),
            # new change
            "jit_extra_obs_groups": list(args.jit_extra_obs_groups),
            "seed": args.seed,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "phase_bins": args.phase_bins,
            "phase_width_s": phase_width_s,
            "episode_warmup_steps": args.episode_warmup_steps,
            # new change
            "support_feedback_gain": args.support_feedback_gain,
            "support_feedback_limit_rad": args.support_feedback_limit_rad,
            # new change
            "support_feedback_excluded_motion_ids": list(
                args.support_feedback_excluded_motion_ids
            ),
            # new change
            "support_feedback_enabled_joint_names": list(
                args.support_feedback_enabled_joint_names
            ),
            # new change
            "support_feedback_safety_gate": {
                "require_double_support": args.support_feedback_require_double_support,
                "max_com_distance_m": args.support_feedback_max_com_distance_m,
                "max_tilt_rad": args.support_feedback_max_tilt_rad,
                "min_episode_age": args.support_feedback_min_episode_age,
                "smooth_gate": args.support_feedback_smooth_gate,
                "active_fraction": float(
                    _safe_divide(
                        support_feedback_gate_active_count,
                        support_feedback_gate_sample_count,
                    ).item()
                )
                if support_feedback_gate_sample_count.item() > 0
                else 0.0,
            },
            "support_policy_baseline_jit": args.support_policy_baseline_jit,
            "support_policy_residual_scale": args.support_policy_residual_scale,
            # new change
            "support_actor_residual_scale": args.support_actor_residual_scale,
            # new change
            "global_support_feedback": {
                "limit_rad": args.global_support_feedback_limit_rad,
                "velocity_horizon_s": args.global_support_feedback_velocity_horizon_s,
                "com_weight": args.global_support_feedback_com_weight,
                "error_scale_m": args.global_support_feedback_error_scale_m,
                "min_episode_age": args.global_support_feedback_min_episode_age,
                "joint_coordination_order": list(support_joint_names),
                "stats": global_support_feedback_stats,
            },
            "support_policy_residual_stats": support_policy_residual_stats,
            # new change
            "knee_target_bias_rad": list(args.knee_target_bias_rad),
            # new change
            "waist_target_bias_rad": list(args.waist_target_bias_rad),
            # new change
            "waist_error_feedback_gain": list(args.waist_error_feedback_gain),
            "waist_error_feedback_limit_rad": list(
                args.waist_error_feedback_limit_rad
            ),
            # new change
            "right_wrist_feedback_gain": args.right_wrist_feedback_gain,
            "right_wrist_feedback_limit_rad": args.right_wrist_feedback_limit_rad,
            # new change
            "bilateral_arm_feedback_gain": args.bilateral_arm_feedback_gain,
            "bilateral_arm_feedback_limit_rad": args.bilateral_arm_feedback_limit_rad,
            # new change
            "bilateral_proximal_feedback_gain": args.bilateral_proximal_feedback_gain,
            "bilateral_proximal_feedback_limit_rad": (
                args.bilateral_proximal_feedback_limit_rad
            ),
            "bilateral_proximal_feedback_motion_ids": list(
                args.bilateral_proximal_feedback_motion_ids
            ),
            "bilateral_proximal_feedback_stats": proximal_feedback_stats,
            "first_bad_count": int(first_bad.sum().item()),
            # new change
            "stationary_support_diagnosis": {
                "zero_external_force_required": True,
                "resolved_support_sensor_rows": support_contact_names,
                "legacy_ankle_sensor_rows": contact_names,
                "support_contact_threshold_n": args.support_contact_threshold_n,
                "slip_speed_threshold_mps": args.slip_speed_threshold_mps,
                "root_speed_threshold_mps": args.root_speed_threshold_mps,
                "first_failure_count": int(first_failure.sum().item()),
            },
            "motions": motion_summaries,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

        # new change
        support_summary_csv = output_dir / "stationary_support_motion_summary.csv"
        with support_summary_csv.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = (
                "motion",
                "first_failure_count",
                "bad_orientation_count",
                "base_contact_count",
                "failure_phase_p10_s",
                "failure_phase_median_s",
                "failure_phase_p90_s",
                "left_contact_ratio",
                "right_contact_ratio",
                "double_support_ratio",
                "no_support_ratio",
                "left_force_mean_n",
                "right_force_mean_n",
                "left_contact_slip_mae_mps",
                "right_contact_slip_mae_mps",
                "left_contact_slip_max_mps",
                "right_contact_slip_max_mps",
                "left_slip_violation_ratio",
                "right_slip_violation_ratio",
                "left_anchor_displacement_max_m",
                "right_anchor_displacement_max_m",
                "root_xy_speed_mae_mps",
                "root_xy_speed_rms_mps",
                "root_xy_speed_max_mps",
                "root_speed_violation_ratio",
                "root_xy_drift_mae_m",
                "root_xy_drift_max_m",
            )
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for motion_name, values in motion_summaries.items():
                support = values["stationary_support"]
                failure_phase = support["failure_source_time_s"]
                writer.writerow(
                    {
                        "motion": motion_name,
                        "first_failure_count": support["first_failure_count"],
                        "bad_orientation_count": support["bad_orientation_count"],
                        "base_contact_count": support["base_contact_count"],
                        "failure_phase_p10_s": failure_phase["p10"],
                        "failure_phase_median_s": failure_phase["median"],
                        "failure_phase_p90_s": failure_phase["p90"],
                        "left_contact_ratio": support["resolved_contact_ratio"]["left"],
                        "right_contact_ratio": support["resolved_contact_ratio"]["right"],
                        "double_support_ratio": support["double_support_ratio"],
                        "no_support_ratio": support["no_support_ratio"],
                        "left_force_mean_n": support["resolved_vertical_force_mean_n"]["left"],
                        "right_force_mean_n": support["resolved_vertical_force_mean_n"]["right"],
                        "left_contact_slip_mae_mps": support["contact_slip_speed_mae_mps"]["left"],
                        "right_contact_slip_mae_mps": support["contact_slip_speed_mae_mps"]["right"],
                        "left_contact_slip_max_mps": support["contact_slip_speed_max_mps"]["left"],
                        "right_contact_slip_max_mps": support["contact_slip_speed_max_mps"]["right"],
                        "left_slip_violation_ratio": support["contact_slip_violation_ratio"]["left"],
                        "right_slip_violation_ratio": support["contact_slip_violation_ratio"]["right"],
                        "left_anchor_displacement_max_m": support["contact_anchor_displacement_max_m"]["left"],
                        "right_anchor_displacement_max_m": support["contact_anchor_displacement_max_m"]["right"],
                        "root_xy_speed_mae_mps": support["root_xy_speed_mae_mps"],
                        "root_xy_speed_rms_mps": support["root_xy_speed_rms_mps"],
                        "root_xy_speed_max_mps": support["root_xy_speed_max_mps"],
                        "root_speed_violation_ratio": support["root_speed_violation_ratio"],
                        "root_xy_drift_mae_m": support["root_xy_drift_mae_m"],
                        "root_xy_drift_max_m": support["root_xy_drift_max_m"],
                    }
                )

        report_lines = [
            "# StandCoordinationV1 phase-resolved diagnosis",
            "",
            f"- Policy: `{pathlib.Path(args.jit_policy).name}`",
            f"- Population: {args.num_envs} environments × {args.steps} steps, seed {args.seed}",
            f"- Accuracy warm-up exclusion: {args.episode_warmup_steps} steps after every reset",
            # new change
            f"- Support feedback gain/limit: {args.support_feedback_gain} / "
            f"{args.support_feedback_limit_rad} rad",
            # new change
            f"- Frozen actor support-residual scale: "
            f"{args.support_actor_residual_scale}",
            # new change
            f"- Global support feedback [limit/tau/com/error-scale/age]: "
            f"{args.global_support_feedback_limit_rad:.6f} rad / "
            f"{args.global_support_feedback_velocity_horizon_s:.3f} s / "
            f"{args.global_support_feedback_com_weight:.3f} / "
            f"{args.global_support_feedback_error_scale_m:.3f} m / "
            f"{args.global_support_feedback_min_episode_age}",
            f"- Global support correction mean/max: "
            f"{global_support_feedback_stats['overall_mean_abs_target_correction_rad']:.8f} / "
            f"{global_support_feedback_stats['overall_max_abs_target_correction_rad']:.8f} rad",
            # new change
            f"- Support feedback excluded motion ids: "
            f"{list(args.support_feedback_excluded_motion_ids)}",
            # new change
            f"- Support feedback enabled joints: "
            f"{list(args.support_feedback_enabled_joint_names) or 'all ten'}",
            # new change
            f"- Support feedback safety gate [double/com/tilt/age/active]: "
            f"{args.support_feedback_require_double_support} / "
            f"{args.support_feedback_max_com_distance_m:.4f} m / "
            f"{args.support_feedback_max_tilt_rad:.4f} rad / "
            f"{args.support_feedback_min_episode_age} / "
            f"{summary['support_feedback_safety_gate']['active_fraction']:.6f} "
            f"(smooth={args.support_feedback_smooth_gate})",
            # new change
            f"- Learned support residual mean/max: "
            f"{support_policy_residual_stats['overall_mean_abs_target_residual_rad']:.8f} / "
            f"{support_policy_residual_stats['overall_max_abs_target_residual_rad']:.8f} rad",
            # new change
            f"- Knee target bias [left, right] rad: {list(args.knee_target_bias_rad)}",
            # new change
            f"- Waist target bias [yaw, roll, pitch] rad: {list(args.waist_target_bias_rad)}",
            # new change
            f"- Waist error-feedback gain [yaw, roll, pitch]: "
            f"{list(args.waist_error_feedback_gain)}",
            f"- Waist error-feedback limit [yaw, roll, pitch] rad: "
            f"{list(args.waist_error_feedback_limit_rad)}",
            # new change
            f"- Right-wrist feedback gain/limit: {args.right_wrist_feedback_gain} / "
            f"{args.right_wrist_feedback_limit_rad} rad",
            # new change
            f"- Bilateral-arm feedback gain/limit: {args.bilateral_arm_feedback_gain} / "
            f"{args.bilateral_arm_feedback_limit_rad} rad",
            # new change
            f"- Bilateral-proximal feedback gain/limit: "
            f"{args.bilateral_proximal_feedback_gain} / "
            f"{args.bilateral_proximal_feedback_limit_rad} rad",
            f"- Bilateral-proximal feedback motion ids: "
            f"{list(args.bilateral_proximal_feedback_motion_ids) or 'all'}",
            f"- Proximal correction mean/max/saturation: "
            f"{proximal_feedback_stats['overall_mean_abs_target_correction_rad']:.6f} / "
            f"{proximal_feedback_stats['overall_max_abs_target_correction_rad']:.6f} rad / "
            f"{100.0 * proximal_feedback_stats['overall_saturation_fraction']:.3f}%",
            f"- First bad-orientation environments: {int(first_bad.sum().item())}/{args.num_envs}",
            # new change
            f"- First non-timeout failures: {int(first_failure.sum().item())}/{args.num_envs}",
            f"- Resolved support sensor rows: {support_contact_names} at "
            f"{args.support_contact_threshold_n:.1f} N vertical-force threshold",
            f"- Slip/root speed thresholds: {args.slip_speed_threshold_mps:.3f} / "
            f"{args.root_speed_threshold_mps:.3f} m/s",
            "",
            "## Motion summary",
            "",
            "| Motion | failures | shoulders mm | elbows mm | wrists mm | waist q deg | COM-support mm | foot XY speed mm/s | legacy contact % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for motion_name, values in motion_summaries.items():
            kp = values["keypoint_mae_m"]
            waist = values["waist_q_mae_deg"]
            com = values["com_support_mae_m"]
            slip = values["unconditional_foot_xy_speed_mae_mps"]
            contact = values["legacy_ankle_sensor_contact_ratio"]
            shoulder_mm = 500.0 * (kp["left_shoulder"] + kp["right_shoulder"])
            elbow_mm = 500.0 * (kp["left_elbow"] + kp["right_elbow"])
            wrist_mm = 500.0 * (kp["left_wrist"] + kp["right_wrist"])
            waist_deg = sum(waist.values()) / 3.0
            report_lines.append(
                f"| {motion_name} | {values['first_bad_count']} | {shoulder_mm:.2f} | "
                f"{elbow_mm:.2f} | {wrist_mm:.2f} | {waist_deg:.3f} | "
                f"{1000.0 * com['distance']:.2f} | "
                f"{500.0 * (slip['left'] + slip['right']):.1f} | "
                f"{50.0 * (contact['left'] + contact['right']):.1f} |"
            )

        report_lines.extend(
            [
                "",
                "The support center is the geometric midpoint of the two ankle-roll links and foot XY speed is unconditional. The legacy ankle contact percentage is reported only as a sensor-health audit because the official G1 contact sensor attributes the dominant forces to knee-link entries.",
                "",
                "## Stationary support diagnosis",
                "",
                "Contact-gated slip uses the empirically resolved left/right support-force rows. Anchor displacement is reset only on contact onset; it therefore measures stance-phase skating rather than ordinary airborne motion.",
                "",
                "| Motion | failures | L/R contact % | double % | L/R contact-slip mm/s | L/R max anchor drift mm | root speed mm/s | root max drift m | failure phase median s |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for motion_name, values in motion_summaries.items():
            support = values["stationary_support"]
            contact = support["resolved_contact_ratio"]
            slip = support["contact_slip_speed_mae_mps"]
            anchor = support["contact_anchor_displacement_max_m"]
            failure_median = support["failure_source_time_s"]["median"]
            failure_text = "n/a" if failure_median is None else f"{failure_median:.3f}"
            report_lines.append(
                f"| {motion_name} | {support['first_failure_count']} | "
                f"{100.0 * contact['left']:.1f}/{100.0 * contact['right']:.1f} | "
                f"{100.0 * support['double_support_ratio']:.1f} | "
                f"{1000.0 * slip['left']:.1f}/{1000.0 * slip['right']:.1f} | "
                f"{1000.0 * anchor['left']:.1f}/{1000.0 * anchor['right']:.1f} | "
                f"{1000.0 * support['root_xy_speed_mae_mps']:.1f} | "
                f"{support['root_xy_drift_max_m']:.3f} | {failure_text} |"
            )

        report_lines.extend(
            [
                "",
                "## Failure phase focus",
                "",
            ]
        )
        for motion_name in ("01_bilateral_sagittal", "06_pitch_coordination"):
            values = motion_summaries[motion_name]
            failure_time = values["failure_source_time_s"]
            report_lines.append(
                f"- {motion_name}: {values['first_bad_count']} failures; source-time "
                f"p10/median/p90 = {failure_time['p10']} / {failure_time['median']} / "
                f"{failure_time['p90']} s."
            )

        report_lines.extend(["", "## Right-wrist bias/lag model", ""])
        for motion_name, values in motion_summaries.items():
            report_lines.append(f"### {motion_name}")
            report_lines.append("")
            for joint_name, model in values["right_wrist_error_model"].items():
                lag = model["estimated_lag_s"]
                lag_text = "n/a" if lag is None else f"{lag:.4f}s"
                report_lines.append(
                    f"- {joint_name}: MAE {model['mae_deg']:.3f}°, bias "
                    f"{model['bias_deg']:+.3f}°, lag {lag_text}, "
                    f"classification `{model['classification']}`."
                )
            report_lines.append("")

        # new change
        global_sample_count = sample_count.sum()
        global_arm_q_abs = torch.rad2deg(
            _safe_divide(arm_q_abs_sum.sum(dim=0), global_sample_count)
        )
        global_arm_q_bias = torch.rad2deg(
            _safe_divide(arm_q_signed_sum.sum(dim=0), global_sample_count)
        )
        global_arm_target_abs = torch.rad2deg(
            _safe_divide(arm_target_abs_sum.sum(dim=0), global_sample_count)
        )
        report_lines.extend(
            [
                "## Bilateral arm q/ref and q/target diagnosis",
                "",
                "| Joint | q-ref MAE deg | q-ref bias deg | q-target MAE deg |",
                "|---|---:|---:|---:|",
            ]
        )
        for joint_index, joint_name in enumerate(arm_joint_names):
            report_lines.append(
                f"| {joint_name} | {global_arm_q_abs[joint_index].item():.3f} | "
                f"{global_arm_q_bias[joint_index].item():+.3f} | "
                f"{global_arm_target_abs[joint_index].item():.3f} |"
            )
        report_lines.append("")

        report_lines.extend(
            [
                "## Files",
                "",
                "- `summary.json`: machine-readable motion summaries and wrist models.",
                "- `phase_bins.csv`: every motion × source-time phase bin.",
                "- `failure_events.csv`: first bad-orientation event per affected environment.",
                # new change
                "- `stationary_support_motion_summary.csv`: one row per motion with resolved contact, stance slip, root velocity/drift, and failures.",
                "- `stationary_support_failure_events.csv`: first non-timeout failure and source phase for every affected environment.",
                "",
            ]
        )
        (output_dir / "report.md").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )
        print(
            f"[STAND_DIAG] complete output_dir={output_dir} "
            f"first_bad={int(first_bad.sum().item())}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # new change
        # Isaac Sim shutdown may otherwise hide the Python traceback behind
        # its own process-exit handling.
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
