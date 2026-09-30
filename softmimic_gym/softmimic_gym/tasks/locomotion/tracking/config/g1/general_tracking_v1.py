"""Nominal multi-motion tracking task for the strict G1 29DOF interface."""

from __future__ import annotations

# new change
import copy
import pathlib

import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm
# new change
from isaaclab.managers import EventTermCfg as EventTerm
# new change
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import softmimic_gym.tasks.locomotion.tracking.mdp as mdp
from softmimic_gym.assets.robots.g1_compat import (
    G1_SOFTMIMIC_29_JOINT_NAMES,
    G1_SOFTMIMIC_LEFT_ARM_ACTION_IDS,
    G1_SOFTMIMIC_RIGHT_ARM_ACTION_IDS,
    G1_SOFTMIMIC_KEYPOINT_BODY_NAMES,
    G1_SOFTMIMIC_NUM_JOINTS,
    g1_softmimic_joint_names,
)
from softmimic_gym.assets.robots.unitree import G1_SOFTMIMIC_29DOF_COMPAT_CFG
from softmimic_gym.tasks.locomotion.tracking.tracking_env_cfg import TrackingEnvCfg


# new change
_PROJECT_ROOT = pathlib.Path(__file__).absolute().parents[7]
_MOTION_ROOT = _PROJECT_ROOT / "datasets" / "motions_csv"

# new change
# Full recordings are used rather than the short *_single clips so every
# sampled 20-second episode can stay inside one continuous motion segment.
GENERAL_TRACKING_MOTIONS: tuple[str, ...] = (
    str(_MOTION_ROOT / "stand.csv"),
    str(_MOTION_ROOT / "walk.csv"),
    str(_MOTION_ROOT / "tpose.csv"),
    str(_MOTION_ROOT / "boxpick.csv"),
    str(_MOTION_ROOT / "pour.csv"),
)

# new change
_STAND_COORDINATION_V1_ROOT = _MOTION_ROOT / "stand_coordination_v1"
STAND_COORDINATION_V1_MOTIONS: tuple[str, ...] = tuple(
    str(_STAND_COORDINATION_V1_ROOT / f"{index:02d}_{name}.csv")
    for index, name in enumerate(
        (
            "neutral",
            "bilateral_sagittal",
            "bilateral_lateral",
            "unilateral_counterbalance",
            "axial_twist",
            "roll_coordination",
            "pitch_coordination",
            "combined_coordination",
        )
    )
)


# new change
@configclass
class G1StructuredArmAdapterObservationsCfg(ObsGroup):
    """A compact current-state feedback group; populated per arm below."""

    joint_q: ObsTerm | None = None
    reference_q: ObsTerm | None = None
    q_minus_reference: ObsTerm | None = None
    joint_dq: ObsTerm | None = None
    reference_dq: ObsTerm | None = None
    phase_sincos: ObsTerm | None = None

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = None


# new change
@configclass
class G1OnlineComplianceArmFeedbackObservationsCfg(ObsGroup):
    """Online compliance state with explicit reference velocity.

    Layout is kept fixed for the analytic probe and later supervised teacher:
    q-q_ref(17), dq(17), dq_ref(17), force/state context(20).
    """

    q_minus_reference: ObsTerm | None = None
    joint_dq: ObsTerm | None = None
    reference_dq: ObsTerm | None = None
    context: ObsTerm | None = None

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = None


# new change
@configclass
class G1StateAwareWaistAdapterObservationsCfg(
    G1StructuredArmAdapterObservationsCfg
):
    """Waist feedback plus direct balance state for joint coordination."""

    com_support_offset: ObsTerm | None = None
    torso_projected_gravity: ObsTerm | None = None
    base_ang_vel: ObsTerm | None = None


# new change
@configclass
class G1StationarySupportAdapterObservationsCfg(
    G1StateAwareWaistAdapterObservationsCfg
):
    """Support-chain state omitted from the frozen model_1375 actor."""

    root_xy_position_error: ObsTerm | None = None
    base_lin_vel: ObsTerm | None = None
    foot_velocities: ObsTerm | None = None
    resolved_support_forces: ObsTerm | None = None
    resolved_support_contacts: ObsTerm | None = None


# new change
@configclass
class G1StationarySupportControlV2ObservationsCfg(
    G1StationarySupportAdapterObservationsCfg
):
    """Full support-chain state plus fixed left/right stance-anchor error."""

    fixed_foot_xy_error: ObsTerm | None = None


# new change
@configclass
class G1SupportRecoveryAwareTeacherV3ObservationsCfg(
    G1StationarySupportControlV2ObservationsCfg
):
    """V2 support state plus explicit instantaneous failure-trend features."""

    failure_trend: ObsTerm | None = None


# new change
@configclass
class G1OnlineComplianceObservationsCfg(ObsGroup):
    """Force-aware feedback used only by the online compliance head."""

    q_minus_reference: ObsTerm | None = None
    joint_dq: ObsTerm | None = None
    context: ObsTerm | None = None

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = None


# new change
def _g1_joint_config() -> mdp.ReferenceCommandCfg.JointConfig:
    return mdp.ReferenceCommandCfg.JointConfig(
        num_joints=G1_SOFTMIMIC_NUM_JOINTS,
        thigh_length=0.3,
        calf_length=0.3,
        robot_type="g1",
        left_leg_indices=mdp.ReferenceCommandCfg.JointConfig.LegIndicesConfig(
            hip_yaw=6, hip_roll=3, hip_pitch=0, knee=9, ankle_pitch=13
        ),
        right_leg_indices=mdp.ReferenceCommandCfg.JointConfig.LegIndicesConfig(
            hip_yaw=7, hip_roll=4, hip_pitch=1, knee=10, ankle_pitch=14
        ),
        left_arm_indices=mdp.ReferenceCommandCfg.JointConfig.ArmIndicesConfig(
            shoulder_pitch=11,
            shoulder_roll=15,
            shoulder_yaw=19,
            elbow=21,
            wrist_pitch=25,
            wrist_roll=23,
            wrist_yaw=27,
        ),
        right_arm_indices=mdp.ReferenceCommandCfg.JointConfig.ArmIndicesConfig(
            shoulder_pitch=12,
            shoulder_roll=16,
            shoulder_yaw=20,
            elbow=22,
            wrist_pitch=26,
            wrist_roll=24,
            wrist_yaw=28,
        ),
    )


# new change
@configclass
class G1GeneralTrackingV1EnvCfg(TrackingEnvCfg):
    """One gate-free actor trained on five 100-percent reference motions."""

    def __post_init__(self):
        super().__post_init__()

        # Robot and scene ---------------------------------------------------
        self.scene.robot = G1_SOFTMIMIC_29DOF_COMPAT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.height_scanner = None
        self.scene.num_envs = 4096
        self.scene.env_spacing = 3.0
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)
        }
        self.scene.terrain.terrain_generator.border_width = 50.0
        self.curriculum.terrain_levels = None

        # Reference command ------------------------------------------------
        base_cmd = self.commands.upper_body_joints
        self.commands.upper_body_joints = mdp.ReferenceCommandCfg(
            asset_name="robot",
            resampling_time_range=(20.0, 20.0),
            n_future_steps=10,
            future_dt=1.0 / 50.0,
            proc_ranges=base_cmd.proc_ranges.copy(),
            joint_config=_g1_joint_config(),
            keypoint_body_ids=list(G1_SOFTMIMIC_KEYPOINT_BODY_NAMES),
            demo_playback_mode="references",
            demo_recording_path=list(GENERAL_TRACKING_MOTIONS),
            demo_start_range=[[0.0, 0.0] for _ in GENERAL_TRACKING_MOTIONS],
            demo_stand_height=0.0,
            demo_pitch_angle=0.0,
            demo_random_init=False,
            demo_random_reset=False,
            demo_zero_reset=False,
            demo_reset_robot_on_clip_end=False,
            demo_loader_class="ProceduralMotionLibFromDemo",
            demo_upper_body_only=False,
            demo_zero_xy=False,
            clip_length=20.0,
            debug_vis=False,
        )

        # One policy/action contract for all motions and all body joints. ----
        residual_scale = {
            name: (
                0.30
                if any(token in name for token in ("hip_", "knee", "ankle"))
                else (0.20 if name.startswith("waist_") else 0.15)
            )
            for name in G1_SOFTMIMIC_29_JOINT_NAMES
        }
        self.actions.joint_pos = mdp.UnifiedReferenceResidualJointPositionActionCfg(
            asset_name="robot",
            joint_names=g1_softmimic_joint_names(),
            preserve_order=True,
            scale=residual_scale,
            offset=0.0,
            use_default_offset=False,
            command_name="upper_body_joints",
            reference_joint_ids=list(range(G1_SOFTMIMIC_NUM_JOINTS)),
            reference_time_index=1,
        )

        policy_joint_cfg = SceneEntityCfg(
            "robot", joint_names=g1_softmimic_joint_names(), preserve_order=True
        )
        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.params["asset_cfg"] = policy_joint_cfg.copy()
            group.joint_vel.params["asset_cfg"] = policy_joint_cfg.copy()
            group.reference_dof_pos.params["future_steps"] = [1]
            group.reference_xy_vel.params["future_steps"] = [1]
            group.reference_yaw_vel.params["future_steps"] = [1]
            group.reference_root_height.params["future_steps"] = [1]
            group.reference_gravity_vector.params["future_steps"] = [1]
            group.reference_foot_contact.params["future_steps"] = [1]
            group.reference_dof_pos_future = ObsTerm(
                func=mdp.reference_dof_pos,
                params={"command_name": "upper_body_joints", "future_steps": [2, 5, 10]},
            )
            group.reference_gravity_vector_future = ObsTerm(
                func=mdp.reference_gravity_vector,
                params={"command_name": "upper_body_joints", "future_steps": [2, 5, 10]},
            )
            group.reference_foot_contact_future = ObsTerm(
                func=mdp.reference_foot_contacts,
                params={"command_name": "upper_body_joints", "future_steps": [2, 5, 10]},
            )

        # The deployed policy uses proprioception/IMU plus reference motion;
        # world-frame base velocity and contact-force details remain critic-only.
        self.observations.policy.base_lin_vel = None
        self.observations.policy.root_height = None
        self.observations.policy.foot_contact_forces = None
        self.observations.policy.foot_velocities = None
        self.observations.policy.keypoint_error = None
        self.observations.critic.keypoint_error = ObsTerm(
            func=mdp.keypoint_error_local_shifted,
            params={
                "command_name": "upper_body_joints",
                "keypoint_body_ids": list(G1_SOFTMIMIC_KEYPOINT_BODY_NAMES),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Nominal tracking objective; no force/compliance terms in V1. -------
        self.rewards.joint_deviation_upper_body_commanded = RewTerm(
            func=mdp.joint_subset_deviation_from_command_mse_exp,
            weight=6.0,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": list(range(G1_SOFTMIMIC_NUM_JOINTS)),
                "sigma": 0.12,
                "asset_cfg": policy_joint_cfg.copy(),
            },
        )
        self.rewards.joint_velocity_tracking = RewTerm(
            func=mdp.joint_subset_velocity_deviation_from_command_mse_exp,
            weight=1.5,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": list(range(G1_SOFTMIMIC_NUM_JOINTS)),
                "sigma": 0.60,
                "asset_cfg": policy_joint_cfg.copy(),
            },
        )
        self.rewards.keypoint_deviation_from_command = RewTerm(
            func=mdp.keypoint_deviation_from_command_local_mse_exp,
            weight=4.0,
            params={
                "command_name": "upper_body_joints",
                "keypoint_body_ids": list(G1_SOFTMIMIC_KEYPOINT_BODY_NAMES),
                "sigma": 0.10,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.keypoint_orientation_deviation_from_command = None
        # new change
        # The release's global helpers consume the entire future horizon.  Use
        # the current-frame local variants so reference and robot tensors are
        # both (num_envs, 3).
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_local
        self.rewards.track_lin_vel_xy_exp.params = {
            "command_name": "upper_body_joints",
            "std": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis"]),
        }
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_local
        self.rewards.track_ang_vel_z_exp.params = {
            "command_name": "upper_body_joints",
            "std": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis"]),
        }
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.gravity_exp.weight = 1.5
        self.rewards.joint_deviation_contacts_commanded = RewTerm(
            func=mdp.joint_deviation_from_command_contacts_prob,
            weight=-0.5,
            params={
                "command_name": "upper_body_joints",
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=".*_ankle_roll_link"
                ),
            },
        )
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_vel_l2.weight = -2.0e-4
        self.rewards.ang_vel_xy_l2.weight = -0.10
        self.rewards.lin_vel_z_l2.weight = -0.20
        self.rewards.alive.weight = 0.5

        # Reset near a random 20-second motion segment.  Do not randomize
        # dynamics until nominal multi-motion tracking has converged.
        self.events.reset_robot.params["jpos_range"] = (-0.02, 0.02)
        self.events.reset_robot.params["jvel_range"] = (-0.05, 0.05)
        self.events.reset_robot.params["pose_range"]["roll"] = (-0.03, 0.03)
        self.events.reset_robot.params["pose_range"]["pitch"] = (-0.03, 0.03)
        self.events.add_base_mass = None
        self.events.add_link_mass = None
        self.events.randomize_base_com = None
        self.events.randomize_joint_parameters = None
        self.events.randomize_joint_damping = None

        self.terminations.base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["pelvis", "torso_link"]
                ),
                "threshold": 1.0,
            },
        )
        # new change
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_reference_orientation,
            params={
                "command_name": "upper_body_joints",
                "limit_angle": 0.7853981633974483,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.terminations.bad_keypoints = None

        self.episode_length_s = 20.0
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = [2.5, 2.5, 1.5]
        self.viewer.lookat = [0.0, 0.0, 0.75]


# new change
def _select_motion_curriculum(cfg: G1GeneralTrackingV1EnvCfg, names: tuple[str, ...]) -> None:
    """Change only the dataset while preserving policy/reward tensor shapes."""

    paths = [str(_MOTION_ROOT / name) for name in names]
    cfg.commands.upper_body_joints.demo_recording_path = paths
    cfg.commands.upper_body_joints.demo_start_range = [[0.0, 0.0] for _ in paths]


# new change
@configclass
class G1GeneralTrackingV1StandBootstrapEnvCfg(G1GeneralTrackingV1EnvCfg):
    """Stage A: learn active balance with the final general-policy interface."""

    def __post_init__(self):
        super().__post_init__()
        _select_motion_curriculum(self, ("stand.csv",))


# new change
@configclass
class G1GeneralTrackingV1WalkBootstrapEnvCfg(G1GeneralTrackingV1EnvCfg):
    """Stage B1: learn full-amplitude walking without stand-credit dilution."""

    def __post_init__(self):
        super().__post_init__()
        _select_motion_curriculum(self, ("walk.csv",))


# new change
@configclass
class G1GeneralTrackingV1LocomotionBootstrapEnvCfg(G1GeneralTrackingV1EnvCfg):
    """Stage B2: consolidate stand and walk into the shared actor."""

    def __post_init__(self):
        super().__post_init__()
        _select_motion_curriculum(self, ("stand.csv", "walk.csv"))


# new change
@configclass
class G1GeneralTrackingV1CoreEnvCfg(G1GeneralTrackingV1EnvCfg):
    """Stage B: stand, walk, bilateral arm motion, and Pour at 100 percent."""

    def __post_init__(self):
        super().__post_init__()
        _select_motion_curriculum(
            self, ("stand.csv", "walk.csv", "tpose.csv", "pour.csv")
        )


# new change
@configclass
class G1GeneralTrackingV1StationaryCoreEnvCfg(G1GeneralTrackingV1EnvCfg):
    """Stand-supported whole-body tracking: Stand, T-pose and Pour only.

    The 29DOF policy remains unified.  Explicit waist, shoulder and upper-body
    keypoint terms improve coordination credit without gates, adapters, motion
    scaling or any change to the accepted actuator model.
    """

    def __post_init__(self):
        super().__post_init__()
        _select_motion_curriculum(self, ("stand.csv", "tpose.csv", "pour.csv"))

        waist_names = [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ]
        shoulder_names = [
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_yaw_joint",
        ]
        waist_reference_ids = [2, 5, 8]
        shoulder_reference_ids = [11, 12, 15, 16, 19, 20]

        self.rewards.stationary_waist_tracking = RewTerm(
            func=mdp.joint_subset_deviation_from_command_mse_exp,
            weight=2.0,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": waist_reference_ids,
                "sigma": 0.08,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=waist_names, preserve_order=True
                ),
            },
        )
        self.rewards.stationary_shoulder_tracking = RewTerm(
            func=mdp.joint_subset_deviation_from_command_mse_exp,
            weight=3.0,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": shoulder_reference_ids,
                "sigma": 0.08,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=shoulder_names, preserve_order=True
                ),
            },
        )
        self.rewards.stationary_torso_keypoint_tracking = RewTerm(
            func=mdp.keypoint_subset_deviation_from_command_local_mse_exp,
            weight=2.0,
            params={
                "command_name": "upper_body_joints",
                "keypoint_body_ids": ["torso_link"],
                "reference_keypoint_ids": [7],
                "sigma": 0.06,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.stationary_arm_keypoint_tracking = RewTerm(
            func=mdp.keypoint_subset_deviation_from_command_local_mse_exp,
            weight=4.0,
            params={
                "command_name": "upper_body_joints",
                "keypoint_body_ids": [
                    "left_shoulder_yaw_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_yaw_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ],
                "reference_keypoint_ids": [8, 9, 10, 11, 12, 13],
                "sigma": 0.06,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )


# new change
@configclass
class G1GeneralTrackingV1StationaryArmsCreditIsolatedEnvCfg(
    G1GeneralTrackingV1StationaryCoreEnvCfg
):
    """Isolate bilateral shoulder, elbow and wrist tracking credit."""

    def __post_init__(self):
        super().__post_init__()

        # Replace the two averaged auxiliary terms.  The original full-body
        # joint/keypoint objectives stay active so this remains a general
        # tracking task rather than a collection of hard gates.
        self.rewards.stationary_shoulder_tracking = None
        self.rewards.stationary_arm_keypoint_tracking = None

        joint_groups = (
            (
                "left_shoulder",
                [
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                ],
                [11, 15, 19],
            ),
            (
                "right_shoulder",
                [
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                ],
                [12, 16, 20],
            ),
            ("left_elbow", ["left_elbow_joint"], [21]),
            ("right_elbow", ["right_elbow_joint"], [22]),
            (
                "left_wrist",
                [
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                ],
                [23, 25, 27],
            ),
            (
                "right_wrist",
                [
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
                [24, 26, 28],
            ),
        )
        for group_name, joint_names, reference_ids in joint_groups:
            setattr(
                self.rewards,
                f"stationary_{group_name}_joint_tracking",
                RewTerm(
                    func=mdp.joint_subset_deviation_from_command_mse_exp,
                    weight=0.5,
                    params={
                        "command_name": "upper_body_joints",
                        "reference_joint_ids": reference_ids,
                        "sigma": 0.08,
                        "asset_cfg": SceneEntityCfg(
                            "robot", joint_names=joint_names, preserve_order=True
                        ),
                    },
                ),
            )

        keypoint_groups = (
            ("left_shoulder", "left_shoulder_yaw_link", 8, 0.25, 0.06),
            ("right_shoulder", "right_shoulder_yaw_link", 11, 0.25, 0.06),
            ("left_elbow", "left_elbow_link", 9, 0.75, 0.05),
            ("right_elbow", "right_elbow_link", 12, 0.75, 0.05),
            ("left_wrist", "left_wrist_yaw_link", 10, 1.0, 0.04),
            ("right_wrist", "right_wrist_yaw_link", 13, 1.0, 0.04),
        )
        for group_name, body_name, reference_id, weight, sigma in keypoint_groups:
            setattr(
                self.rewards,
                f"stationary_{group_name}_keypoint_tracking",
                RewTerm(
                    func=mdp.keypoint_subset_deviation_from_command_local_mse_exp,
                    weight=weight,
                    params={
                        "command_name": "upper_body_joints",
                        "keypoint_body_ids": [body_name],
                        "reference_keypoint_ids": [reference_id],
                        "sigma": sigma,
                        "asset_cfg": SceneEntityCfg("robot"),
                    },
                ),
            )


# new change
@configclass
class G1StandCoordinationV1EnvCfg(
    G1GeneralTrackingV1StationaryArmsCreditIsolatedEnvCfg
):
    """Dedicated 100% standing waist--shoulder coordination task.

    All references keep the root, lower-body pose, and bilateral contacts at
    the neutral stand values.  Only the waist and two seven-DOF arm chains are
    excited.  There is no compliance force, gate, adapter, or motion scaling.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.demo_recording_path = list(
            STAND_COORDINATION_V1_MOTIONS
        )
        self.commands.upper_body_joints.demo_start_range = [
            [0.0, 0.0] for _ in STAND_COORDINATION_V1_MOTIONS
        ]
        self.commands.upper_body_joints.demo_playback_mode = "references"
        self.commands.upper_body_joints.demo_upper_body_only = False
        self.commands.upper_body_joints.clip_length = 20.0
        self.commands.upper_body_joints.resampling_time_range = (20.0, 20.0)

        lower_body_names = [
            G1_SOFTMIMIC_29_JOINT_NAMES[index]
            for index in range(G1_SOFTMIMIC_NUM_JOINTS)
            if index not in (2, 5, 8, *G1_SOFTMIMIC_LEFT_ARM_ACTION_IDS, *G1_SOFTMIMIC_RIGHT_ARM_ACTION_IDS)
        ]
        lower_body_reference_ids = [
            index
            for index in range(G1_SOFTMIMIC_NUM_JOINTS)
            if index not in (2, 5, 8, *G1_SOFTMIMIC_LEFT_ARM_ACTION_IDS, *G1_SOFTMIMIC_RIGHT_ARM_ACTION_IDS)
        ]
        self.rewards.stand_coordination_lower_body_tracking = RewTerm(
            func=mdp.joint_subset_deviation_from_command_mse_exp,
            weight=2.0,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": lower_body_reference_ids,
                "sigma": 0.06,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=lower_body_names, preserve_order=True
                ),
            },
        )
        self.rewards.stand_coordination_waist_velocity_tracking = RewTerm(
            func=mdp.joint_subset_velocity_deviation_from_command_mse_exp,
            weight=0.75,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": [2, 5, 8],
                "sigma": 0.35,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "waist_yaw_joint",
                        "waist_roll_joint",
                        "waist_pitch_joint",
                    ],
                    preserve_order=True,
                ),
            },
        )
        bilateral_arm_ids = list(
            G1_SOFTMIMIC_LEFT_ARM_ACTION_IDS + G1_SOFTMIMIC_RIGHT_ARM_ACTION_IDS
        )
        self.rewards.stand_coordination_arm_velocity_tracking = RewTerm(
            func=mdp.joint_subset_velocity_deviation_from_command_mse_exp,
            weight=1.0,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": bilateral_arm_ids,
                "sigma": 0.50,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[G1_SOFTMIMIC_29_JOINT_NAMES[index] for index in bilateral_arm_ids],
                    preserve_order=True,
                ),
            },
        )
        self.rewards.gravity_exp.weight = 2.5
        self.rewards.feet_stumble.weight = -0.01
        self.episode_length_s = 20.0


# new change
@configclass
class G1StandCoordinationAccuracyV1EnvCfg(G1StandCoordinationV1EnvCfg):
    """Short, credit-isolated accuracy refinement for accepted model_1375.

    The reference set, mapping, actuators, lower-body objectives, stability
    rewards, and termination contract are inherited unchanged.  Only reward
    credit for the three waist axes and six wrist axes is made explicit.  The
    paired runner freezes the actor trunk and every non-waist/non-wrist output
    row, so this task cannot silently replace the accepted balance policy.
    """

    def __post_init__(self):
        super().__post_init__()

        # Replace averaged waist credit with one term per physical axis.  The
        # diagnosis found small global MAE but phase-local waist-pitch peaks of
        # roughly two degrees in Sagittal and Pitch Coordination.
        self.rewards.stationary_waist_tracking = None
        waist_terms = (
            ("yaw", "waist_yaw_joint", 2),
            ("roll", "waist_roll_joint", 5),
            ("pitch", "waist_pitch_joint", 8),
        )
        for axis, joint_name, reference_id in waist_terms:
            setattr(
                self.rewards,
                f"accuracy_waist_{axis}_tracking",
                RewTerm(
                    func=mdp.joint_subset_deviation_from_command_mse_exp,
                    weight=0.75,
                    params={
                        "command_name": "upper_body_joints",
                        "reference_joint_ids": [reference_id],
                        "sigma": 0.04,
                        "asset_cfg": SceneEntityCfg(
                            "robot", joint_names=[joint_name], preserve_order=True
                        ),
                    },
                ),
            )

        # Replace left/right three-axis averages with explicit per-axis terms.
        # The right wrist receives more credit because its 2.3--4.8 degree
        # signed error is fixed-bias dominated across all eight motions.
        self.rewards.stationary_left_wrist_joint_tracking = None
        self.rewards.stationary_right_wrist_joint_tracking = None
        wrist_terms = (
            ("left", "roll", "left_wrist_roll_joint", 23, 0.35),
            ("right", "roll", "right_wrist_roll_joint", 24, 0.75),
            ("left", "pitch", "left_wrist_pitch_joint", 25, 0.35),
            ("right", "pitch", "right_wrist_pitch_joint", 26, 0.75),
            ("left", "yaw", "left_wrist_yaw_joint", 27, 0.35),
            ("right", "yaw", "right_wrist_yaw_joint", 28, 0.75),
        )
        for side, axis, joint_name, reference_id, weight in wrist_terms:
            setattr(
                self.rewards,
                f"accuracy_{side}_wrist_{axis}_tracking",
                RewTerm(
                    func=mdp.joint_subset_deviation_from_command_mse_exp,
                    weight=weight,
                    params={
                        "command_name": "upper_body_joints",
                        "reference_joint_ids": [reference_id],
                        "sigma": 0.04,
                        "asset_cfg": SceneEntityCfg(
                            "robot", joint_names=[joint_name], preserve_order=True
                        ),
                    },
                ),
            )

        # Spatial wrist credit is tightened without changing shoulder/elbow
        # rewards.  Sagittal was the only motion above 15 mm at both wrists.
        self.rewards.stationary_left_wrist_keypoint_tracking.weight = 1.5
        self.rewards.stationary_left_wrist_keypoint_tracking.params["sigma"] = 0.035
        self.rewards.stationary_right_wrist_keypoint_tracking.weight = 2.5
        self.rewards.stationary_right_wrist_keypoint_tracking.params["sigma"] = 0.035

        # Most right-wrist error is bias, but combined coordination also showed
        # a reproducible roll lag.  This grouped velocity term addresses that
        # secondary effect without diluting it across fourteen arm joints.
        self.rewards.accuracy_right_wrist_velocity_tracking = RewTerm(
            func=mdp.joint_subset_velocity_deviation_from_command_mse_exp,
            weight=0.5,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": [24, 26, 28],
                "sigma": 0.35,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "right_wrist_roll_joint",
                        "right_wrist_pitch_joint",
                        "right_wrist_yaw_joint",
                    ],
                    preserve_order=True,
                ),
            },
        )


# new change
@configclass
class G1StandCoordinationAccuracyV2EnvCfg(G1StandCoordinationAccuracyV1EnvCfg):
    """Structured bilateral-arm accuracy stage based on accepted model_1375.

    This keeps the 100% StandCoordination references, actuators, waist/lower
    body policy, stability rewards, and terminations unchanged.  It adds two
    side-specific 37D feedback groups and isolates shoulder/elbow credit down
    to each physical joint.  The paired runner freezes the complete legacy
    actor and trains only the two new seven-DOF adapters.
    """

    def __post_init__(self):
        super().__post_init__()

        # The V1 wrist axes are already credit-isolated.  Replace only the
        # grouped shoulder/elbow terms so upstream wrist-position errors are
        # assigned to the joint that physically causes them.
        self.rewards.stationary_left_shoulder_joint_tracking = None
        self.rewards.stationary_right_shoulder_joint_tracking = None
        self.rewards.stationary_left_elbow_joint_tracking = None
        self.rewards.stationary_right_elbow_joint_tracking = None
        proximal_terms = (
            ("left_shoulder_pitch", "left_shoulder_pitch_joint", 11),
            ("right_shoulder_pitch", "right_shoulder_pitch_joint", 12),
            ("left_shoulder_roll", "left_shoulder_roll_joint", 15),
            ("right_shoulder_roll", "right_shoulder_roll_joint", 16),
            ("left_shoulder_yaw", "left_shoulder_yaw_joint", 19),
            ("right_shoulder_yaw", "right_shoulder_yaw_joint", 20),
            ("left_elbow", "left_elbow_joint", 21),
            ("right_elbow", "right_elbow_joint", 22),
        )
        for term_name, joint_name, reference_id in proximal_terms:
            setattr(
                self.rewards,
                f"accuracy_{term_name}_tracking",
                RewTerm(
                    func=mdp.joint_subset_deviation_from_command_mse_exp,
                    weight=0.5,
                    params={
                        "command_name": "upper_body_joints",
                        "reference_joint_ids": [reference_id],
                        "sigma": 0.04,
                        "asset_cfg": SceneEntityCfg(
                            "robot", joint_names=[joint_name], preserve_order=True
                        ),
                    },
                ),
            )

        def add_adapter_group(side: str, reference_ids: tuple[int, ...]) -> None:
            joint_names = [G1_SOFTMIMIC_29_JOINT_NAMES[index] for index in reference_ids]
            asset_cfg = SceneEntityCfg(
                "robot", joint_names=joint_names, preserve_order=True
            )
            group = G1StructuredArmAdapterObservationsCfg()
            group.joint_q = ObsTerm(
                func=mdp.joint_pos, params={"asset_cfg": asset_cfg.copy()}
            )
            group.reference_q = ObsTerm(
                func=mdp.reference_joint_pos_subset,
                params={
                    "command_name": "upper_body_joints",
                    "reference_joint_ids": list(reference_ids),
                },
            )
            group.q_minus_reference = ObsTerm(
                func=mdp.joint_position_error_to_reference_subset,
                params={
                    "command_name": "upper_body_joints",
                    "reference_joint_ids": list(reference_ids),
                    "asset_cfg": asset_cfg.copy(),
                },
            )
            group.joint_dq = ObsTerm(
                func=mdp.joint_vel, params={"asset_cfg": asset_cfg.copy()}
            )
            group.reference_dq = ObsTerm(
                func=mdp.reference_joint_vel_subset,
                params={
                    "command_name": "upper_body_joints",
                    "reference_joint_ids": list(reference_ids),
                },
            )
            group.phase_sincos = ObsTerm(
                func=mdp.reference_motion_phase_sincos,
                params={"command_name": "upper_body_joints"},
            )
            setattr(self.observations, f"{side}_arm_adapter", group)

        add_adapter_group("left", G1_SOFTMIMIC_LEFT_ARM_ACTION_IDS)
        add_adapter_group("right", G1_SOFTMIMIC_RIGHT_ARM_ACTION_IDS)


# new change
@configclass
class G1StandCoordinationLowDisturbanceDataEnvCfg(
    G1StandCoordinationAccuracyV2EnvCfg
):
    """Inference-only environment for labelled low-disturbance rollouts.

    This class deliberately changes no reference, reward, actuator,
    observation, termination, or policy setting from the accepted AccuracyV2
    environment.  External wrenches are injected only by the data-generation
    script so a zero-wrench run remains strictly equivalent to model_1375.
    Keeping a separate task id prevents later compliance experiments from
    silently modifying the accepted nominal-control task.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0


# new change
@configclass
class G1StandCoordinationOnlineCompliantEnvCfg(
    G1StandCoordinationAccuracyV2EnvCfg
):
    """True on-policy compliant Stand task; no augmented CSV or saved dataset.

    Each environment samples ``body/F/K`` on the GPU during its rollout.  The
    command produces ``x_ref + F/K``, the event applies that exact force, and
    PPO receives the transition immediately through the force-aware adapter.
    """

    def __post_init__(self):
        super().__post_init__()

        nominal = self.commands.upper_body_joints
        self.commands.upper_body_joints = mdp.OnlineComplianceReferenceCommandCfg(
            asset_name=nominal.asset_name,
            resampling_time_range=nominal.resampling_time_range,
            n_future_steps=nominal.n_future_steps,
            future_dt=nominal.future_dt,
            proc_ranges=nominal.proc_ranges.copy(),
            joint_config=nominal.joint_config.copy(),
            keypoint_body_ids=list(nominal.keypoint_body_ids),
            demo_playback_mode=nominal.demo_playback_mode,
            demo_recording_path=list(nominal.demo_recording_path),
            demo_start_range=list(nominal.demo_start_range),
            demo_stand_height=nominal.demo_stand_height,
            demo_pitch_angle=nominal.demo_pitch_angle,
            demo_random_init=nominal.demo_random_init,
            demo_random_reset=nominal.demo_random_reset,
            demo_zero_reset=nominal.demo_zero_reset,
            demo_reset_robot_on_clip_end=nominal.demo_reset_robot_on_clip_end,
            demo_loader_class=nominal.demo_loader_class,
            demo_upper_body_only=nominal.demo_upper_body_only,
            demo_zero_xy=nominal.demo_zero_xy,
            clip_length=nominal.clip_length,
            debug_vis=False,
            force_body_names=(
                "torso_link",
                "left_elbow_link",
                "left_wrist_yaw_link",
                "right_elbow_link",
                "right_wrist_yaw_link",
            ),
            # new change
            # First online curriculum: the 8 N ceiling caused two incremental
            # failures in a 32x500 A/B gate, so begin at 6 N and expand only
            # after the force-aware adapter has learned the low-force regime.
            force_range_n=(2.0, 6.0),
            stiffness_range_n_per_m=(250.0, 600.0),
            max_displacement_m=0.025,
            vertical_direction_scale=0.25,
            warmup_time_s=1.0,
            active_time_s=0.8,
            recovery_time_s=1.2,
        )

        self.events.apply_online_compliance_force = EventTerm(
            func=mdp.apply_forcefield_force_torque,
            mode="interval",
            interval_range_s=(0.005, 0.005),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "upper_body_joints",
            },
        )

        upper_ids = tuple(
            sorted(
                (2, 5, 8)
                + G1_SOFTMIMIC_LEFT_ARM_ACTION_IDS
                + G1_SOFTMIMIC_RIGHT_ARM_ACTION_IDS
            )
        )
        upper_names = [G1_SOFTMIMIC_29_JOINT_NAMES[index] for index in upper_ids]
        upper_asset_cfg = SceneEntityCfg(
            "robot", joint_names=upper_names, preserve_order=True
        )
        compliance_group = G1OnlineComplianceObservationsCfg()
        compliance_group.q_minus_reference = ObsTerm(
            func=mdp.joint_position_error_to_reference_subset,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": list(upper_ids),
                "asset_cfg": upper_asset_cfg.copy(),
            },
        )
        compliance_group.joint_dq = ObsTerm(
            func=mdp.joint_vel, params={"asset_cfg": upper_asset_cfg.copy()}
        )
        compliance_group.context = ObsTerm(
            func=mdp.online_compliance_context,
            params={
                "command_name": "upper_body_joints",
                "force_scale_n": 8.0,
                "displacement_scale_m": 0.025,
                "stiffness_scale_n_per_m": 600.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.observations.online_compliance = compliance_group

        # The value function must know the sampled force before the robot has
        # visibly moved; otherwise the disturbance is partially observable to
        # PPO even though the actor is force-aware.  This term intentionally
        # starts a fresh critic when bootstrapping from model_1375.
        self.observations.critic.online_compliance_context = ObsTerm(
            func=mdp.online_compliance_context,
            params={
                "command_name": "upper_body_joints",
                "force_scale_n": 8.0,
                "displacement_scale_m": 0.025,
                "stiffness_scale_n_per_m": 600.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # new change
        # The nominal reference remains visible for bootstrapping model_1375,
        # while the fresh critic additionally receives the IK-adapted current
        # and future 29DOF targets.  Thus value learning is not forced to infer
        # the generated q_ref solely from F/K.
        self.observations.critic.adapted_reference_dof_pos = ObsTerm(
            func=mdp.adapted_reference_dof_pos,
            params={
                "command_name": "upper_body_joints",
                "future_steps": [1],
            },
        )
        self.observations.critic.adapted_reference_dof_pos_future = ObsTerm(
            func=mdp.adapted_reference_dof_pos,
            params={
                "command_name": "upper_body_joints",
                "future_steps": [2, 5, 10],
            },
        )

        self.rewards.online_compliant_link_tracking = RewTerm(
            func=mdp.online_compliant_keypoint_tracking_exp,
            weight=4.0,
            params={
                "command_name": "upper_body_joints",
                "sigma": 0.04,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        # Nominal COM, torso, foot-slip and termination constraints inherited
        # from AccuracyV2 remain active and unchanged.
        self.episode_length_s = 20.0


# new change
@configclass
class G1StandCoordinationOnlineCompliantZeroForceEnvCfg(
    G1StandCoordinationOnlineCompliantEnvCfg
):
    """Strict zero-wrench equivalence gate for the online reference chain."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.force_range_n = (0.0, 0.0)


# new change
@configclass
class G1StandCoordinationOnlineCompliantVisualEnvCfg(
    G1StandCoordinationOnlineCompliantEnvCfg
):
    """Single-environment visual data-quality gate; never used for training."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.debug_vis = True
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0
        self.episode_length_s = 40.0
        # new change
        # Closer three-quarter view so force arrows and 14 keypoint markers are
        # readable in a 720p recording.
        self.viewer.eye = (2.05, 2.05, 1.55)
        self.viewer.lookat = (0.0, 0.0, 0.82)


# new change
@configclass
class G1StandCoordinationOnlineArmLocalIKV2EnvCfg(
    G1StandCoordinationOnlineCompliantEnvCfg
):
    """Wrist-only local-arm IK matching the released offline augmentation."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.upper_body_joints
        # The released SoftMimic constants expose only the two wrist links.
        command.force_body_names = (
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        )
        command.force_body_context_size = 5
        command.force_body_context_slots = (2, 4)
        # Match the visible displacement scale of released Pour samples while
        # retaining the conservative 2--6 N first curriculum.
        command.force_range_n = (2.0, 6.0)
        command.stiffness_range_n_per_m = (100.0, 300.0)
        command.max_displacement_m = 0.040
        command.arm_delta_limit_rad = 0.35
        command.waist_delta_limit_rad = 0.0
        command.hip_delta_limit_rad = 0.0
        command.knee_delta_limit_rad = 0.0
        command.ankle_delta_limit_rad = 0.0
        command.ik_damping = 0.015
        command.posture_weight = 1.0e-4
        command.arm_local_ik_enabled = True
        command.cop_aware_com_target_enabled = False
        command.exclude_active_link_from_keypoint_task = True
        # new change
        # Evaluate FK/Jacobians at q_ref with the exact MJCF used by the
        # released offline augmentation.  Never use the live robot pose as the
        # reference-IK linearization point.
        command.reference_kinematics_backend = "pytorch_mjcf"
        command.reference_mjcf_path = (
            "softmimic_deploy/src/assets/g1/g1_29dof_fixed_base.xml"
        )
        command.reference_ik_iterations = 8
        command.max_active_link_residual_m = 0.003
        command.max_active_link_relative_residual = 0.20
        command.min_active_direction_cosine = 0.95
        command.max_foot_motion_m = 1.0e-5
        command.max_com_residual_m = 0.010


# new change
@configclass
class G1StandCoordinationCompliantStandLowForceArmsV1EnvCfg(
    G1StandCoordinationOnlineArmLocalIKV2EnvCfg
):
    """First trainable compliance stage with protected nominal support.

    The accepted model_1375 lower body, waist and nominal actor stay frozen.
    Online IK changes only the selected wrist/arm reference and the matching
    physical 4--7 N wrench is applied to that wrist.  Dedicated bounded
    support terms make whole-body translation an explicit failure mode for the
    two trainable arm compliance heads.
    """

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.upper_body_joints
        command.difficulty_label = "COMPLIANT STAND V1 LOW 4-7N ARM-ONLY"
        command.force_range_n = (4.0, 7.0)
        command.stiffness_range_n_per_m = (80.0, 300.0)
        command.max_displacement_m = 0.140
        command.arm_delta_limit_rad = 0.70
        command.ramp_time_range_s = (0.4, 0.9)
        command.max_active_link_residual_m = 0.005
        command.physical_force_enabled = True
        command.debug_vis = False
        # new change
        # Match the validated FrozenController closed loop: the buffered wrist
        # wrench is applied during every PhysX sub-step, not only once per
        # control step.  This removes a train/test force-integration mismatch.
        self.sim.physx.enable_external_forces_every_iteration = True

        fixed_foot_cfg = SceneEntityCfg(
            "robot",
            body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        # Keep these terms separate from tracking rewards so the compliance
        # head receives direct credit for yielding locally instead of moving
        # the complete robot.  All penalties are bounded for long rollouts.
        self.rewards.compliance_support_root_xy_position = RewTerm(
            func=mdp.stationary_root_xy_position_error_bounded_huber,
            weight=-1.0,
            params={
                "command_name": "upper_body_joints",
                "asset_cfg": SceneEntityCfg("robot"),
                "delta_m": 0.10,
            },
        )
        self.rewards.compliance_support_root_xy_velocity = RewTerm(
            func=mdp.stationary_root_xy_velocity_bounded_huber,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "delta_mps": 0.15},
        )
        self.rewards.compliance_support_fixed_foot_xy = RewTerm(
            func=mdp.fixed_foot_xy_position_error_bounded_huber,
            weight=-2.0,
            params={
                "command_name": "upper_body_joints",
                "reference_keypoint_ids": [3, 6],
                "asset_cfg": fixed_foot_cfg.copy(),
                "delta_m": 0.10,
            },
        )
        self.rewards.compliance_support_com = RewTerm(
            func=mdp.whole_body_com_support_center_exp,
            weight=2.0,
            params={
                "sigma": 0.04,
                "asset_cfg": SceneEntityCfg("robot"),
                "support_foot_cfg": fixed_foot_cfg.copy(),
            },
        )
        self.rewards.compliance_support_torso_orientation = RewTerm(
            func=mdp.target_orientation_exp,
            weight=2.0,
            params={
                "command_name": "upper_body_joints",
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["torso_link"], preserve_order=True
                ),
            },
        )
        self.rewards.compliance_support_failure = RewTerm(
            func=mdp.is_terminated,
            weight=-50.0,
        )
        self.episode_length_s = 20.0


# new change
@configclass
class G1StandCoordinationCompliantStandLowForceArmsV1ZeroForceEnvCfg(
    G1StandCoordinationCompliantStandLowForceArmsV1EnvCfg
):
    """F=0/q_delta=0 equivalence task for the arm-only compliance wrapper."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.upper_body_joints
        command.difficulty_label = "COMPLIANT STAND V1 ZERO FORCE EQUIVALENCE"
        command.force_range_n = (0.0, 0.0)
        command.physical_force_enabled = False
        self.events.apply_online_compliance_force = None


# new change
def _configure_fixed_friction_ab(
    env_cfg,
    *,
    static_friction: float,
    dynamic_friction: float,
    label: str,
) -> None:
    """Fix robot/ground friction for an inference-only physics A/B.

    The base task's startup material event overwrites robot USD materials, so
    changing the USD alone would not produce a controlled experiment.  Keep
    the ground at 1.0/1.0 with multiply combine and change only the fixed robot
    coefficients; the resulting effective pair is therefore exactly the pair
    encoded in the task name.
    """

    env_cfg.scene.terrain.physics_material.friction_combine_mode = "multiply"
    env_cfg.scene.terrain.physics_material.static_friction = 1.0
    env_cfg.scene.terrain.physics_material.dynamic_friction = 1.0
    env_cfg.sim.physics_material = env_cfg.scene.terrain.physics_material
    material_params = env_cfg.events.physics_material.params
    material_params["static_friction_range"] = (
        static_friction,
        static_friction,
    )
    material_params["dynamic_friction_range"] = (
        dynamic_friction,
        dynamic_friction,
    )
    material_params["restitution_range"] = (0.0, 0.0)
    env_cfg.commands.upper_body_joints.difficulty_label = (
        f"{env_cfg.commands.upper_body_joints.difficulty_label} | {label}"
    )


# new change
@configclass
class G1StandCoordinationFrictionABLowForceBaselineEnvCfg(
    G1StandCoordinationCompliantStandLowForceArmsV1EnvCfg
):
    """model_1394 ForceGate low-force baseline: effective 0.8/0.6."""

    def __post_init__(self):
        super().__post_init__()
        _configure_fixed_friction_ab(
            self,
            static_friction=0.8,
            dynamic_friction=0.6,
            label="FRICTION AB BASELINE STATIC0.8 DYNAMIC0.6",
        )


# new change
@configclass
class G1StandCoordinationFrictionABLowForceMediumEnvCfg(
    G1StandCoordinationCompliantStandLowForceArmsV1EnvCfg
):
    """model_1394 ForceGate low-force medium probe: effective 1.1/0.9."""

    def __post_init__(self):
        super().__post_init__()
        _configure_fixed_friction_ab(
            self,
            static_friction=1.1,
            dynamic_friction=0.9,
            label="FRICTION AB MEDIUM STATIC1.1 DYNAMIC0.9",
        )


# new change
@configclass
class G1StandCoordinationFrictionABLowForceHighEnvCfg(
    G1StandCoordinationCompliantStandLowForceArmsV1EnvCfg
):
    """model_1394 ForceGate low-force high probe: effective 1.4/1.2."""

    def __post_init__(self):
        super().__post_init__()
        _configure_fixed_friction_ab(
            self,
            static_friction=1.4,
            dynamic_friction=1.2,
            label="FRICTION AB HIGH STATIC1.4 DYNAMIC1.2",
        )


# new change
@configclass
class G1StandCoordinationForceEventABLowForce2to4NEnvCfg(
    G1StandCoordinationFrictionABLowForceBaselineEnvCfg
):
    """Inference-only 2--4 N event probe at accepted 0.8/0.6 friction."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.upper_body_joints
        command.force_range_n = (2.0, 4.0)
        command.difficulty_label = (
            "FORCE EVENT AB LOW 2-4N | FRICTION STATIC0.8 DYNAMIC0.6"
        )


# new change
@configclass
class G1StandCoordinationForceEventABSmooth2to4NEnvCfg(
    G1StandCoordinationForceEventABLowForce2to4NEnvCfg
):
    """Half-cosine 2--4 N event probe; all non-envelope settings unchanged."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.upper_body_joints
        command.force_envelope_profile = "half_cosine"
        command.difficulty_label = (
            "FORCE EVENT AB HALF-COSINE 2-4N | FRICTION STATIC0.8 DYNAMIC0.6"
        )


# new change
@configclass
class G1StandCoordinationForceEventABSmooth4to7NEnvCfg(
    G1StandCoordinationFrictionABLowForceBaselineEnvCfg
):
    """Half-cosine 4--7 N event probe; all non-envelope settings unchanged."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.upper_body_joints
        command.force_envelope_profile = "half_cosine"
        command.difficulty_label = (
            "FORCE EVENT AB HALF-COSINE 4-7N | FRICTION STATIC0.8 DYNAMIC0.6"
        )


# new change
@configclass
class G1StandCoordinationForceEventABSlopeMatchedSmooth2to4NEnvCfg(
    G1StandCoordinationForceEventABSmooth2to4NEnvCfg
):
    """Half-cosine 2--4 N probe with peak slope matched to linear ramps.

    A half-cosine's peak normalized slope is pi/2.  Stretching the ramp by
    pi/2 therefore preserves the original linear task's peak force slew rate
    while retaining zero slope at onset and release boundaries.
    """

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.upper_body_joints
        slope_match = 1.5707963267948966
        command.ramp_time_range_s = tuple(
            value * slope_match for value in command.ramp_time_range_s
        )
        command.difficulty_label = (
            "FORCE EVENT AB SLOPE-MATCHED HALF-COSINE 2-4N | "
            "FRICTION STATIC0.8 DYNAMIC0.6"
        )


# new change
@configclass
class G1StandCoordinationCompliantStandSmooth2to4NArmsV2EnvCfg(
    G1StandCoordinationForceEventABSmooth2to4NEnvCfg
):
    """Accepted 2--4 N half-cosine curriculum for bilateral arm adapters.

    This intentionally inherits the exact fixed 0.8/0.6 physics and event
    settings used by the three-seed inference acceptance test.  Friction
    randomization belongs to the later robustness stage, after the arm-only
    response has learned without changing the nominal support controller.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.difficulty_label = (
            "COMPLIANT STAND V2 SMOOTH 2-4N BILATERAL ARM ADAPTER TRAINING"
        )


# new change
@configclass
class G1StandCoordinationCompliantStandSmooth2to4NAnalyticArmFeedbackEnvCfg(
    G1StandCoordinationCompliantStandSmooth2to4NArmsV2EnvCfg
):
    """Inference probe exposing q/dq/dq-ref plus force context to both arms."""

    def __post_init__(self):
        super().__post_init__()
        upper_ids = tuple(
            sorted(
                (2, 5, 8)
                + G1_SOFTMIMIC_LEFT_ARM_ACTION_IDS
                + G1_SOFTMIMIC_RIGHT_ARM_ACTION_IDS
            )
        )
        upper_names = [G1_SOFTMIMIC_29_JOINT_NAMES[index] for index in upper_ids]
        upper_asset_cfg = SceneEntityCfg(
            "robot", joint_names=upper_names, preserve_order=True
        )
        group = G1OnlineComplianceArmFeedbackObservationsCfg()
        group.q_minus_reference = ObsTerm(
            func=mdp.joint_position_error_to_reference_subset,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": list(upper_ids),
                "asset_cfg": upper_asset_cfg.copy(),
            },
        )
        group.joint_dq = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": upper_asset_cfg.copy()},
        )
        group.reference_dq = ObsTerm(
            func=mdp.reference_joint_vel_subset,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": list(upper_ids),
            },
        )
        group.context = ObsTerm(
            func=mdp.online_compliance_context,
            params={
                "command_name": "upper_body_joints",
                "force_scale_n": 8.0,
                "displacement_scale_m": 0.025,
                "stiffness_scale_n_per_m": 600.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.observations.online_compliance = group
        self.commands.upper_body_joints.difficulty_label = (
            "COMPLIANT STAND V2 ANALYTIC BILATERAL ARM FEEDBACK PROBE"
        )


# new change
@configclass
class G1StandCoordinationCompliantStandSmooth4to7NArmFeedbackStressEnvCfg(
    G1StandCoordinationCompliantStandSmooth2to4NAnalyticArmFeedbackEnvCfg
):
    """Unseen-event medium-force stress test; no controller setting changes."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.upper_body_joints
        command.force_range_n = (4.0, 7.0)
        command.difficulty_label = (
            "COMPLIANT STAND SUPERVISED ARM FEEDBACK STRESS 4-7N | "
            "HALF-COSINE | FRICTION STATIC0.8 DYNAMIC0.6"
        )


# new change
@configclass
class G1StandCoordinationCompliantStandSmooth4to7NArmFeedbackStressVisualEnvCfg(
    G1StandCoordinationCompliantStandSmooth4to7NArmFeedbackStressEnvCfg
):
    """Single-env annotated view of force, IK keypoints and recovery."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.debug_vis = True
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0
        self.episode_length_s = 40.0
        self.viewer.eye = (2.05, 2.05, 1.55)
        self.viewer.lookat = (0.0, 0.0, 0.82)


# new change
@configclass
class G1StandCoordinationCompliantStandSmooth4to7NArmFeedbackFriction10x08EnvCfg(
    G1StandCoordinationCompliantStandSmooth4to7NArmFeedbackStressEnvCfg
):
    """Current supervised-arm controller at fixed effective 1.0/0.8 friction."""

    def __post_init__(self):
        super().__post_init__()
        _configure_fixed_friction_ab(
            self,
            static_friction=1.0,
            dynamic_friction=0.8,
            label="CURRENT STRESS FRICTION STATIC1.0 DYNAMIC0.8",
        )


# new change
@configclass
class G1StandCoordinationCompliantStandSmooth4to7NArmFeedbackFriction12x10EnvCfg(
    G1StandCoordinationCompliantStandSmooth4to7NArmFeedbackStressEnvCfg
):
    """Current supervised-arm controller at fixed effective 1.2/1.0 friction."""

    def __post_init__(self):
        super().__post_init__()
        _configure_fixed_friction_ab(
            self,
            static_friction=1.2,
            dynamic_friction=1.0,
            label="CURRENT STRESS FRICTION STATIC1.2 DYNAMIC1.0",
        )


# new change
@configclass
class G1StandCoordinationCompliantStandSmooth4to7NRecoveryFDEnvCfg(
    G1StandCoordinationCompliantStandSmooth4to7NArmFeedbackStressEnvCfg
):
    """Strict-pair recovery controllability task around the accepted arm controller.

    Physics, 4--7 N force events, mapping and 0.8/0.6 friction are unchanged.
    The only addition is the previously audited 87D measured support group so
    baseline and small waist/support action branches can be scored from the
    same simulator snapshot.  No observation is added to the frozen actor.
    """

    def __post_init__(self):
        super().__post_init__()
        support_source = G1StandCoordinationSupportRecoveryAwareTeacherV3EnvCfg()
        self.observations.support_adapter = copy.deepcopy(
            support_source.observations.support_adapter
        )
        _configure_fixed_friction_ab(
            self,
            static_friction=0.8,
            dynamic_friction=0.6,
            label="RECOVERY FD STRICT PAIR | FRICTION STATIC0.8 DYNAMIC0.6",
        )
        self.commands.upper_body_joints.debug_vis = False
        self.episode_length_s = 40.0


# new change
@configclass
class G1StandCoordinationOnlineArmLocalIKV2BaseActorEnvCfg(
    G1StandCoordinationOnlineArmLocalIKV2EnvCfg
):
    """Inference gate using the exact accepted AccuracyV2 actor/critic shape."""

    def __post_init__(self):
        super().__post_init__()
        # These terms exist only for the future compliance critic.  Removing
        # them lets model_1375 load with its original runner and avoids testing
        # a newly wrapped actor before strict zero-force equivalence is proven.
        self.observations.critic.online_compliance_context = None
        self.observations.critic.adapted_reference_dof_pos = None
        self.observations.critic.adapted_reference_dof_pos_future = None


# new change
@configclass
class G1StandCoordinationOnlineArmLocalIKV2ZeroForceBaseActorEnvCfg(
    G1StandCoordinationOnlineArmLocalIKV2BaseActorEnvCfg
):
    """F=0/q_delta=0 equivalence gate for the unwrapped model_1375 actor."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.force_range_n = (0.0, 0.0)
        self.commands.upper_body_joints.physical_force_enabled = False
        self.events.apply_online_compliance_force = None


# new change
@configclass
class G1StandCoordinationFrictionABZeroForceBaselineEnvCfg(
    G1StandCoordinationOnlineArmLocalIKV2ZeroForceBaseActorEnvCfg
):
    """model_1375 zero-force baseline: effective friction 0.8/0.6."""

    def __post_init__(self):
        super().__post_init__()
        _configure_fixed_friction_ab(
            self,
            static_friction=0.8,
            dynamic_friction=0.6,
            label="FRICTION AB BASELINE STATIC0.8 DYNAMIC0.6",
        )


# new change
@configclass
class G1StandCoordinationFrictionABZeroForceMediumEnvCfg(
    G1StandCoordinationOnlineArmLocalIKV2ZeroForceBaseActorEnvCfg
):
    """model_1375 zero-force medium-friction probe: effective 1.1/0.9."""

    def __post_init__(self):
        super().__post_init__()
        _configure_fixed_friction_ab(
            self,
            static_friction=1.1,
            dynamic_friction=0.9,
            label="FRICTION AB MEDIUM STATIC1.1 DYNAMIC0.9",
        )


# new change
@configclass
class G1StandCoordinationFrictionABZeroForceHighEnvCfg(
    G1StandCoordinationOnlineArmLocalIKV2ZeroForceBaseActorEnvCfg
):
    """model_1375 zero-force high-friction probe: effective 1.4/1.2."""

    def __post_init__(self):
        super().__post_init__()
        _configure_fixed_friction_ab(
            self,
            static_friction=1.4,
            dynamic_friction=1.2,
            label="FRICTION AB HIGH STATIC1.4 DYNAMIC1.2",
        )


# new change
@configclass
class G1StandCoordinationOnlineArmLocalIKV2ReferenceVisualEnvCfg(
    G1StandCoordinationOnlineArmLocalIKV2EnvCfg
):
    """Generator-only visual gate: draw F/K but apply no physical wrench."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.debug_vis = True
        self.commands.upper_body_joints.physical_force_enabled = False
        self.events.apply_online_compliance_force = None
        # new change
        # Match the offline Mink generator viewer: this task validates the
        # kinematic q_ref itself, not balance or learned actor behavior.
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        self.events.write_online_reference_kinematic_state = EventTerm(
            func=mdp.write_online_reference_kinematic_state,
            mode="interval",
            interval_range_s=(0.005, 0.005),
            params={
                "command_name": "upper_body_joints",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0
        self.episode_length_s = 40.0
        # new change
        # Tight three-quarter shot for judging local elbow/wrist motion rather
        # than only whole-body drift.
        self.viewer.eye = (2.65, 2.65, 1.65)
        self.viewer.lookat = (0.0, 0.0, 0.55)


# new change
@configclass
class G1StandCoordinationOnlineArmLocalIKV2ReferenceBatchEnvCfg(
    G1StandCoordinationOnlineArmLocalIKV2ReferenceVisualEnvCfg
):
    """Marker-free batched q_ref/FK acceptance gate; never used for training."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.debug_vis = False
        self.scene.num_envs = 512
        self.scene.env_spacing = 2.5


# new change
@configclass
class G1StandCoordinationOnlineArmLocalIKV2ReleaseAmplitudeVisualEnvCfg(
    G1StandCoordinationOnlineArmLocalIKV2ReferenceVisualEnvCfg
):
    """Generator-only preview spanning the released augmentation amplitudes.

    This is intentionally isolated from training and physical-wrench tasks.
    The root, waist and legs remain exactly frozen; infeasible wrist targets
    are rejected by the same strict q-ref FK/IK gates as the safe preview.
    """

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.upper_body_joints
        # Released examples reach roughly 0.14 m wrist displacement.  Sample a
        # broad but less pathological subset of the official 0--140 N,
        # 10--1000 N/m envelope so the motion is visually inspectable.
        command.force_range_n = (1.0, 13.0)
        command.stiffness_range_n_per_m = (70.0, 350.0)
        command.max_displacement_m = 0.140
        command.arm_delta_limit_rad = 0.70
        command.ramp_time_range_s = (0.4, 0.9)
        command.max_active_link_residual_m = 0.005


# new change
@configclass
class G1StandCoordinationOnlineArmLocalIKV2ReleaseAmplitudeBatchEnvCfg(
    G1StandCoordinationOnlineArmLocalIKV2ReleaseAmplitudeVisualEnvCfg
):
    """Marker-free batched acceptance test for release-amplitude references."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.debug_vis = False
        self.scene.num_envs = 512
        self.scene.env_spacing = 2.5


# new change
@configclass
class G1StandCoordinationOnlineArmLocalIKV2LowVisualEnvCfg(
    G1StandCoordinationOnlineArmLocalIKV2ReleaseAmplitudeVisualEnvCfg
):
    """Low disturbance preview: 4--7 N, reference generation only."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.upper_body_joints
        command.difficulty_label = "LOW 4-7N"
        command.force_range_n = (4.0, 7.0)
        command.stiffness_range_n_per_m = (80.0, 300.0)


# new change
@configclass
class G1StandCoordinationOnlineArmLocalIKV2MediumVisualEnvCfg(
    G1StandCoordinationOnlineArmLocalIKV2ReleaseAmplitudeVisualEnvCfg
):
    """Medium disturbance preview: 7--10 N, reference generation only."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.upper_body_joints
        command.difficulty_label = "MEDIUM 7-10N"
        command.force_range_n = (7.0, 10.0)
        command.stiffness_range_n_per_m = (80.0, 300.0)


# new change
@configclass
class G1StandCoordinationOnlineArmLocalIKV2HighVisualEnvCfg(
    G1StandCoordinationOnlineArmLocalIKV2ReleaseAmplitudeVisualEnvCfg
):
    """High disturbance preview: 10--15 N, reference generation only."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.upper_body_joints
        command.difficulty_label = "HIGH 10-15N"
        command.force_range_n = (10.0, 15.0)
        command.stiffness_range_n_per_m = (80.0, 300.0)


# new change
@configclass
class G1StandCoordinationFrozenControllerLowForceClosedLoopEnvCfg(
    G1StandCoordinationOnlineArmLocalIKV2BaseActorEnvCfg
):
    """Frozen model_1375 closed-loop gate with synchronized 4--7 N wrist force.

    Unlike the reference-only visual tasks, this configuration keeps the root
    free, never writes q_ref into simulator state, and lets the deterministic
    legacy policy produce every action.  The online command supplies only the
    adapted reference and force context; the event applies the same accepted
    force to the selected physical wrist link.
    """

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.upper_body_joints
        command.difficulty_label = "FROZEN CLOSED-LOOP LOW 4-7N"
        command.force_range_n = (4.0, 7.0)
        command.stiffness_range_n_per_m = (80.0, 300.0)
        command.max_displacement_m = 0.140
        command.arm_delta_limit_rad = 0.70
        command.ramp_time_range_s = (0.4, 0.9)
        command.max_active_link_residual_m = 0.005
        command.physical_force_enabled = True
        command.debug_vis = False

        # Apply the buffered wrench throughout every physics sub-step.  The G1
        # asset already uses four velocity iterations, so no actuator or solver
        # gain is changed here.
        self.sim.physx.enable_external_forces_every_iteration = True

        # Explicit invariants separating this task from the kinematic preview.
        self.scene.robot.spawn.articulation_props.fix_root_link = False
        if getattr(self.events, "write_online_reference_kinematic_state", None) is not None:
            raise RuntimeError(
                "FrozenController ClosedLoop must never write q_ref into simulator state."
            )
        if getattr(self.events, "apply_online_compliance_force", None) is None:
            raise RuntimeError(
                "FrozenController LowForce requires the synchronized physical-force event."
            )

        # 1500 control steps are 30 s.  Keep the episode longer so timeout
        # resets cannot contaminate the requested long-rollout failure count.
        self.episode_length_s = 40.0


# new change
@configclass
class G1StandCoordinationFrozenControllerZeroForceClosedLoopEnvCfg(
    G1StandCoordinationFrozenControllerLowForceClosedLoopEnvCfg
):
    """Strict F=0 closed-loop baseline with identical physics and policy."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.upper_body_joints
        command.difficulty_label = "FROZEN CLOSED-LOOP ZERO FORCE"
        command.force_range_n = (0.0, 0.0)
        command.physical_force_enabled = False
        self.events.apply_online_compliance_force = None


# new change
@configclass
class G1StandCoordinationStationarySupportDiagnosisEnvCfg(
    G1StandCoordinationFrozenControllerZeroForceClosedLoopEnvCfg
):
    """Read-only zero-force support diagnosis for accepted model_1375.

    This deliberately changes no reference, reward, action mapping, actuator,
    termination, or physics parameter.  Its separate task ID prevents support
    diagnosis from being confused with a trainable controller experiment.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.difficulty_label = (
            "STATIONARY SUPPORT DIAGNOSIS ZERO FORCE"
        )
        self.commands.upper_body_joints.debug_vis = False


# new change
@configclass
class G1StandArmWaistCOMCoordinationEnvCfg(G1StandCoordinationAccuracyV2EnvCfg):
    """Joint proximal-arm accuracy and protected waist/COM coordination.

    This is an independent successor candidate, not a mutation of AccuracyV2.
    It preserves the 100% references, actuator settings, strict mapping, wrist
    objectives, lower-body controller, and termination definitions.
    """

    def __post_init__(self):
        super().__post_init__()

        # Compact 17D waist feedback: q/q-ref/error/dq/ref-dq (3 each) plus
        # phase sin/cos.  The paired actor caps its target-space authority.
        waist_ids = (2, 5, 8)
        waist_names = [G1_SOFTMIMIC_29_JOINT_NAMES[index] for index in waist_ids]
        waist_asset_cfg = SceneEntityCfg(
            "robot", joint_names=waist_names, preserve_order=True
        )
        waist_group = G1StructuredArmAdapterObservationsCfg()
        waist_group.joint_q = ObsTerm(
            func=mdp.joint_pos, params={"asset_cfg": waist_asset_cfg.copy()}
        )
        waist_group.reference_q = ObsTerm(
            func=mdp.reference_joint_pos_subset,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": list(waist_ids),
            },
        )
        waist_group.q_minus_reference = ObsTerm(
            func=mdp.joint_position_error_to_reference_subset,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": list(waist_ids),
                "asset_cfg": waist_asset_cfg.copy(),
            },
        )
        waist_group.joint_dq = ObsTerm(
            func=mdp.joint_vel, params={"asset_cfg": waist_asset_cfg.copy()}
        )
        waist_group.reference_dq = ObsTerm(
            func=mdp.reference_joint_vel_subset,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": list(waist_ids),
            },
        )
        waist_group.phase_sincos = ObsTerm(
            func=mdp.reference_motion_phase_sincos,
            params={"command_name": "upper_body_joints"},
        )
        self.observations.waist_adapter = waist_group

        # Explicit stability objectives.  COM uses the geometric midpoint of
        # the two support feet because the official G1 USD's legacy contact
        # attribution can appear on knee links.
        self.rewards.arm_waist_com_support_tracking = RewTerm(
            func=mdp.whole_body_com_support_center_exp,
            weight=2.0,
            params={
                "sigma": 0.04,
                "asset_cfg": SceneEntityCfg("robot"),
                "support_foot_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                    ],
                    preserve_order=True,
                ),
            },
        )
        self.rewards.arm_waist_torso_orientation_tracking = RewTerm(
            func=mdp.target_orientation_exp,
            weight=2.0,
            params={
                "command_name": "upper_body_joints",
                "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            },
        )
        self.rewards.arm_waist_failure_penalty = RewTerm(
            func=mdp.is_terminated,
            weight=-25.0,
        )


# new change
@configclass
class G1StandArmWaistCOMCoordinationV2EnvCfg(
    G1StandArmWaistCOMCoordinationEnvCfg
):
    """State-aware successor after V1 showed negligible waist control credit."""

    def __post_init__(self):
        super().__post_init__()

        previous = self.observations.waist_adapter
        state_aware = G1StateAwareWaistAdapterObservationsCfg()
        for term_name in (
            "joint_q",
            "reference_q",
            "q_minus_reference",
            "joint_dq",
            "reference_dq",
            "phase_sincos",
        ):
            setattr(state_aware, term_name, getattr(previous, term_name))
        state_aware.com_support_offset = ObsTerm(
            func=mdp.whole_body_com_support_offset_b,
            # new change
            # Typical offsets are 1--2 cm; scale to O(0.1) so the small head
            # can use them without increasing its output authority.
            scale=10.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "support_foot_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                    ],
                    preserve_order=True,
                ),
            },
        )
        state_aware.torso_projected_gravity = ObsTerm(
            func=mdp.projected_gravity_bodies,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["torso_link"], preserve_order=True
                )
            },
        )
        state_aware.base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.25,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.observations.waist_adapter = state_aware


# new change
@configclass
class G1StandArmWaistCOMCoordinationV3EnvCfg(
    G1StandArmWaistCOMCoordinationV2EnvCfg
):
    """V3 adds bounded state-aware support-chain coordination capacity."""

    def __post_init__(self):
        super().__post_init__()

        # Pitch chain first, then roll chain. These ids match the V3 actor's
        # output projection and are indices in the strict 29DOF interface.
        support_ids = (0, 1, 13, 14, 3, 4, 17, 18)
        support_names = [
            G1_SOFTMIMIC_29_JOINT_NAMES[index] for index in support_ids
        ]
        support_asset_cfg = SceneEntityCfg(
            "robot", joint_names=support_names, preserve_order=True
        )
        support_group = G1StateAwareWaistAdapterObservationsCfg()
        support_group.joint_q = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": support_asset_cfg.copy()},
        )
        support_group.reference_q = ObsTerm(
            func=mdp.reference_joint_pos_subset,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": list(support_ids),
            },
        )
        support_group.q_minus_reference = ObsTerm(
            func=mdp.joint_position_error_to_reference_subset,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": list(support_ids),
                "asset_cfg": support_asset_cfg.copy(),
            },
        )
        support_group.joint_dq = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": support_asset_cfg.copy()},
        )
        support_group.reference_dq = ObsTerm(
            func=mdp.reference_joint_vel_subset,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": list(support_ids),
            },
        )
        support_group.phase_sincos = ObsTerm(
            func=mdp.reference_motion_phase_sincos,
            params={"command_name": "upper_body_joints"},
        )
        support_group.com_support_offset = ObsTerm(
            func=mdp.whole_body_com_support_offset_b,
            scale=10.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "support_foot_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                    ],
                    preserve_order=True,
                ),
            },
        )
        support_group.torso_projected_gravity = ObsTerm(
            func=mdp.projected_gravity_bodies,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["torso_link"], preserve_order=True
                )
            },
        )
        support_group.base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.25,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.observations.support_adapter = support_group


# new change
@configclass
class G1StandCoordinationStationarySupportAdapterEnvCfg(
    G1StandArmWaistCOMCoordinationV3EnvCfg
):
    """Correct model_1375 skating with one bounded support-only adapter.

    The inherited actor, bilateral arm heads, waist head, references, mapping,
    actuators, and terminations remain unchanged.  Only the new support head
    receives the missing translational/contact state and is trainable.
    """

    def __post_init__(self):
        super().__post_init__()

        previous = self.observations.support_adapter
        enhanced = G1StationarySupportAdapterObservationsCfg()
        for term_name in (
            "joint_q",
            "reference_q",
            "q_minus_reference",
            "joint_dq",
            "reference_dq",
            "phase_sincos",
            "com_support_offset",
            "torso_projected_gravity",
            "base_ang_vel",
        ):
            setattr(enhanced, term_name, getattr(previous, term_name))

        enhanced.root_xy_position_error = ObsTerm(
            func=mdp.root_xy_position_error_to_reference_b,
            params={
                "command_name": "upper_body_joints",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        enhanced.base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        enhanced.foot_velocities = ObsTerm(
            func=mdp.foot_velocities,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                    ],
                    preserve_order=True,
                )
            },
        )
        resolved_support_cfg = SceneEntityCfg(
            "contact_forces",
            body_names=["left_knee_link", "right_knee_link"],
            preserve_order=True,
        )
        enhanced.resolved_support_forces = ObsTerm(
            func=mdp.resolved_support_vertical_forces,
            scale=1.0 / 500.0,
            params={"sensor_cfg": resolved_support_cfg.copy()},
        )
        enhanced.resolved_support_contacts = ObsTerm(
            func=mdp.resolved_support_contacts,
            params={
                "sensor_cfg": resolved_support_cfg.copy(),
                "threshold_n": 20.0,
            },
        )
        self.observations.support_adapter = enhanced

        # The old ankle sensor rows carry almost no force on the official G1
        # asset, so the inherited slide term was effectively zero.  Resolve
        # force and velocity in their respective sensor/body index spaces.
        self.rewards.feet_stumble = RewTerm(
            func=mdp.feet_slide_proportional,
            weight=-0.005,
            params={
                "sensor_cfg": resolved_support_cfg.copy(),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                    ],
                    preserve_order=True,
                ),
            },
        )
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.15
        self.rewards.stationary_root_xy_velocity = RewTerm(
            func=mdp.stationary_root_xy_velocity_l2,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.stationary_root_xy_position = RewTerm(
            func=mdp.stationary_root_xy_position_error_l2,
            weight=-0.5,
            params={
                "command_name": "upper_body_joints",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.stationary_double_support = RewTerm(
            func=mdp.resolved_double_support,
            weight=0.5,
            params={
                "sensor_cfg": resolved_support_cfg.copy(),
                "threshold_n": 20.0,
            },
        )


# new change
@configclass
class G1StandCoordinationStationarySupportControlV2EnvCfg(
    G1StandCoordinationStationarySupportAdapterEnvCfg
):
    """Credit-isolated fixed-foot support control with a complete 10DOF chain.

    The accepted model_1375 actor, bilateral arms, waist, references,
    actuators, physics, and termination definitions remain unchanged.  V2
    adds bilateral knee pitch to the support head and exposes fixed-world foot
    anchor errors so whole-body translation cannot disappear in a root-local
    keypoint objective.
    """

    def __post_init__(self):
        super().__post_init__()

        # Full support order: sagittal hip/knee/ankle followed by lateral
        # hip/ankle.  The model and masked Gaussian use this exact ordering.
        support_ids = (0, 1, 9, 10, 13, 14, 3, 4, 17, 18)
        support_names = [
            G1_SOFTMIMIC_29_JOINT_NAMES[index] for index in support_ids
        ]
        support_asset_cfg = SceneEntityCfg(
            "robot", joint_names=support_names, preserve_order=True
        )

        previous = self.observations.support_adapter
        enhanced = G1StationarySupportControlV2ObservationsCfg()
        enhanced.joint_q = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": support_asset_cfg.copy()},
        )
        enhanced.reference_q = ObsTerm(
            func=mdp.reference_joint_pos_subset,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": list(support_ids),
            },
        )
        enhanced.q_minus_reference = ObsTerm(
            func=mdp.joint_position_error_to_reference_subset,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": list(support_ids),
                "asset_cfg": support_asset_cfg.copy(),
            },
        )
        enhanced.joint_dq = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": support_asset_cfg.copy()},
        )
        enhanced.reference_dq = ObsTerm(
            func=mdp.reference_joint_vel_subset,
            params={
                "command_name": "upper_body_joints",
                "reference_joint_ids": list(support_ids),
            },
        )
        for term_name in (
            "phase_sincos",
            "com_support_offset",
            "torso_projected_gravity",
            "base_ang_vel",
            "root_xy_position_error",
            "base_lin_vel",
            "foot_velocities",
            "resolved_support_forces",
            "resolved_support_contacts",
        ):
            setattr(enhanced, term_name, getattr(previous, term_name))

        fixed_foot_cfg = SceneEntityCfg(
            "robot",
            body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        enhanced.fixed_foot_xy_error = ObsTerm(
            func=mdp.fixed_foot_xy_error_to_reference_b,
            scale=10.0,
            params={
                "command_name": "upper_body_joints",
                "reference_keypoint_ids": [3, 6],
                "asset_cfg": fixed_foot_cfg.copy(),
            },
        )
        self.observations.support_adapter = enhanced

        # Support-only credit: the protected actor still produces the entire
        # 29DOF action, but fixed arm/waist tracking terms do not dominate the
        # new head's advantage.  Lower-body posture and safety regularizers
        # remain active constraints.
        for reward_name in (
            "joint_deviation_upper_body_commanded",
            "keypoint_deviation_from_command",
            "joint_velocity_tracking",
            "stationary_torso_keypoint_tracking",
            "stationary_left_shoulder_keypoint_tracking",
            "stationary_right_shoulder_keypoint_tracking",
            "stationary_left_elbow_keypoint_tracking",
            "stationary_right_elbow_keypoint_tracking",
            "stationary_left_wrist_keypoint_tracking",
            "stationary_right_wrist_keypoint_tracking",
            "stand_coordination_waist_velocity_tracking",
            "stand_coordination_arm_velocity_tracking",
            "accuracy_waist_yaw_tracking",
            "accuracy_waist_roll_tracking",
            "accuracy_waist_pitch_tracking",
            "accuracy_left_wrist_roll_tracking",
            "accuracy_right_wrist_roll_tracking",
            "accuracy_left_wrist_pitch_tracking",
            "accuracy_right_wrist_pitch_tracking",
            "accuracy_left_wrist_yaw_tracking",
            "accuracy_right_wrist_yaw_tracking",
            "accuracy_right_wrist_velocity_tracking",
            "accuracy_left_shoulder_pitch_tracking",
            "accuracy_right_shoulder_pitch_tracking",
            "accuracy_left_shoulder_roll_tracking",
            "accuracy_right_shoulder_roll_tracking",
            "accuracy_left_shoulder_yaw_tracking",
            "accuracy_right_shoulder_yaw_tracking",
            "accuracy_left_elbow_tracking",
            "accuracy_right_elbow_tracking",
        ):
            setattr(self.rewards, reward_name, None)

        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.feet_stumble.weight = -0.01
        self.rewards.stationary_root_xy_velocity.weight = -4.0
        self.rewards.stationary_root_xy_position.weight = -2.0
        self.rewards.stationary_double_support.weight = 2.0
        # new change
        # Give the new support head its own COM/orientation/failure credit
        # channels.  The inherited arm/waist names are disabled so future
        # reward audits cannot mistake these terms for upper-body objectives.
        self.rewards.arm_waist_com_support_tracking = None
        self.rewards.arm_waist_torso_orientation_tracking = None
        self.rewards.arm_waist_failure_penalty = None
        self.rewards.stationary_support_com = RewTerm(
            func=mdp.whole_body_com_support_center_exp,
            weight=2.0,
            params={
                "sigma": 0.04,
                "asset_cfg": SceneEntityCfg("robot"),
                "support_foot_cfg": fixed_foot_cfg.copy(),
            },
        )
        self.rewards.stationary_support_torso_orientation = RewTerm(
            func=mdp.target_orientation_exp,
            weight=2.0,
            params={
                "command_name": "upper_body_joints",
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["torso_link"]
                ),
            },
        )
        self.rewards.stationary_support_failure = RewTerm(
            func=mdp.is_terminated,
            weight=-50.0,
        )
        self.rewards.stationary_fixed_foot_xy = RewTerm(
            func=mdp.fixed_foot_xy_position_error_l2,
            weight=-10.0,
            params={
                "command_name": "upper_body_joints",
                "reference_keypoint_ids": [3, 6],
                "asset_cfg": fixed_foot_cfg.copy(),
            },
        )


# new change
@configclass
class G1StandCoordinationStationarySupportControlV2SupervisedInitEnvCfg(
    G1StandCoordinationStationarySupportControlV2EnvCfg
):
    """V2 support task with bounded fixed-foot credit for supervised bootstrap."""

    def __post_init__(self):
        super().__post_init__()
        fixed_foot_cfg = SceneEntityCfg(
            "robot",
            body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        # The penalty is bounded to [-1, 0], preventing long-rollout global
        # drift from producing unbounded critic targets while preserving a
        # quadratic basin below 10 cm.
        self.rewards.stationary_fixed_foot_xy = RewTerm(
            func=mdp.fixed_foot_xy_position_error_bounded_huber,
            weight=-1.0,
            params={
                "command_name": "upper_body_joints",
                "reference_keypoint_ids": [3, 6],
                "asset_cfg": fixed_foot_cfg,
                "delta_m": 0.10,
            },
        )


# new change
@configclass
class G1StandCoordinationSupportRecoveryAwareTeacherV3EnvCfg(
    G1StandCoordinationStationarySupportControlV2SupervisedInitEnvCfg
):
    """Independent 87D recovery-aware support-teacher task from model_1375."""

    def __post_init__(self):
        super().__post_init__()
        previous = self.observations.support_adapter
        enhanced = G1SupportRecoveryAwareTeacherV3ObservationsCfg()
        for term_name in (
            "joint_q",
            "reference_q",
            "q_minus_reference",
            "joint_dq",
            "reference_dq",
            "phase_sincos",
            "com_support_offset",
            "torso_projected_gravity",
            "base_ang_vel",
            "root_xy_position_error",
            "base_lin_vel",
            "foot_velocities",
            "resolved_support_forces",
            "resolved_support_contacts",
            "fixed_foot_xy_error",
        ):
            setattr(enhanced, term_name, getattr(previous, term_name))
        enhanced.failure_trend = ObsTerm(
            func=mdp.support_failure_trend_features,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "support_foot_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                    ],
                    preserve_order=True,
                ),
                "torso_cfg": SceneEntityCfg(
                    "robot", body_names=["torso_link"], preserve_order=True
                ),
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["left_knee_link", "right_knee_link"],
                    preserve_order=True,
                ),
            },
        )
        self.observations.support_adapter = enhanced


# new change
@configclass
class G1StandCoordinationStationaryFoundationV2EnvCfg(
    G1StandCoordinationSupportRecoveryAwareTeacherV3EnvCfg
):
    """Zero-force foundation task with bounded world-anchor support credit.

    This class reuses only TeacherV3's 87D measured support observation.  The
    policy model is a direct residual head; no teacher correction or safety
    Gate is present in this task.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.difficulty_label = (
            "STATIONARY FOUNDATION V2 ZERO FORCE"
        )
        self.commands.upper_body_joints.debug_vis = False

        # Replace the two unbounded drift losses that destabilized the old
        # SupportControlV2 critic.  Both terms remain sensitive near the
        # reference but are bounded to [0, 1) per environment.
        self.rewards.stationary_root_xy_position = RewTerm(
            func=mdp.stationary_root_xy_position_error_bounded_huber,
            weight=-1.0,
            params={
                "command_name": "upper_body_joints",
                "asset_cfg": SceneEntityCfg("robot"),
                "delta_m": 0.10,
            },
        )
        self.rewards.stationary_root_xy_velocity = RewTerm(
            func=mdp.stationary_root_xy_velocity_bounded_huber,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "delta_mps": 0.15,
            },
        )
        # The force-weighted slide loss is unbounded and redundant with the
        # bounded fixed-world foot anchors in this stationary task.
        self.rewards.feet_stumble = None

        resolved_support_cfg = SceneEntityCfg(
            "contact_forces",
            body_names=["left_knee_link", "right_knee_link"],
            preserve_order=True,
        )
        self.rewards.stationary_support_force_margin = RewTerm(
            func=mdp.resolved_support_force_margin,
            weight=0.5,
            params={
                "sensor_cfg": resolved_support_cfg,
                "threshold_n": 20.0,
                "transition_n": 20.0,
            },
        )
        # Keep all remaining mapping, actuators, COM/orientation terms and
        # termination logic inherited from the verified model_1375 task.


# new change
@configclass
class G1StandCoordinationStationarySupportContainmentV3EnvCfg(
    G1StandCoordinationStationaryFoundationV2EnvCfg
):
    """Credit-correct stationary support stage with bounded 6 mrad authority."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.difficulty_label = (
            "STATIONARY SUPPORT CONTAINMENT V3 ZERO FORCE"
        )
        fixed_foot_cfg = SceneEntityCfg(
            "robot",
            body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )

        # Keep every loss bounded, but make translation credit dominate the
        # positive COM/orientation terms inside the 40 cm containment basin.
        self.rewards.stationary_root_xy_position.weight = -4.0
        self.rewards.stationary_root_xy_position.params["delta_m"] = 0.20
        self.rewards.stationary_root_xy_velocity.weight = -3.0
        self.rewards.stationary_root_xy_velocity.params["delta_mps"] = 0.20
        self.rewards.stationary_fixed_foot_xy.weight = -4.0
        self.rewards.stationary_fixed_foot_xy.params["delta_m"] = 0.20
        self.rewards.stationary_double_support.weight = 0.5
        self.rewards.stationary_support_force_margin.weight = 0.25
        self.rewards.stationary_support_com.weight = 1.0
        self.rewards.stationary_support_torso_orientation.weight = 1.0
        self.rewards.stationary_support_failure.weight = -100.0
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.alive.weight = 0.25

        self.terminations.stationary_support_containment = DoneTerm(
            func=mdp.stationary_support_containment,
            params={
                "command_name": "upper_body_joints",
                "reference_keypoint_ids": [3, 6],
                "root_limit_m": 0.40,
                "foot_limit_m": 0.40,
                "asset_cfg": SceneEntityCfg("robot"),
                "support_foot_cfg": fixed_foot_cfg,
            },
        )
        self.episode_length_s = 20.0


# new change
@configclass
class G1StandCoordinationStationarySupportActorRowsV4EnvCfg(
    G1StandCoordinationStationarySupportContainmentV3EnvCfg
):
    """Containment task for direct fine-tuning of legacy support output rows."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.difficulty_label = (
            "STATIONARY SUPPORT ACTOR ROWS V4 ZERO FORCE"
        )
        # The ten selected rows must be allowed to depart modestly from q_ref
        # to retain dynamic balance.  Keep tracking credit present, but stop it
        # from overwhelming world-anchor and root-velocity containment credit.
        self.rewards.stand_coordination_lower_body_tracking.weight = 0.5
        self.rewards.joint_deviation_contacts_commanded.weight = -0.1


# new change
@configclass
class G1StandCoordinationStationarySupportStateResidualV5EnvCfg(
    G1StandCoordinationStationarySupportActorRowsV4EnvCfg
):
    """State-visible bounded support stage after the V4 observability audit."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.difficulty_label = (
            "STATIONARY SUPPORT STATE RESIDUAL V5 ZERO FORCE"
        )


# new change
@configclass
class G1StandCoordinationStationaryBaseStateAwareV6EnvCfg(
    G1StandCoordinationStationarySupportStateResidualV5EnvCfg
):
    """State-aware stationary-base learning with protected upper body."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.difficulty_label = (
            "STATIONARY BASE STATE AWARE V6 ZERO FORCE"
        )
        # Root/foot containment is the sole optimization target of this stage;
        # keep posture as a soft constraint rather than forcing the new head
        # back to the drifting model_1375 lower-body targets.
        self.rewards.stationary_root_xy_position.weight = -6.0
        self.rewards.stationary_root_xy_velocity.weight = -5.0
        self.rewards.stationary_fixed_foot_xy.weight = -5.0
        self.rewards.stationary_double_support.weight = 1.0
        self.rewards.stationary_support_com.weight = 1.5
        self.rewards.stationary_support_torso_orientation.weight = 1.5
        self.rewards.stationary_support_force_margin.weight = 0.5
        self.rewards.stationary_support_failure.weight = -100.0
        self.rewards.stand_coordination_lower_body_tracking.weight = 0.1
        self.rewards.joint_deviation_contacts_commanded.weight = -0.02


# new change
@configclass
class G1StandCoordinationStationaryCriticAwareV7EnvCfg(
    G1StandCoordinationStationaryBaseStateAwareV6EnvCfg
):
    """Expose delayed stationary-containment state to the privileged critic.

    V6 gave the support actor the required 87D state but retained model_1375's
    critic observation.  That critic could see velocity, but not absolute root
    displacement or fixed-foot anchor error, even though those variables
    determine the delayed containment termination.  V7 appends only the
    missing privileged state; policy observations, actions, rewards,
    actuators and termination thresholds are unchanged.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.difficulty_label = (
            "STATIONARY CRITIC AWARE V7 ZERO FORCE"
        )
        support = self.observations.support_adapter
        # These terms are appended to the critic's existing history-aware
        # group.  The V7 checkpoint pads the old first critic layer with zero
        # columns, so its initial value output remains exactly model_1375.
        self.observations.critic.stationary_root_xy_position_error = copy.deepcopy(
            support.root_xy_position_error
        )
        self.observations.critic.stationary_com_support_offset = copy.deepcopy(
            support.com_support_offset
        )
        self.observations.critic.stationary_fixed_foot_xy_error = copy.deepcopy(
            support.fixed_foot_xy_error
        )
        self.observations.critic.stationary_resolved_support_forces = copy.deepcopy(
            support.resolved_support_forces
        )
        self.observations.critic.stationary_resolved_support_contacts = copy.deepcopy(
            support.resolved_support_contacts
        )
        self.observations.critic.stationary_failure_trend = copy.deepcopy(
            support.failure_trend
        )


# new change
@configclass
class G1StandCoordinationFrozenNeighborGateZeroForceEnvCfg(
    G1StandCoordinationSupportRecoveryAwareTeacherV3EnvCfg
):
    """Inference-only zero-force environment for the accepted neighbor gate."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.difficulty_label = (
            "FROZEN MODEL_1375 NEIGHBOR GATE ZERO FORCE"
        )
        self.commands.upper_body_joints.debug_vis = False
        self.episode_length_s = 40.0


# new change
@configclass
class G1StandCoordinationFrozenNeighborGateLowForceEnvCfg(
    G1StandCoordinationFrozenControllerLowForceClosedLoopEnvCfg
):
    """Inference-only 4--7 N task with the identical 87D support state.

    The online IK command, synchronized wrist wrench, original 29DOF actor,
    actuator gains, mapping and physics all come from the already validated
    low-force closed-loop task.  Only the read-only support observation group
    is copied from TeacherV3 so the external neighbor gate can evaluate state.
    """

    def __post_init__(self):
        super().__post_init__()
        support_source = G1StandCoordinationSupportRecoveryAwareTeacherV3EnvCfg()
        self.observations.support_adapter = copy.deepcopy(
            support_source.observations.support_adapter
        )
        self.commands.upper_body_joints.difficulty_label = (
            "FROZEN MODEL_1375 NEIGHBOR GATE LOW 4-7N"
        )
        self.commands.upper_body_joints.debug_vis = False
        self.episode_length_s = 40.0


# new change
@configclass
class G1StandCoordinationV2FrozenInferenceZeroForceEnvCfg(
    G1StandCoordinationFrozenNeighborGateZeroForceEnvCfg
):
    """Independent zero-force task for the accepted V2 multi-target gate."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.difficulty_label = (
            "FROZEN MODEL_1375 V2 MULTI-TARGET GATE ZERO FORCE"
        )
        self.commands.upper_body_joints.debug_vis = False
        self.episode_length_s = 40.0


# new change
@configclass
class G1StandCoordinationV2FrozenInferenceLowForceEnvCfg(
    G1StandCoordinationFrozenNeighborGateLowForceEnvCfg
):
    """Independent 4--7 N task for the accepted V2 multi-target gate."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.upper_body_joints.difficulty_label = (
            "FROZEN MODEL_1375 V2 MULTI-TARGET GATE LOW 4-7N"
        )
        self.commands.upper_body_joints.debug_vis = False
        self.episode_length_s = 40.0


# new change
@configclass
class G1StandCoordinationV1NeutralBootstrapEnvCfg(G1StandCoordinationV1EnvCfg):
    """Curriculum stage A: learn active balance on the new neutral reference."""

    def __post_init__(self):
        super().__post_init__()
        neutral_path = STAND_COORDINATION_V1_MOTIONS[0]
        self.commands.upper_body_joints.demo_recording_path = [neutral_path]
        self.commands.upper_body_joints.demo_start_range = [[0.0, 0.0]]


# new change
@configclass
class G1GeneralTrackingV1StationaryStructuredArmsAdapterEnvCfg(
    G1GeneralTrackingV1StationaryArmsCreditIsolatedEnvCfg
):
    """Credit-isolated stationary task with generic bilateral feedback features."""

    def __post_init__(self):
        super().__post_init__()

        def add_adapter_group(side: str, reference_ids: tuple[int, ...]) -> None:
            joint_names = [G1_SOFTMIMIC_29_JOINT_NAMES[index] for index in reference_ids]
            asset_cfg = SceneEntityCfg(
                "robot", joint_names=joint_names, preserve_order=True
            )
            group = G1StructuredArmAdapterObservationsCfg()
            group.joint_q = ObsTerm(
                func=mdp.joint_pos, params={"asset_cfg": asset_cfg.copy()}
            )
            group.reference_q = ObsTerm(
                func=mdp.reference_joint_pos_subset,
                params={
                    "command_name": "upper_body_joints",
                    "reference_joint_ids": list(reference_ids),
                },
            )
            group.q_minus_reference = ObsTerm(
                func=mdp.joint_position_error_to_reference_subset,
                params={
                    "command_name": "upper_body_joints",
                    "reference_joint_ids": list(reference_ids),
                    "asset_cfg": asset_cfg.copy(),
                },
            )
            group.joint_dq = ObsTerm(
                func=mdp.joint_vel, params={"asset_cfg": asset_cfg.copy()}
            )
            group.reference_dq = ObsTerm(
                func=mdp.reference_joint_vel_subset,
                params={
                    "command_name": "upper_body_joints",
                    "reference_joint_ids": list(reference_ids),
                },
            )
            group.phase_sincos = ObsTerm(
                func=mdp.reference_motion_phase_sincos,
                params={"command_name": "upper_body_joints"},
            )
            setattr(self.observations, f"{side}_arm_adapter", group)

        add_adapter_group("left", G1_SOFTMIMIC_LEFT_ARM_ACTION_IDS)
        add_adapter_group("right", G1_SOFTMIMIC_RIGHT_ARM_ACTION_IDS)
