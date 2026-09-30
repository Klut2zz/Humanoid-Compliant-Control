"""Minimal configuration set for deployable G1 SoftMimic tasks."""

from __future__ import annotations

import os
import pathlib
from typing import Callable, Dict, Tuple, Type

from isaaclab.utils import configclass

import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import softmimic_gym.tasks.locomotion.tracking.mdp as mdp
from softmimic_gym.assets.robots.unitree import G1_CFG
from softmimic_gym.tasks.locomotion.tracking.tracking_env_cfg import TrackingEnvCfg


# ---------------------------------------------------------------------------
# Shared path helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = pathlib.Path(__file__).absolute().parents[7]
_RELEASE_FORCEFIELD_ROOT = _PROJECT_ROOT / "compliant_motion_augmentation" / "release_examples"

_G1_KEYPOINT_BODY_IDS: Tuple[int, ...] = (
    0,
    9,
    12,
    26,
    10,
    13,
    23,
    11,
    28,
    30,
    36,
    29,
    31,
    37,
)


def _sequential_paths(root: pathlib.Path, pattern: str, count: int) -> Tuple[str, ...]:
    return tuple(str(root / pattern.format(i=i)) for i in range(1, count + 1))


# ---------------------------------------------------------------------------
# Augmented-reference helper stages
# ---------------------------------------------------------------------------

@configclass
class G1DeployableBaseEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Robot setup
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.usd_path = os.path.join(
            _PROJECT_ROOT,
            "softmimic_gym/softmimic_gym/assets/robots/data/g1_full_reduced_handcollisions.usd",
        )
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False
        self.scene.height_scanner = None

        # Command configuration
        base_cmd = self.commands.upper_body_joints
        proc_ranges = base_cmd.proc_ranges.copy()
        proc_ranges.min_height = 0.4
        proc_ranges.max_height = 0.9
        proc_ranges.min_standing_height = 0.4
        proc_ranges.max_standing_height = 0.9
        proc_ranges.min_pitch = -1.5
        proc_ranges.max_pitch = 2.0
        proc_ranges.min_standing_pitch = -1.5
        proc_ranges.max_standing_pitch = 2.0

        joint_config = mdp.ReferenceCommandCfg.JointConfig(
            num_joints=29,
            thigh_length=0.3,
            calf_length=0.3,
            robot_type="g1",
            left_leg_indices=mdp.ReferenceCommandCfg.JointConfig.LegIndicesConfig(
                hip_yaw=6,
                hip_roll=3,
                hip_pitch=0,
                knee=9,
                ankle_pitch=13,
            ),
            right_leg_indices=mdp.ReferenceCommandCfg.JointConfig.LegIndicesConfig(
                hip_yaw=7,
                hip_roll=4,
                hip_pitch=1,
                knee=10,
                ankle_pitch=14,
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

        demo_paths = [
            str(_RELEASE_FORCEFIELD_ROOT / "forcefield" / "stand" / f"stand_augmented_mink_{i:03d}.csv")
            for i in range(1, 41)
        ]
        demo_start_range = [[0, 0] for _ in demo_paths]

        self.commands.upper_body_joints = mdp.ComplianceAugmentedReferenceCommandCfg(
            asset_name="robot",
            resampling_time_range=(10000.0, 10000.0),
            n_future_steps=100,
            future_dt=1.0 / 50.0,
            proc_ranges=proc_ranges,
            joint_config=joint_config,
            keypoint_body_ids=list(_G1_KEYPOINT_BODY_IDS),
            demo_playback_mode="references",
            demo_recording_path=demo_paths,
            demo_start_range=demo_start_range,
            demo_stand_height=0.0,
            demo_pitch_angle=0.0,
            demo_random_init=True,
            demo_random_reset=False,
            demo_zero_reset=False,
            demo_loader_class="AugmentedMotionLibFromDemo",
            debug_vis=True,
        )
        upper_body = self.commands.upper_body_joints
        upper_body.clip_length = None

        # Events
        self.events.apply_forcefield_force = EventTerm(
            func=mdp.apply_forcefield_force_torque,
            mode="interval",
            interval_range_s=(0.005, 0.005),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "upper_body_joints",
            },
        )

        # Rewards
        keypoints = list(_G1_KEYPOINT_BODY_IDS)
        self.rewards.keypoint_deviation_from_command.func = mdp.keypoint_deviation_from_command_local
        self.rewards.keypoint_deviation_from_command.params["keypoint_body_ids"] = keypoints
        self.rewards.keypoint_orientation_deviation_from_command.params["keypoint_body_ids"] = keypoints
        self.rewards.contact_forces = None
        self.rewards.ang_vel_xy_l2 = None
        self.rewards.lin_vel_z_l2 = None
        self.rewards.height_exp = None
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_local
        self.rewards.track_ang_vel_z_exp.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["pelvis"])
        self.rewards.track_ang_vel_z_exp.params["std"] = 2.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_local
        self.rewards.track_lin_vel_xy_exp.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["pelvis"])
        self.rewards.gravity_exp.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["pelvis"])
        self.rewards.joint_deviation_contacts_commanded = RewTerm(
            func=mdp.joint_deviation_from_command_contacts_prob,
            weight=-0.4,
            params={
                "command_name": "upper_body_joints",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            },
        )
        self.rewards.dof_pos_limits = RewTerm(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
        )
        self.rewards.force_link_keypoint_tracking_local = RewTerm(
            func=mdp.force_link_keypoint_tracking_local,
            params={"command_name": "upper_body_joints", "sigma": 0.1},
            weight=3.0,
        )
        self.rewards.force_link_keypoint_orientation_tracking = RewTerm(
            func=mdp.force_link_keypoint_orientation_tracking_local,
            params={"command_name": "upper_body_joints", "sigma": 0.1},
            weight=3.0,
        )
        self.rewards.force_command_tracking = RewTerm(
            func=mdp.force_command_tracking,
            params={"command_name": "upper_body_joints", "sigma": 20.0},
            weight=2.0,
        )
        self.rewards.torque_command_tracking = RewTerm(
            func=mdp.torque_command_tracking,
            params={"command_name": "upper_body_joints", "sigma": 2.0},
            weight=2.0,
        )
        # self.rewards.joint_deviation_upper_body_commanded = None

        # Terminations
        self.terminations.base_contact = None
        self.terminations.bad_orientation = None
        self.terminations.bad_keypoints = DoneTerm(
            func=mdp.bad_keypoint_deviation_local,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "pelvis",
                        "left_hip_yaw_link",
                        "left_knee_link",
                        "left_ankle_roll_link",
                        "right_hip_yaw_link",
                        "right_knee_link",
                        "right_ankle_roll_link",
                        "torso_link",
                        "left_shoulder_yaw_link",
                        "left_elbow_link",
                        "left_wrist_yaw_link",
                        "right_shoulder_yaw_link",
                        "right_elbow_link",
                        "right_wrist_yaw_link",
                    ],
                ),
                "keypoint_body_ids": keypoints,
                "threshold": 0.5,
                "adaptive": False,
                "command_name": "upper_body_joints",
            },
        )

        # Observations
        self.observations.policy.base_lin_vel = None
        self.observations.policy.foot_contact = None
        self.observations.policy.root_height = None
        self.observations.policy.foot_contact_forces = None
        self.observations.policy.foot_velocities = None
        self.observations.policy.keypoint_error = None
        # self.observations.policy.reference_dof_pos = None
        # self.observations.policy.reference_xy_vel = None
        # self.observations.policy.reference_yaw_vel = None
        # self.observations.policy.reference_gravity_vector = None
        # self.observations.policy.reference_foot_contact = None
        # self.observations.policy.reference_root_height = None
        self.observations.policy.reference_dof_pos_future = None
        self.observations.policy.reference_xy_vel_future = None
        self.observations.policy.reference_yaw_vel_future = None
        self.observations.policy.reference_gravity_vector_future = None
        self.observations.policy.reference_foot_contact_future = None
        self.observations.policy.reference_root_height_future = None
        self.observations.policy.force_applied = None
        self.observations.policy.torque_applied = None

        self.observations.policy.desired_stiffness_log = ObsTerm(
            func=mdp.desired_stiffness_log,
            params={"command_name": "upper_body_joints"},
            history_length=3,
        )
        self.observations.policy.desired_rotational_stiffness_log = ObsTerm(
            func=mdp.desired_rotational_stiffness_log,
            params={"command_name": "upper_body_joints"},
            history_length=3,
        )
        self.observations.critic.desired_stiffness_log = self.observations.policy.desired_stiffness_log.copy()
        self.observations.critic.desired_rotational_stiffness_log = (
            self.observations.policy.desired_rotational_stiffness_log.copy()
        )

        critic = self.observations.critic
        critic.force_applied = ObsTerm(
            func=mdp.forcefield_force_applied,
            params={"command_name": "upper_body_joints", "observe_force_body_ids": [36, 37]},
            scale=1.0 / 50.0,
            history_length=3,
        )
        critic.torque_applied = ObsTerm(
            func=mdp.forcefield_torque_applied,
            params={"command_name": "upper_body_joints", "observe_force_body_ids": [36, 37]},
            scale=1.0 / 5.0,
            history_length=3,
        )
        critic.forcefield_stiffness_log = ObsTerm(
            func=mdp.forcefield_stiffness_log,
            params={"command_name": "upper_body_joints"},
            history_length=3,
        )
        critic.forcefield_rotational_stiffness_log = ObsTerm(
            func=mdp.forcefield_rotational_stiffness_log,
            params={"command_name": "upper_body_joints"},
            history_length=3,
        )
        critic.forcefield_force_desired = ObsTerm(
            func=mdp.forcefield_force_desired,
            params={"command_name": "upper_body_joints", "observe_force_body_ids": [36, 37]},
            scale=1.0 / 50.0,
            history_length=3,
        )
        critic.forcefield_torque_desired = ObsTerm(
            func=mdp.forcefield_torque_desired,
            params={"command_name": "upper_body_joints", "observe_force_body_ids": [36, 37]},
            scale=1.0 / 5.0,
            history_length=3,
        )
        critic.all_adapted_keypoints_pos_error_local = ObsTerm(
            func=mdp.all_adapted_keypoints_pos_error_local,
            params={"command_name": "upper_body_joints"},
            history_length=3,
        )
        critic.all_adapted_keypoints_quat_error_local = ObsTerm(
            func=mdp.all_adapted_keypoints_quat_error_local,
            params={"command_name": "upper_body_joints"},
            history_length=3,
        )

        critic.adapted_reference_dof_pos = critic.reference_dof_pos.copy()
        critic.adapted_reference_xy_vel = critic.reference_xy_vel.copy()
        critic.adapted_reference_yaw_vel = critic.reference_yaw_vel.copy()
        critic.adapted_reference_root_height = critic.reference_root_height.copy()
        critic.adapted_reference_gravity_vector = critic.reference_gravity_vector.copy()
        critic.adapted_reference_foot_contact = critic.reference_foot_contact.copy()

        future_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        critic.reference_dof_pos_future = ObsTerm(
            func=mdp.reference_dof_pos, params={"command_name": "upper_body_joints", "future_steps": future_steps}
        )
        critic.reference_xy_vel_future = ObsTerm(
            func=mdp.reference_xy_vel, params={"command_name": "upper_body_joints", "future_steps": future_steps}
        )
        critic.reference_yaw_vel_future = ObsTerm(
            func=mdp.reference_ang_vel, params={"command_name": "upper_body_joints", "future_steps": future_steps}
        )
        critic.reference_root_height_future = ObsTerm(
            func=mdp.reference_root_height, params={"command_name": "upper_body_joints", "future_steps": future_steps}
        )
        critic.reference_gravity_vector_future = ObsTerm(
            func=mdp.reference_gravity_vector, params={"command_name": "upper_body_joints", "future_steps": future_steps}
        )
        critic.reference_foot_contact_future = ObsTerm(
            func=mdp.reference_foot_contacts, params={"command_name": "upper_body_joints", "future_steps": future_steps}
        )

        critic.adapted_reference_dof_pos.func = mdp.adapted_reference_dof_pos
        critic.adapted_reference_xy_vel.func = mdp.adapted_reference_xy_vel
        critic.adapted_reference_yaw_vel.func = mdp.adapted_reference_ang_vel
        critic.adapted_reference_root_height.func = mdp.adapted_reference_root_height
        critic.adapted_reference_gravity_vector.func = mdp.adapted_reference_gravity_vector
        critic.adapted_reference_foot_contact.func = mdp.adapted_reference_foot_contacts

        critic.adapted_reference_dof_pos_future = critic.reference_dof_pos_future.copy()
        critic.adapted_reference_xy_vel_future = critic.reference_xy_vel_future.copy()
        critic.adapted_reference_yaw_vel_future = critic.reference_yaw_vel_future.copy()
        critic.adapted_reference_root_height_future = critic.reference_root_height_future.copy()
        critic.adapted_reference_gravity_vector_future = critic.reference_gravity_vector_future.copy()
        critic.adapted_reference_foot_contact_future = critic.reference_foot_contact_future.copy()

        critic.adapted_reference_dof_pos_future.func = mdp.adapted_reference_dof_pos
        critic.adapted_reference_xy_vel_future.func = mdp.adapted_reference_xy_vel
        critic.adapted_reference_yaw_vel_future.func = mdp.adapted_reference_ang_vel
        critic.adapted_reference_root_height_future.func = mdp.adapted_reference_root_height
        critic.adapted_reference_gravity_vector_future.func = mdp.adapted_reference_gravity_vector
        critic.adapted_reference_foot_contact_future.func = mdp.adapted_reference_foot_contacts

        self.observations.critic.keypoint_error = ObsTerm(
            func=mdp.keypoint_error_local_shifted,
            params={
                "command_name": "upper_body_joints",
                "keypoint_body_ids": keypoints,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.observations.policy.history_length = None
        self.observations.critic.history_length = None

        for group in (critic, self.observations.policy):
            for term_name in (
                "projected_gravity",
                "joint_pos",
                "joint_vel",
                "actions",
                "base_ang_vel",
                "reference_dof_pos",
                "reference_xy_vel",
                "reference_yaw_vel",
                "reference_root_height",
                "reference_gravity_vector",
                "reference_foot_contact",
            ):
                term = getattr(group, term_name, None)
                if term is not None:
                    term.history_length = 3

        for term_name in ("base_lin_vel", "keypoint_error", "foot_contact", "root_height"):
            term = getattr(critic, term_name, None)
            if term is not None:
                term.history_length = 3

        for term_name in (
            "adapted_reference_dof_pos",
            "adapted_reference_xy_vel",
            "adapted_reference_yaw_vel",
            "adapted_reference_root_height",
            "adapted_reference_gravity_vector",
            "adapted_reference_foot_contact",
        ):
            term = getattr(critic, term_name, None)
            if term is not None:
                term.history_length = 3

        # Curriculum and terrain
        self.curriculum.terrain_levels = None
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        }

        # Events
        self.events.physics_material.params["static_friction_range"] = (0.5, 2.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 2.0)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.5)
        self.events.reset_robot.params["pose_range"]["z"] = (0.0, 0.0)
        self.events.reset_robot.params["jpos_range"] = (-0.2, 0.2)
        self.events.randomize_joint_parameters.params["armature_distribution_params"] = (0.01, 0.1)
        self.events.randomize_joint_parameters.params["friction_distribution_params"] = (0.0, 0.01)
        self.events.randomize_joint_parameters.params["lower_limit_distribution_params"] = (-0.02, 0.02)
        self.events.randomize_joint_parameters.params["upper_limit_distribution_params"] = (-0.02, 0.02)
        self.events.randomize_joint_parameters.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=".*")
        self.events.randomize_joint_damping.params["damping_distribution_params"] = (0.0, 2.0)
        self.events.randomize_joint_damping.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=".*")

        # Episode and viewer
        self.episode_length_s = 30.0
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = [1.7, 0.0, 0.25]
        self.viewer.lookat = [0.0, 0.0, 0.0]
        self.viewer.resolution = (1080, 1080)

# ---------------------------------------------------------------------------
# Motion- and mode-specific helpers
# ---------------------------------------------------------------------------


def _set_motion_dataset(cfg: TrackingEnvCfg, pattern: str, count: int) -> None:
    cfg.commands.upper_body_joints.demo_recording_path = list(_sequential_paths(_RELEASE_FORCEFIELD_ROOT, pattern, count))
    cfg.commands.upper_body_joints.demo_start_range = [
        [0, 0] for _ in cfg.commands.upper_body_joints.demo_recording_path
    ]


def _motion_default(cfg: TrackingEnvCfg) -> None:
    # Base motion already configured by G1DeployableBaseEnvCfg.
    pass


def _motion_gmt_walk_stand(cfg: TrackingEnvCfg) -> None:
    _set_motion_dataset(
        cfg,
        "forcefield/walk/walk_augmented_mink_{i:03d}.csv",
        40,
    )
    cfg.scene.terrain.terrain_generator.border_width = 50


def _motion_pour(cfg: TrackingEnvCfg) -> None:
    _set_motion_dataset(
        cfg,
        "forcefield/pour/pour_augmented_mink_{i:03d}.csv",
        40,
    )


def _motion_amass_pick1(cfg: TrackingEnvCfg) -> None:
    _set_motion_dataset(
        cfg,
        "forcefield/boxpick/boxpick_augmented_mink_{i:03d}.csv",
        40,
    )
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.eye = [0.0, 1.5, 0.25]
    cfg.viewer.lookat = [0.0, 0.0, 0.0]


def _motion_lafant_pose_loop(cfg: TrackingEnvCfg) -> None:
    _set_motion_dataset(
        cfg,
        "forcefield/tpose/tpose_augmented_mink_{i:03d}.csv",
        40,
    )


def _apply_mode_hybrid(cfg: TrackingEnvCfg) -> None:
    cfg.commands.upper_body_joints.force_computation_mode = "forcefield"
    paths = cfg.commands.upper_body_joints.demo_recording_path
    if not paths:
        return
    collision_paths = [p.replace("/forcefield/", "/collision-emulator/") for p in paths]
    zero_source = paths[:5] or paths
    zero_paths = [p.replace("/forcefield/", "/zero-wrench/") for p in zero_source]
    repeats = max(len(paths) // max(len(zero_source), 1), 1)
    zero_paths = zero_paths * repeats
    cfg.commands.upper_body_joints.demo_recording_path = list(paths + collision_paths + zero_paths)
    cfg.commands.upper_body_joints.demo_start_range = [
        [0, 0] for _ in cfg.commands.upper_body_joints.demo_recording_path
    ]
    cfg.rewards.force_command_tracking = RewTerm(
        func=mdp.force_command_tracking,
        params={
            "command_name": "upper_body_joints",
            "sigma": 20.0,
        },
        weight=2.0,
    )
    cfg.rewards.torque_command_tracking = RewTerm(
        func=mdp.torque_command_tracking,
        params={
            "command_name": "upper_body_joints",
            "sigma": 2.0,
        },
        weight=2.0,
    )


def _apply_mode_deepmimic(cfg: TrackingEnvCfg) -> None:
    _apply_mode_hybrid(cfg)
    cfg.commands.upper_body_joints.force_computation_mode = "feedforward"
    cfg.commands.upper_body_joints.override_reference = False
    cfg.rewards.force_link_keypoint_tracking_local = None
    cfg.rewards.force_link_keypoint_orientation_tracking = None
    cfg.rewards.force_command_tracking = None
    cfg.rewards.torque_command_tracking = None


def _apply_mode_deepmimic_forcefield(cfg: TrackingEnvCfg) -> None:
    _apply_mode_hybrid(cfg)
    cfg.commands.upper_body_joints.force_computation_mode = "forcefield"
    cfg.commands.upper_body_joints.override_reference = False
    cfg.rewards.force_link_keypoint_tracking_local = None
    cfg.rewards.force_link_keypoint_orientation_tracking = None
    cfg.rewards.force_command_tracking = None
    cfg.rewards.torque_command_tracking = None


def _apply_mode_deepmimic_compliant_forcefield(cfg: TrackingEnvCfg) -> None:
    cfg.commands.upper_body_joints.force_computation_mode = "forcefield"
    cfg.commands.upper_body_joints.override_reference = False
    cfg.terminations.bad_keypoints.func = mdp.bad_force_link_and_keypoint_deviation_local
    cfg.terminations.bad_keypoints.params["threshold"] = 0.5
    cfg.rewards.force_command_tracking = RewTerm(
        func=mdp.force_command_tracking,
        params={
            "command_name": "upper_body_joints",
            "sigma": 20.0,
        },
        weight=2.0,
    )
    cfg.rewards.torque_command_tracking = RewTerm(
        func=mdp.torque_command_tracking,
        params={
            "command_name": "upper_body_joints",
            "sigma": 2.0,
        },
        weight=2.0,
    )


# ---------------------------------------------------------------------------
# Motion/mode registry
# ---------------------------------------------------------------------------

MotionApplyFn = Callable[[TrackingEnvCfg], None]
ModeApplyFn = Callable[[TrackingEnvCfg], None]

MOTION_VARIANTS: Dict[str, Dict[str, object]] = {
    "": {
        "id_segment": None,
        "suffix": "",
        "allowed_modes": ("Hybrid", "DeepMimic", "DeepMimicForceField", "DeepMimicCompliantForceField"),
        "apply": _motion_default,
    },
    "GMTWalkStand": {
        "id_segment": "GMTWalkStand",
        "suffix": "GMTWalkStand",
        "allowed_modes": ("Hybrid", "DeepMimic", "DeepMimicForceField", "DeepMimicCompliantForceField"),
        "apply": _motion_gmt_walk_stand,
    },
    "Pour": {
        "id_segment": "Pour",
        "suffix": "Pour",
        "allowed_modes": ("Hybrid", "DeepMimic", "DeepMimicForceField", "DeepMimicCompliantForceField"),
        "apply": _motion_pour,
    },
    "AMASSPick1": {
        "id_segment": "AMASSPick1",
        "suffix": "AMASSPick1",
        "allowed_modes": ("Hybrid", "DeepMimic", "DeepMimicForceField", "DeepMimicCompliantForceField"),
        "apply": _motion_amass_pick1,
    },
    "LAFANTPoseLoop": {
        "id_segment": "LAFANTPoseLoop",
        "suffix": "LAFANTPoseLoop",
        "allowed_modes": ("Hybrid", "DeepMimic", "DeepMimicForceField", "DeepMimicCompliantForceField"),
        "apply": _motion_lafant_pose_loop,
    },
}

MODE_VARIANTS: Dict[str, Dict[str, object]] = {
    "Hybrid": {
        "id_segment": "Hybrid",
        "suffix": "Hybrid",
        "apply": _apply_mode_hybrid,
    },
    "DeepMimic": {
        "id_segment": "DeepMimic",
        "suffix": "DeepMimic",
        "apply": _apply_mode_deepmimic,
    },
    "DeepMimicForceField": {
        "id_segment": "DeepMimic-ForceField",
        "suffix": "DeepMimicForceField",
        "apply": _apply_mode_deepmimic_forcefield,
    },
    "DeepMimicCompliantForceField": {
        "id_segment": "DeepMimic-Compliant-ForceField",
        "suffix": "DeepMimicCompliantForceField",
        "apply": _apply_mode_deepmimic_compliant_forcefield,
    },
}


SUPPORTED_MOTION_VARIANTS = {
    key: {
        "id_segment": value["id_segment"],
        "modes": value["allowed_modes"],
    }
    for key, value in MOTION_VARIANTS.items()
}
SUPPORTED_MODE_VARIANTS = {
    key: {"id_segment": value["id_segment"]} for key, value in MODE_VARIANTS.items()
}

AVAILABLE_DEPLOYABLE_ENV_CFGS: Dict[Tuple[str, str], type] = {}


def _build_deployable_env_cfg_class(
    class_name: str,
    motion_apply: MotionApplyFn,
    mode_apply: ModeApplyFn,
) -> Type[TrackingEnvCfg]:
    @configclass
    class GeneratedEnvCfg(G1DeployableBaseEnvCfg):  # type: ignore
        def __post_init__(self, _motion=motion_apply, _mode=mode_apply):
            super().__post_init__()
            _motion(self)
            _mode(self)

    GeneratedEnvCfg.__name__ = class_name
    GeneratedEnvCfg.__qualname__ = class_name
    GeneratedEnvCfg.__module__ = __name__
    return GeneratedEnvCfg


for motion_key, motion_meta in MOTION_VARIANTS.items():
    motion_apply: MotionApplyFn = motion_meta["apply"]  # type: ignore[assignment]
    motion_suffix: str = motion_meta["suffix"]  # type: ignore[assignment]
    allowed_modes: Tuple[str, ...] = motion_meta["allowed_modes"]  # type: ignore[assignment]
    for mode_key, mode_meta in MODE_VARIANTS.items():
        if allowed_modes and mode_key not in allowed_modes:
            continue
        mode_apply: ModeApplyFn = mode_meta["apply"]  # type: ignore[assignment]
        mode_suffix: str = mode_meta["suffix"]  # type: ignore[assignment]

        class_name = "G1AugmentedReferenceForceTorqueControlVariableStiffness"
        if motion_suffix:
            class_name += motion_suffix
        class_name += f"{mode_suffix}DeployableMimicEnvCfg"

        generated_cfg = _build_deployable_env_cfg_class(class_name, motion_apply, mode_apply)
        globals()[class_name] = generated_cfg
        AVAILABLE_DEPLOYABLE_ENV_CFGS[(motion_key, mode_key)] = generated_cfg
