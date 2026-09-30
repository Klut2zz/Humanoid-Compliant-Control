from __future__ import annotations

import math
from dataclasses import MISSING
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import softmimic_gym.tasks.locomotion.tracking.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(0.75, 0.75, 0.75), 
            intensity=3000.0,
            angle=60.0
            ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    upper_body_joints = mdp.ReferenceCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0, 6.0),
            n_future_steps=0,
            future_dt=0.1,
            proc_ranges=mdp.ReferenceCommandCfg.ProcRanges(
                min_freq=0.8,
                max_freq=1.3,
                # min_height=0.98,
                # max_height=0.98,
                min_height=0.9,
                max_height=1.05,
                min_vel=-0.8,
                max_vel=0.8,
                min_yaw_vel=-1.0,
                max_yaw_vel=1.0,
                # min_pitch=-0.3,
                # max_pitch=0.8,
                min_pitch=-0.0,
                max_pitch=0.0,
                standing_prob=0.3,
                # standing_prob=1.0,
                stance_duration=0.6,
                swing_height=0.1,
                num_steps_to_stand=3,
                dynamic_standing_only=False,
                # min_standing_pitch=-0.3,
                # max_standing_pitch=0.8,
                min_standing_pitch=-0.0,
                max_standing_pitch=0.0,
                # min_standing_height=0.98,
                # max_standing_height=0.98,
                # min_standing_height=0.9,
                # max_standing_height=1.05,
                min_standing_height = 0.7,
                max_standing_height = 0.85,
                # random_arms=False,
                random_arms=True, 
            ),
            joint_config=mdp.ReferenceCommandCfg.JointConfig(
                num_joints=27,
                thigh_length=0.4,
                calf_length=0.4,
                robot_type="h1_2",
                left_leg_indices=mdp.ReferenceCommandCfg.JointConfig.LegIndicesConfig(
                    hip_yaw=0,
                    hip_roll=7,
                    hip_pitch=3,
                    knee=11,
                    ankle_pitch=15,
                ),
                right_leg_indices=mdp.ReferenceCommandCfg.JointConfig.LegIndicesConfig(
                    hip_yaw=1,
                    hip_roll=8,
                    hip_pitch=4,
                    knee=12,
                    ankle_pitch=16,
                ),
                left_arm_indices=mdp.ReferenceCommandCfg.JointConfig.ArmIndicesConfig(
                    shoulder_pitch=5,
                    shoulder_roll=9,
                    shoulder_yaw=13,
                    elbow=17,
                    wrist_pitch=23,
                    wrist_roll=21,
                    wrist_yaw=25,
                ),
                right_arm_indices=mdp.ReferenceCommandCfg.JointConfig.ArmIndicesConfig(
                    shoulder_pitch=6,
                    shoulder_roll=10,
                    shoulder_yaw=14,
                    elbow=18,
                    wrist_pitch=24,
                    wrist_roll=22,
                    wrist_yaw=26,
                ),
            ),
            demo_recording_path=None,
            demo_playback_mode=None,
            demo_start_range=None,
            demo_stand_height=None,
            demo_pitch_angle=None,
            demo_random_reset=None,
            ranges=None,
            debug_vis=True,
        )

    base_velocity = None


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        history_length = 3

        # sensor observations (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05)
        actions = ObsTerm(func=mdp.last_action)

        # privileged observations
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        root_height = ObsTerm(func=mdp.root_height,params={"asset_cfg": SceneEntityCfg("robot", body_names=["pelvis"])})#, noise=Unoise(n_min=-0.05, n_max=0.05))
        foot_contact = ObsTerm(func=mdp.foot_contacts, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link")})
        foot_contact_forces = ObsTerm(func=mdp.foot_contact_forces, params={"asset_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link")}, scale=1./600.)
        foot_velocities = ObsTerm(func=mdp.foot_velocities, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link")}, scale=0.05)

        # reference commands
        reference_dof_pos = ObsTerm(func=mdp.reference_dof_pos, params={"command_name": "upper_body_joints"})
        reference_xy_vel = ObsTerm(func=mdp.reference_xy_vel, params={"command_name": "upper_body_joints"})
        reference_yaw_vel = ObsTerm(func=mdp.reference_ang_vel, params={"command_name": "upper_body_joints"})
        reference_root_height = ObsTerm(func=mdp.reference_root_height, params={"command_name": "upper_body_joints"})
        reference_gravity_vector = ObsTerm(func=mdp.reference_gravity_vector, params={"command_name": "upper_body_joints"})
        reference_foot_contact = ObsTerm(func=mdp.reference_foot_contacts, params={"command_name": "upper_body_joints"})
        
        # critic only observations (preserved order)
        keypoint_error = None
        velocity_commands = None

        # external force observations (privileged)
        force_applied = None
        torque_applied = None
        desired_stiffness = None
        desired_rotational_stiffness = None
        desired_stiffness_log = None
        desired_rotational_stiffness_log = None
        desired_stiffness_inv = None
        desired_rotational_stiffness_inv = None

        # future reference commands (placeholder)
        reference_dof_pos_future = None
        reference_xy_vel_future = None
        reference_yaw_vel_future = None
        reference_root_height_future = None
        reference_gravity_vector_future = None
        reference_foot_contact_future = None

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*torso_link"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )

    randomize_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.01),
            "armature_distribution_params": (0.0, 0.2),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    
    reset_robot = EventTerm(
        func=mdp.reset_robot_state_by_command,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (-0.0, 0.0), "z": (0.0, 0.0),
                                "roll": (-0.0, 0.0), "pitch": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "jpos_range": (-0.2, 0.2),
            "jvel_range": (0.0, 0.0),
            "command_name": "upper_body_joints",
        },
    )
    reset_base = None
    reset_robot_joints = None
    
    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                body_names=["pelvis", 
                            "left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link", 
                            "right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link",
                            ]
            ),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
        },
    )
    
    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*torso_link"),
            "max_displacement": 0.02,
        },
    )

    randomize_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            # "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[
                    "torso_joint",
                    ".*_hip_yaw_joint",
                    ".*_hip_pitch_joint",
                    ".*_hip_roll_joint",
                    ".*_knee_joint",
                    ".*_ankle_.*joint",
                ]
            ),
            # "friction_distribution_params": (0.0, 0.0),
            "armature_distribution_params": (0.0, 0.6),
            # "lower_limit_distribution_params": (0.00, 0.01),
            # "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    randomize_joint_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            # "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[
                    "torso_joint",
                    ".*_hip_yaw_joint",
                    ".*_hip_pitch_joint",
                    ".*_hip_roll_joint",
                    ".*_knee_joint",
                    ".*_ankle_.*joint",
                ]
            ),
            "damping_distribution_params": (0.0, 12.0),
            "operation": "add",
            "distribution": "uniform",
        },
    )

   

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_global,
        weight=0.5,
        params={"command_name": "upper_body_joints", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_global,
        weight=0.5,
        params={"command_name": "upper_body_joints", "std": 0.5},
    )

    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-4.0e-4*0.7, params={"asset_cfg": SceneEntityCfg("robot")})
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-6.0e-7*0.7, params={"asset_cfg": SceneEntityCfg("robot")})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-6e-3*0.7)

    contact_forces = RewTerm(
        func=mdp.contact_forces,
        # weight=-6e-4, # -6e-4,
        weight=-1e-4, # -6e-4,
        params={
            # "threshold": 600., # G1 -- overridden in G1 envs
            "threshold": 700., # H12
            # "threshold": 800.,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", 
                body_names=".*ankle_roll_link"
            ),
        },
    )

    gravity_exp = RewTerm(
        func=mdp.target_orientation_exp, 
        weight=0.5, 
        params={
            "command_name": "upper_body_joints",
            },
    )

    height_exp = None
    contact_forces_upper = None

    joint_deviation_upper_body_commanded = RewTerm(
        func=mdp.joint_deviation_from_command_exp,
        weight=2.0,
        params={
            "command_name": "upper_body_joints",
            },
    )
    
    joint_deviation_contacts_commanded = None
    
    keypoint_deviation_from_command = RewTerm(
        func=mdp.keypoint_deviation_from_command_local,
        weight=2.0,
        params={
            "command_name": "upper_body_joints",
            "keypoint_body_ids": [], # Robot-specific; write in child class
            "sigma": 0.3,
            },
    )

    keypoint_orientation_deviation_from_command = RewTerm(
        func=mdp.keypoint_orientation_deviation_from_command_local,
        weight=2.0,
        params={
            "command_name": "upper_body_joints",
            "keypoint_body_ids": [], # Robot-specific; write in child class
            "sigma": 0.4,
            },
    )
    
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    feet_stumble = RewTerm(
        func=mdp.feet_slide_proportional,
        weight=-0.005,
        # weight=-0.02,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
        },
    )
    
    alive = RewTerm(
        func=mdp.is_alive,
        weight=1.5,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"), "threshold": 1.0},
    )

    # limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": math.pi / 4,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    bad_keypoints = None

    # new change
    # Disabled globally; stationary containment tasks opt in explicitly.
    stationary_support_containment = None

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = None

##
# Recorder configuration
##

class PreStepReferencePositionRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_pre_step(self):
        return "commands/reference_position", self._env.command_manager.get_command("upper_body_joints")[:, :self._env.command_manager.get_term("upper_body_joints").num_joints]

@configclass
class PreStepReferencePositionRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = PreStepReferencePositionRecorder

class PostStepTargetPositionRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_pre_step(self):
        return "commands/target_position", self._env.scene["robot"].data.joint_pos_target.clone()

@configclass
class PostStepTargetPositionRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = PostStepTargetPositionRecorder

class PreStepJointPosErrorRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_pre_step(self):
        joint_pos_error = self._env.scene["robot"].data.joint_pos_target.clone() - self._env.scene["robot"].data.joint_pos.clone()
        return "states/articulation/robot/joint_pos_error", joint_pos_error[:, :]
    
@configclass
class PreStepJointPosErrorRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = PreStepJointPosErrorRecorder

class PostStepTorsoPoseRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_post_step(self):
        torso_id = self._env.scene["robot"].find_bodies("torso_link")[0][0]
        torso_pose = self._env.scene["robot"].data.body_state_w[:, torso_id, :7].clone()
        return "states/articulation/robot/torso_pose", torso_pose[:, :]
    
@configclass
class PostStepTorsoPoseRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = PostStepTorsoPoseRecorder

class PostStepTorsoVelocityRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_post_step(self):
        torso_id = self._env.scene["robot"].find_bodies("torso_link")[0][0]
        torso_vel = self._env.scene["robot"].data.body_state_w[:, torso_id, 7:].clone()
        return "states/articulation/robot/torso_velocity", torso_vel[:, :]
    
@configclass
class PostStepTorsoVelocityRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = PostStepTorsoVelocityRecorder


class PreStepAppliedTorqueRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_pre_step(self):
        applied_torques = self._env.scene["robot"].data.applied_torque.clone()
        return "states/articulation/robot/applied_torque", applied_torques[:, :]
    
@configclass
class PreStepAppliedTorqueRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = PreStepAppliedTorqueRecorder

class PreStepTargetVelocityRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_pre_step(self):
        target_velocity_commands = torch.zeros(self._env.num_envs, 3, device=self._env.device)
        cmd_vel_xy = self._env.command_manager.get_term("upper_body_joints").root_state["root_vel"][:, 0, :2]
        cmd_vel_z = self._env.command_manager.get_term("upper_body_joints").root_state["root_ang_vel"][:, 0, 2]
        target_velocity_commands[:, :2] = cmd_vel_xy
        target_velocity_commands[:, 2] = cmd_vel_z
        return "commands/target_velocity", target_velocity_commands[:, :]
    
@configclass
class PreStepTargetVelocityRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = PreStepTargetVelocityRecorder

class PostStepAccelerationRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_post_step(self):
        joint_acc = self._env.scene["robot"].data.joint_acc.clone()
        return "states/articulation/robot/joint_acceleration", joint_acc[:, :]
    
@configclass
class PostStepAccelerationRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = PostStepAccelerationRecorder


class PostStepWBCObservationRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_post_step(self):
        wbc_observation = self._env.obs_buf["policy"]
        return "wbc_state/wbc_observation", wbc_observation[:, :]
    
@configclass
class PostStepWBCObservationRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = PostStepWBCObservationRecorder


class RewardRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_post_step(self):
        return "rewards/total_reward", self._env.reward_buf.clone()
    
@configclass
class RewardRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = RewardRecorder

@configclass
class RecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Default for now."""

    pre_step_reference_position = PreStepReferencePositionRecorderCfg()
    pre_step_target_velocity = PreStepTargetVelocityRecorderCfg()
    post_step_torso_pose = PostStepTorsoPoseRecorderCfg()
    post_step_torso_velocity = PostStepTorsoVelocityRecorderCfg()
    pre_step_joint_pos_error = PreStepJointPosErrorRecorderCfg()
    pre_step_applied_torque = PreStepAppliedTorqueRecorderCfg()
    post_step_acceleration = PostStepAccelerationRecorderCfg()
    post_step_wbc_observation = PostStepWBCObservationRecorderCfg()
    post_step_target_position = PostStepTargetPositionRecorderCfg()
    reward = RewardRecorderCfg()

##
# Environment configuration
##


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    recorders: RecorderManagerCfg = RecorderManagerCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        self.observations.critic.enable_corruption = False

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
