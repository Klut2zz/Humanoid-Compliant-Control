import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .reference_command import ReferenceCommand
from .compliance_augmented_reference_command import ComplianceAugmentedReferenceCommand
# new change
from .online_compliance_reference_command import OnlineComplianceReferenceCommand
from .fixed_stiffness_command import FixedStiffnessCommand

@configclass
class ReferenceCommandCfg(CommandTermCfg):
    """Configuration for the joint position command generator."""

    class_type: type = ReferenceCommand

    # new change
    # Preserve the released deterministic command stream by default while
    # allowing validation tasks to use event seeds that were never seen by a
    # teacher dataset or training rollout.
    command_random_seed: int = 42

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    
    @configclass
    class Ranges:
        """Uniform distribution ranges for the joint position commands."""

        min_pos: list[float] = MISSING
        """Minimum position for each joint."""
        max_pos: list[float] = MISSING
        """Maximum position for each joint."""
        
    ranges: Ranges = None
    """Distribution ranges for the joint position commands."""
    
    @configclass
    class ProcRanges:
        """Uniform distribution ranges for the procedurally generated references."""
        min_freq: float = MISSING
        max_freq: float = MISSING
        min_height: float = MISSING
        max_height: float = MISSING
        min_vel: float = MISSING
        max_vel: float = MISSING
        min_yaw_vel: float = MISSING
        max_yaw_vel: float = MISSING
        min_pitch: float = MISSING
        max_pitch: float = MISSING
        standing_prob: float = MISSING
        stance_duration: float = MISSING
        swing_height: float = MISSING
        num_steps_to_stand: int = MISSING
        dynamic_standing_only: bool = MISSING
        min_standing_pitch: float = MISSING
        max_standing_pitch: float = MISSING
        min_standing_height: float = MISSING
        max_standing_height: float = MISSING
        random_arms: bool = MISSING

    proc_ranges: ProcRanges = MISSING
    
    @configclass
    class VelocityRanges:
        """Uniform distribution ranges for the command update speed."""
            
        min_vel: list[float] = MISSING
        """Minimum velocity for each joint."""
        max_vel: list[float] = MISSING
        """Maximum velocity for each joint."""
    
    @configclass
    class JointConfig:
        """Joint configuration for the reference command."""

        num_joints: int = MISSING
        """Number of joints in the asset."""
        thigh_length: float = MISSING
        calf_length: float = MISSING   
        robot_type: str = MISSING     

        @configclass
        class LegIndicesConfig: 
            hip_yaw: int = MISSING
            hip_roll: int = MISSING
            hip_pitch: int = MISSING
            knee: int = MISSING
            ankle_pitch: int = MISSING
        
        @configclass
        class ArmIndicesConfig:
            shoulder_pitch: int = MISSING
            shoulder_roll: int = MISSING
            shoulder_yaw: int = MISSING
            elbow: int = MISSING
            wrist_pitch: int = MISSING
            wrist_roll: int = MISSING
            wrist_yaw: int = MISSING

        left_leg_indices: LegIndicesConfig = MISSING
        right_leg_indices: LegIndicesConfig = MISSING
        left_arm_indices: ArmIndicesConfig = MISSING
        right_arm_indices: ArmIndicesConfig = MISSING

    joint_config: JointConfig = MISSING

    # TODO(remove)
    n_future_steps: int = MISSING
    future_dt: float = MISSING

    demo_recording_path: str | list[str] = MISSING
    demo_playback_mode: str = MISSING
    demo_start_range: float = MISSING
    demo_stand_height: float = 0.0
    demo_pitch_angle: float = 0.0
    demo_random_reset: bool = MISSING
    demo_zero_reset: bool = False
    demo_base_policy_path: str = None
    demo_random_init: bool = False
    demo_joint_order_mode: str = "default"
    demo_upper_body_only: bool = False
    demo_zero_xy: bool = False
    demo_reset_robot_on_clip_end: bool = True
    demo_loader_class: str = "ProceduralMotionLibFromDemo"
    """Path to the demo recording file."""

    clip_length: float = None

    keypoint_body_ids: list[int] = MISSING
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_marker",
        markers={
            f"goal_{i}": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            ) for i in range(14)
        },
    )
    local_goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/local_goal_marker",
        markers={
            f"local_goal_{i}": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ) for i in range(14)
        },
    )
    local_horizon_goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/local_horizon_goal_marker",
        markers={
            f"local_horizon_goal_{i}_horizon_{h}": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, h / 20.0, 0.0)),
            ) for h in range(10) for i in range(14)
        },
    )
    current_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/current_marker",
        markers={
            f"kp_{i}": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ) for i in range(14)
        },
    )
    """The configuration for the goal pose visualization marker. Defaults to a DexCube marker."""

@configclass
class ComplianceAugmentedReferenceCommandCfg(ReferenceCommandCfg):
    """Configuration for the ComplianceAugmentedReferenceCommand."""
    
    class_type: type = ComplianceAugmentedReferenceCommand
    """The class type of the command generator."""

    override_reference: bool = True
    force_computation_mode: str = "feedforward" # ["feedforward", "recompute"]
    debug_vis_keypoints: bool = False
    enable_metrics: bool = True

    force_vector_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/force_vector_marker",
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.3, 0.3, 0.3),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            )
        }
    )
    adapted_keypoints_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/adapted_keypoints_marker",
        markers={
            f"kp": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            )# for i in range(14)
        },
    )
    recomputed_force_vector_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/recomputed_force_marker",
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.3, 0.3, 0.3),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 1.0)),
            )
        }
    )
    # torque_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/Command/torque_visualizer_marker",
    #     markers={
    #         f"cylinder": sim_utils.CylinderCfg(
    #             radius=0.05,
    #             height=0.2,
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
    #         )# for i in range(14)
    #     },
    # )
    ff_setpoint_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/ff_setpoint_visualizer",
        markers={
            f"ff_setpoint": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )
        },
    )
    # ff_plane_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/Command/ff_plane_visualizer",
    #     markers={
    #         f"ff_plane": sim_utils.CuboidCfg(
    #             size=(0.1, 0.1, 0.01),
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    #         )
    #     }
    # )
    # target_torque_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/Command/target_torque_marker",
    #     markers={
    #         "arrow": sim_utils.UsdFileCfg(
    #             usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
    #             scale=(0.3, 0.3, 0.3),
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)), # Cyan
    #         )
    #     }
    # )
    # computed_torque_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/Command/computed_torque_marker",
    #     markers={
    #         "arrow": sim_utils.UsdFileCfg(
    #             usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
    #             scale=(0.3, 0.3, 0.3),
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.75, 0.4)), # Magenta
    #         )
    #     }
    # )


# new change
@configclass
class OnlineComplianceReferenceCommandCfg(ReferenceCommandCfg):
    """GPU-resident on-policy disturbance and compliant-target generator."""

    class_type: type = OnlineComplianceReferenceCommand
    force_body_names: tuple[str, ...] = (
        "torso_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )
    force_range_n: tuple[float, float] = (2.0, 8.0)
    stiffness_range_n_per_m: tuple[float, float] = (250.0, 600.0)
    max_displacement_m: float = 0.025
    # new change
    # Optional human-readable label used only by recorded quality previews.
    difficulty_label: str = ""
    vertical_direction_scale: float = 0.25
    warmup_time_s: float = 1.0
    active_time_s: float = 0.8
    recovery_time_s: float = 1.2

    # new change
    # Event timing follows the original offline augmentation: random quiet
    # period, linear ramp-up, hold, and linear ramp-down.  The fixed timing
    # fields above are retained for backward compatibility with the first
    # Cartesian-only prototype and are no longer used by the full generator.
    wait_time_range_s: tuple[float, float] = (0.5, 1.5)
    ramp_time_range_s: tuple[float, float] = (0.2, 0.6)
    hold_time_range_s: tuple[float, float] = (0.5, 1.0)
    # new change
    # Keep the released tasks bit-for-bit linear unless an independent A/B
    # task explicitly selects the zero-slope half-cosine envelope.
    force_envelope_profile: str = "linear"

    # new change
    # GPU damped-least-squares task weights mirror the relative priorities of
    # compliant_motion_augmentation/ik_solver.py.  They are deliberately
    # conservative for the first low-disturbance online curriculum.
    ik_damping: float = 0.04
    posture_weight: float = 1.0e-3
    active_link_position_weight: float = 5.0
    foot_position_weight: float = 2.5
    foot_orientation_weight: float = 0.5
    keypoint_position_weight: float = 1.0e-2
    keypoint_orientation_weight: float = 1.0e-3
    com_xy_weight: float = 0.1
    pelvis_orientation_weight: float = 0.03
    waist_knee_posture_weight: float = 0.01

    # new change
    arm_delta_limit_rad: float = 0.08
    waist_delta_limit_rad: float = 0.015
    hip_delta_limit_rad: float = 0.008
    knee_delta_limit_rad: float = 0.006
    ankle_delta_limit_rad: float = 0.006
    joint_velocity_limit_fraction: float = 0.50
    max_active_link_residual_m: float = 0.006
    max_active_link_relative_residual: float = 0.50
    min_active_direction_cosine: float = 0.80
    max_foot_motion_m: float = 0.008
    max_com_residual_m: float = 0.025
    min_feasible_scale: float = 0.25

    # new change
    # ArmLocalIKV2 compatibility switches.  Defaults preserve the first
    # whole-body online prototype; the independent V2 task enables these to
    # reproduce the offline Mink structure for wrist disturbances.
    arm_local_ik_enabled: bool = False
    cop_aware_com_target_enabled: bool = True
    exclude_active_link_from_keypoint_task: bool = False
    physical_force_enabled: bool = True
    force_body_context_size: int = 5
    force_body_context_slots: tuple[int, ...] = ()

    # new change
    # Optional reference-state kinematics.  ``simulation`` preserves the first
    # prototype, whose Jacobians come from the live articulation.  The MJCF
    # backend evaluates FK/Jacobians at q_ref and is therefore suitable for
    # generating a reference independently of policy tracking error.
    reference_kinematics_backend: str = "simulation"
    reference_mjcf_path: str = ""
    reference_ik_iterations: int = 8

    # new change
    # Dedicated online-quality visualization.  These are separate from the
    # legacy ReferenceCommand markers so nominal, adapted, and measured points
    # remain visually distinguishable in recorded validation videos.
    online_nominal_keypoints_visualizer_cfg: VisualizationMarkersCfg = (
        VisualizationMarkersCfg(
            prim_path="/Visuals/OnlineCompliance/nominal_keypoints",
            markers={
                "nominal": sim_utils.SphereCfg(
                    radius=0.022,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.05, 0.05)
                    ),
                )
            },
        )
    )
    online_adapted_keypoints_visualizer_cfg: VisualizationMarkersCfg = (
        VisualizationMarkersCfg(
            prim_path="/Visuals/OnlineCompliance/adapted_keypoints",
            markers={
                "adapted": sim_utils.SphereCfg(
                    radius=0.030,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.85, 0.0)
                    ),
                )
            },
        )
    )
    online_actual_keypoints_visualizer_cfg: VisualizationMarkersCfg = (
        VisualizationMarkersCfg(
            prim_path="/Visuals/OnlineCompliance/actual_keypoints",
            markers={
                "actual": sim_utils.SphereCfg(
                    radius=0.016,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.35, 1.0)
                    ),
                )
            },
        )
    )
    online_force_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/OnlineCompliance/force",
        markers={
            "force": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                # new change
                # Visualization-only enlargement for recorded quality gates.
                scale=(0.42, 0.18, 0.18),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.45, 0.0)
                ),
            )
        },
    )
    online_setpoint_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/OnlineCompliance/accepted_setpoint",
        markers={
            "accepted": sim_utils.SphereCfg(
                radius=0.040,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.1)
                ),
            )
        },
    )
    online_rejected_setpoint_visualizer_cfg: VisualizationMarkersCfg = (
        VisualizationMarkersCfg(
            prim_path="/Visuals/OnlineCompliance/rejected_setpoint",
            markers={
                "rejected": sim_utils.SphereCfg(
                    radius=0.045,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.9, 0.0, 0.9)
                    ),
                )
            },
        )
    )
    online_com_support_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/OnlineCompliance/com_support",
        markers={
            "com": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.7)
                ),
            ),
            "support": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 1.0)
                ),
            ),
        },
    )


@configclass
class FixedStiffnessCommandCfg(CommandTermCfg):
    """Configuration for the ObstacleMotionCommand."""

    class_type: type = FixedStiffnessCommand

    desired_stiffness: float = MISSING
    desired_rotational_stiffness: float = MISSING
    forcefield_stiffness: float = MISSING
    forcefield_rotational_stiffness: float = MISSING
