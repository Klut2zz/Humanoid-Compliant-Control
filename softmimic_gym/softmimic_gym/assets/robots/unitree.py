# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`H1_CFG`: H1 humanoid robot
* :obj:`G1_CFG`: G1 humanoid robot
* :obj:`H1_2_CFG`: H1_2 humanoid robot

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import os, pathlib
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# new change
from isaaclab_assets.robots.unitree import G1_29DOF_CFG

G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(pathlib.Path(__file__).parents[0], "data/g1_minimal.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 88.,
                ".*_hip_roll_joint": 88.,
                ".*_hip_pitch_joint": 88.,
                ".*_knee_joint": 139.,
                "waist_yaw_joint": 88.,
                "waist_roll_joint": 50.,
                "waist_pitch_joint": 50.,
                },
            velocity_limit={
                ".*_hip_yaw_joint": 32.,
                ".*_hip_roll_joint": 32.,
                ".*_hip_pitch_joint": 32.,
                ".*_knee_joint": 20.,
                "waist_yaw_joint": 32.,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.,
            },
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "waist_yaw_joint": 200.0,
                "waist_roll_joint": 200.0,
                "waist_pitch_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
            },
            armature={
                ".*_hip_yaw_joint": 0.01,
                ".*_hip_roll_joint": 0.01,
                ".*_hip_pitch_joint": 0.01,
                ".*_knee_joint": 0.01,
                "waist_yaw_joint": 0.01,
                "waist_roll_joint": 0.01,
                "waist_pitch_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=50.,
            velocity_limit=37.,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness={
                ".*_ankle_pitch_joint": 20.,
                ".*_ankle_roll_joint": 20.,
            },
            damping={
                ".*_ankle_pitch_joint": 2.,
                ".*_ankle_roll_joint": 2.,
            },
            armature={
                ".*_ankle_pitch_joint": 0.01,
                ".*_ankle_roll_joint": 0.01,
            }
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*",
                ".*_elbow_.*",
                ".*_wrist_.*",
            ],
            effort_limit={
                ".*_shoulder_.*": 25.,
                ".*_elbow_.*":25.,
                ".*_wrist_.*": 25.,
            },
            velocity_limit={
                ".*_shoulder_.*": 37.,
                ".*_elbow_.*":37.,
                ".*_wrist_.*": 37.,
            },
            stiffness={
                ".*_shoulder_.*": 40.,
                ".*_elbow_.*":40.,
                ".*_wrist_.*": 5.,
            },
            damping={
                ".*_shoulder_.*": 5.,
                ".*_elbow_.*":5.,
                ".*_wrist_.*": 1.,
            },
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_wrist_.*": 0.001,
            },
        ),
    },
)

# new change
# Use IsaacLab's maintained G1 USD, while retaining the actuator behavior and
# neutral orientation that proved compatible with SoftMimic's 29DOF CSVs.  The
# official hand actuator remains assigned, but hands never enter the policy.
G1_SOFTMIMIC_29DOF_COMPAT_CFG = G1_29DOF_CFG.copy()
G1_SOFTMIMIC_29DOF_COMPAT_CFG.spawn.activate_contact_sensors = True
G1_SOFTMIMIC_29DOF_COMPAT_CFG.spawn.articulation_props.enabled_self_collisions = False
G1_SOFTMIMIC_29DOF_COMPAT_CFG.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.74),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        ".*_hip_pitch_joint": -0.20,
        ".*_knee_joint": 0.42,
        ".*_ankle_pitch_joint": -0.23,
        ".*_elbow_joint": 0.87,
        "left_shoulder_roll_joint": 0.16,
        "left_shoulder_pitch_joint": 0.35,
        "right_shoulder_roll_joint": -0.16,
        "right_shoulder_pitch_joint": 0.35,
    },
    joint_vel={".*": 0.0},
)
G1_SOFTMIMIC_29DOF_COMPAT_CFG.actuators = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ],
        effort_limit={
            ".*_hip_yaw_joint": 88.0,
            ".*_hip_roll_joint": 88.0,
            ".*_hip_pitch_joint": 88.0,
            ".*_knee_joint": 139.0,
            "waist_yaw_joint": 88.0,
            "waist_roll_joint": 50.0,
            "waist_pitch_joint": 50.0,
        },
        velocity_limit={
            ".*_hip_yaw_joint": 32.0,
            ".*_hip_roll_joint": 32.0,
            ".*_hip_pitch_joint": 32.0,
            ".*_knee_joint": 20.0,
            "waist_yaw_joint": 32.0,
            "waist_roll_joint": 37.0,
            "waist_pitch_joint": 37.0,
        },
        stiffness={
            ".*_hip_yaw_joint": 150.0,
            ".*_hip_roll_joint": 150.0,
            ".*_hip_pitch_joint": 200.0,
            ".*_knee_joint": 200.0,
            "waist_yaw_joint": 200.0,
            "waist_roll_joint": 200.0,
            "waist_pitch_joint": 200.0,
        },
        damping=5.0,
        armature=0.01,
    ),
    "feet": ImplicitActuatorCfg(
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        effort_limit=50.0,
        velocity_limit=37.0,
        stiffness=20.0,
        damping=2.0,
        armature=0.01,
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"],
        effort_limit=25.0,
        velocity_limit=37.0,
        stiffness={
            ".*_shoulder_.*": 40.0,
            ".*_elbow_.*": 40.0,
            ".*_wrist_.*": 5.0,
        },
        damping={
            ".*_shoulder_.*": 5.0,
            ".*_elbow_.*": 5.0,
            ".*_wrist_.*": 1.0,
        },
        armature={
            ".*_shoulder_.*": 0.01,
            ".*_elbow_.*": 0.01,
            ".*_wrist_.*": 0.001,
        },
    ),
    "hands": G1_29DOF_CFG.actuators["hands"].copy(),
}
