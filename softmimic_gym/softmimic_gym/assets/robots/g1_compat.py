"""Strict 29DOF interface shared by the general G1 tracking stack.

The IsaacLab G1 articulation contains 29 body joints plus 14 finger joints.
SoftMimic reference CSV files contain only the 29 body joints below.  Every
action, observation, reset, reward, and diagnostic must use this exact order.
"""

from __future__ import annotations


# new change
G1_SOFTMIMIC_29_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

# new change
G1_SOFTMIMIC_NUM_JOINTS = len(G1_SOFTMIMIC_29_JOINT_NAMES)

# new change
# Canonical action-row groups used by safe stationary fine-tuning.  These are
# indices into the strict 29DOF tuple above, never Isaac articulation ids.
G1_SOFTMIMIC_WAIST_ACTION_IDS: tuple[int, ...] = (2, 5, 8)
G1_SOFTMIMIC_LEFT_ARM_ACTION_IDS: tuple[int, ...] = (11, 15, 19, 21, 23, 25, 27)
G1_SOFTMIMIC_RIGHT_ARM_ACTION_IDS: tuple[int, ...] = (12, 16, 20, 22, 24, 26, 28)
G1_SOFTMIMIC_BILATERAL_ARM_ACTION_IDS: tuple[int, ...] = tuple(
    sorted(G1_SOFTMIMIC_LEFT_ARM_ACTION_IDS + G1_SOFTMIMIC_RIGHT_ARM_ACTION_IDS)
)
G1_SOFTMIMIC_STATIONARY_UPPER_ACTION_IDS: tuple[int, ...] = tuple(
    sorted(G1_SOFTMIMIC_WAIST_ACTION_IDS + G1_SOFTMIMIC_BILATERAL_ARM_ACTION_IDS)
)
G1_SOFTMIMIC_LOWER_BODY_ACTION_IDS: tuple[int, ...] = tuple(
    action_id
    for action_id in range(G1_SOFTMIMIC_NUM_JOINTS)
    if action_id not in G1_SOFTMIMIC_STATIONARY_UPPER_ACTION_IDS
)

# new change
G1_SOFTMIMIC_KEYPOINT_BODY_NAMES: tuple[str, ...] = (
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
)


# new change
def g1_softmimic_joint_names() -> list[str]:
    """Return a mutable copy suitable for IsaacLab configuration objects."""

    return list(G1_SOFTMIMIC_29_JOINT_NAMES)


if G1_SOFTMIMIC_NUM_JOINTS != 29:
    raise RuntimeError("The SoftMimic G1 body interface must contain 29 joints.")
if len(set(G1_SOFTMIMIC_29_JOINT_NAMES)) != G1_SOFTMIMIC_NUM_JOINTS:
    raise RuntimeError("The SoftMimic G1 body-joint interface contains duplicates.")
if len(set(G1_SOFTMIMIC_KEYPOINT_BODY_NAMES)) != 14:
    raise RuntimeError("The SoftMimic G1 keypoint interface must contain 14 bodies.")
# new change
if len(G1_SOFTMIMIC_BILATERAL_ARM_ACTION_IDS) != 14:
    raise RuntimeError("The strict G1 interface must expose fourteen arm action rows.")
if len(G1_SOFTMIMIC_LOWER_BODY_ACTION_IDS) != 12:
    raise RuntimeError("The strict G1 interface must expose twelve lower-body action rows.")
