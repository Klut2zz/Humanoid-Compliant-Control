"""Generate the deterministic 100% StandCoordinationV1 reference set.

The references use the symmetric neutral pose from
``G1_SOFTMIMIC_29DOF_COMPAT_CFG``.  Smooth, bounded waist/arm excursions are
added around that exact 29DOF posture.  The amplitudes below define the full
100-percent task; the environment applies no runtime motion scaling.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np


# new change
PROJECT_ROOT = pathlib.Path(__file__).absolute().parents[1]
LEGACY_STAND_PATH = PROJECT_ROOT / "datasets" / "motions_csv" / "stand.csv"
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "motions_csv" / "stand_coordination_v1"
FPS = 30.0
DURATION_S = 24.0
NUM_FRAMES = int(DURATION_S * FPS) + 1

# new change
JOINT_NAMES = (
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
JOINT_INDEX = {name: index for index, name in enumerate(JOINT_NAMES)}
# new change
# ReferenceCommand returns ``raw_csv_q[:, REINDEX]`` in canonical policy order.
# Consequently generation must write canonical_q[i] into raw_csv_q[REINDEX[i]].
REFERENCE_REINDEX = np.asarray(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)
CSV_JOINT_NAMES = tuple(JOINT_NAMES[index] for index in np.argsort(REFERENCE_REINDEX))
WAIST_AND_ARM_JOINTS = {
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    *(name for name in JOINT_NAMES if any(token in name for token in ("shoulder", "elbow", "wrist"))),
}

# new change
# This exactly mirrors G1_SOFTMIMIC_29DOF_COMPAT_CFG.init_state.  Keeping the
# row explicit makes reference generation independent of Isaac Sim imports.
NEUTRAL_ROOT_POS = (0.0, 0.0, 0.74)
NEUTRAL_ROOT_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)
NEUTRAL_JOINT_POS = {
    "left_hip_pitch_joint": -0.20,
    "right_hip_pitch_joint": -0.20,
    "waist_yaw_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "waist_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_knee_joint": 0.42,
    "right_knee_joint": 0.42,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_pitch_joint": 0.35,
    "left_ankle_pitch_joint": -0.23,
    "right_ankle_pitch_joint": -0.23,
    "left_shoulder_roll_joint": 0.16,
    "right_shoulder_roll_joint": -0.16,
    "left_ankle_roll_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.87,
    "right_elbow_joint": 0.87,
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


# new change
# Each motion visits target A, returns to neutral, visits target B, and returns
# again.  Deltas are radians relative to the exact stand.csv reference row.
MOTION_TARGETS: dict[str, tuple[dict[str, float], dict[str, float]]] = {
    "00_neutral": ({}, {}),
    "01_bilateral_sagittal": (
        {
            "waist_pitch_joint": -0.06,
            "left_shoulder_pitch_joint": 0.30,
            "right_shoulder_pitch_joint": 0.30,
            "left_elbow_joint": 0.15,
            "right_elbow_joint": 0.15,
            "left_wrist_pitch_joint": -0.12,
            "right_wrist_pitch_joint": 0.12,
        },
        {
            "waist_pitch_joint": 0.04,
            "left_shoulder_pitch_joint": -0.18,
            "right_shoulder_pitch_joint": -0.18,
            "left_elbow_joint": -0.10,
            "right_elbow_joint": -0.10,
            "left_wrist_pitch_joint": 0.08,
            "right_wrist_pitch_joint": -0.08,
        },
    ),
    "02_bilateral_lateral": (
        {
            "waist_pitch_joint": 0.02,
            "left_shoulder_roll_joint": -0.45,
            "right_shoulder_roll_joint": 0.45,
            "left_elbow_joint": 0.12,
            "right_elbow_joint": 0.12,
        },
        {
            "waist_yaw_joint": -0.04,
            "left_shoulder_roll_joint": -0.28,
            "right_shoulder_roll_joint": 0.28,
            "left_shoulder_yaw_joint": 0.15,
            "right_shoulder_yaw_joint": -0.15,
        },
    ),
    "03_unilateral_counterbalance": (
        {
            "waist_yaw_joint": -0.05,
            "waist_roll_joint": 0.07,
            "left_shoulder_pitch_joint": 0.25,
            "left_shoulder_roll_joint": -0.35,
            "left_shoulder_yaw_joint": -0.20,
            "left_elbow_joint": 0.25,
            "left_wrist_pitch_joint": -0.15,
            "right_shoulder_pitch_joint": -0.08,
        },
        {
            "waist_yaw_joint": 0.05,
            "waist_roll_joint": -0.07,
            "right_shoulder_pitch_joint": 0.25,
            "right_shoulder_roll_joint": 0.35,
            "right_shoulder_yaw_joint": 0.20,
            "right_elbow_joint": 0.25,
            "right_wrist_pitch_joint": 0.15,
            "left_shoulder_pitch_joint": -0.08,
        },
    ),
    "04_axial_twist": (
        {
            "waist_yaw_joint": 0.10,
            "left_shoulder_yaw_joint": -0.15,
            "right_shoulder_yaw_joint": -0.15,
            "left_elbow_joint": 0.10,
            "right_elbow_joint": 0.10,
        },
        {
            "waist_yaw_joint": -0.10,
            "left_shoulder_yaw_joint": 0.15,
            "right_shoulder_yaw_joint": 0.15,
            "left_elbow_joint": -0.08,
            "right_elbow_joint": -0.08,
        },
    ),
    "05_roll_coordination": (
        {
            "waist_roll_joint": 0.08,
            "left_shoulder_roll_joint": -0.25,
            "right_shoulder_roll_joint": -0.10,
            "left_shoulder_pitch_joint": 0.10,
            "right_shoulder_pitch_joint": -0.05,
        },
        {
            "waist_roll_joint": -0.08,
            "left_shoulder_roll_joint": 0.10,
            "right_shoulder_roll_joint": 0.25,
            "left_shoulder_pitch_joint": -0.05,
            "right_shoulder_pitch_joint": 0.10,
        },
    ),
    "06_pitch_coordination": (
        {
            "waist_pitch_joint": 0.08,
            "left_shoulder_pitch_joint": -0.20,
            "right_shoulder_pitch_joint": -0.20,
            "left_elbow_joint": 0.12,
            "right_elbow_joint": 0.12,
        },
        {
            "waist_pitch_joint": -0.08,
            "left_shoulder_pitch_joint": 0.20,
            "right_shoulder_pitch_joint": 0.20,
            "left_elbow_joint": -0.08,
            "right_elbow_joint": -0.08,
        },
    ),
    "07_combined_coordination": (
        {
            "waist_yaw_joint": 0.08,
            "waist_roll_joint": 0.05,
            "waist_pitch_joint": 0.05,
            "left_shoulder_pitch_joint": 0.22,
            "left_shoulder_roll_joint": -0.30,
            "left_shoulder_yaw_joint": -0.16,
            "left_elbow_joint": 0.22,
            "left_wrist_roll_joint": -0.12,
            "right_shoulder_pitch_joint": -0.10,
            "right_shoulder_roll_joint": -0.08,
        },
        {
            "waist_yaw_joint": -0.08,
            "waist_roll_joint": -0.05,
            "waist_pitch_joint": -0.05,
            "right_shoulder_pitch_joint": 0.22,
            "right_shoulder_roll_joint": 0.30,
            "right_shoulder_yaw_joint": 0.16,
            "right_elbow_joint": 0.22,
            "right_wrist_roll_joint": 0.12,
            "left_shoulder_pitch_joint": -0.10,
            "left_shoulder_roll_joint": 0.08,
        },
    ),
}


def _minimum_jerk(alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha, 0.0, 1.0)
    return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5


def _pulse(times: np.ndarray, start: float, rise_end: float, hold_end: float, end: float) -> np.ndarray:
    value = np.zeros_like(times)
    rising = (times >= start) & (times < rise_end)
    value[rising] = _minimum_jerk((times[rising] - start) / (rise_end - start))
    value[(times >= rise_end) & (times < hold_end)] = 1.0
    falling = (times >= hold_end) & (times < end)
    value[falling] = 1.0 - _minimum_jerk((times[falling] - hold_end) / (end - hold_end))
    return value


def _delta_vector(values: dict[str, float]) -> np.ndarray:
    unknown = set(values) - set(JOINT_INDEX)
    if unknown:
        raise KeyError(f"Unknown StandCoordinationV1 joints: {sorted(unknown)}")
    vector = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    for name, value in values.items():
        if name not in WAIST_AND_ARM_JOINTS:
            raise ValueError(f"Only waist/arm joints may vary, got {name!r}.")
        vector[JOINT_INDEX[name]] = value
    return vector


def _canonical_to_csv(canonical_q: np.ndarray) -> np.ndarray:
    raw_q = np.empty_like(canonical_q)
    raw_q[..., REFERENCE_REINDEX] = canonical_q
    return raw_q


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    times = np.arange(NUM_FRAMES, dtype=np.float64) / FPS
    pulse_a = _pulse(times, 2.0, 5.5, 7.5, 10.5)
    pulse_b = _pulse(times, 12.0, 15.5, 17.5, 20.5)
    neutral_q = np.asarray(
        [NEUTRAL_JOINT_POS[name] for name in JOINT_NAMES], dtype=np.float64
    )
    neutral_csv_q = _canonical_to_csv(neutral_q)
    baseline = np.concatenate(
        (
            np.asarray(NEUTRAL_ROOT_POS, dtype=np.float64),
            np.asarray(NEUTRAL_ROOT_QUAT_XYZW, dtype=np.float64),
            neutral_csv_q,
            np.ones(2, dtype=np.float64),
        )
    )
    generated_files: list[dict[str, object]] = []

    for motion_name, (target_a, target_b) in MOTION_TARGETS.items():
        data = np.repeat(baseline[None, :], NUM_FRAMES, axis=0)
        canonical_q = neutral_q[None, :] + (
            pulse_a[:, None] * _delta_vector(target_a)[None, :]
            + pulse_b[:, None] * _delta_vector(target_b)[None, :]
        )
        data[:, 7:36] = _canonical_to_csv(canonical_q)
        path = OUTPUT_ROOT / f"{motion_name}.csv"
        np.savetxt(path, data, delimiter=",", fmt="%.9f")
        generated_files.append(
            {
                "name": path.name,
                "frames": NUM_FRAMES,
                "duration_s": DURATION_S,
                "target_a_delta_rad": target_a,
                "target_b_delta_rad": target_b,
            }
        )
        print(f"[GENERATED] {path.relative_to(PROJECT_ROOT)} frames={NUM_FRAMES}")

    manifest = {
        "name": "StandCoordinationV1",
        "version": 1,
        "fps": FPS,
        "duration_s": DURATION_S,
        "reference_scale": 1.0,
        "baseline_source": "G1_SOFTMIMIC_29DOF_COMPAT_CFG.init_state",
        "neutral_row": baseline.tolist(),
        "legacy_stand_csv": str(LEGACY_STAND_PATH.relative_to(PROJECT_ROOT)),
        "legacy_stand_sha256": hashlib.sha256(LEGACY_STAND_PATH.read_bytes()).hexdigest(),
        "joint_order": JOINT_NAMES,
        "csv_joint_order": CSV_JOINT_NAMES,
        "reference_reindex": REFERENCE_REINDEX.tolist(),
        "neutral_canonical_q_rad": neutral_q.tolist(),
        "invariants": {
            "root_pose": "G1_SOFTMIMIC_29DOF_COMPAT_CFG neutral pose",
            "lower_body_joints": "G1_SOFTMIMIC_29DOF_COMPAT_CFG neutral pose",
            "foot_contacts": [1.0, 1.0],
        },
        "timing_s": {
            "neutral_hold_1": [0.0, 2.0],
            "target_a_transition": [2.0, 5.5],
            "target_a_hold": [5.5, 7.5],
            "target_a_return": [7.5, 10.5],
            "neutral_hold_2": [10.5, 12.0],
            "target_b_transition": [12.0, 15.5],
            "target_b_hold": [15.5, 17.5],
            "target_b_return": [17.5, 20.5],
            "terminal_neutral_hold": [20.5, 24.0],
        },
        "files": generated_files,
    }
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[GENERATED] {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
