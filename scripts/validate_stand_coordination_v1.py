"""Validate StandCoordinationV1 data, dynamics, limits, and FK continuity."""

from __future__ import annotations

import json
import pathlib

import mujoco
import numpy as np
import torch

from softmimic_deploy.src.motion_lib.keypoint_extractor import KeypointExtractor


# new change
PROJECT_ROOT = pathlib.Path(__file__).absolute().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "datasets" / "motions_csv" / "stand_coordination_v1"
MANIFEST_PATH = REFERENCE_ROOT / "manifest.json"
EXPECTED_COLUMNS = 7 + 29 + 2
MAX_JOINT_SPEED_RAD_S = 0.50
MAX_JOINT_ACCEL_RAD_S2 = 0.80
MAX_KEYPOINT_SPEED_M_S = 0.75


def _joint_limits(joint_names: tuple[str, ...]) -> np.ndarray:
    extractor = KeypointExtractor(robot_type="g1")
    limits = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(extractor.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo FK model is missing joint {name!r}.")
        limits.append(extractor.model.jnt_range[joint_id])
    return np.asarray(limits, dtype=np.float64)


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    joint_names = tuple(manifest["joint_order"])
    reindex = np.asarray(manifest["reference_reindex"], dtype=np.int64)
    if sorted(reindex.tolist()) != list(range(29)):
        raise ValueError("reference_reindex must be a permutation of 0..28.")
    fps = float(manifest["fps"])
    if manifest.get("reference_scale") != 1.0:
        raise ValueError("StandCoordinationV1 must use an unscaled 100% reference.")
    limits = _joint_limits(joint_names)
    baseline = np.asarray(manifest["neutral_row"], dtype=np.float64)
    if baseline.shape != (EXPECTED_COLUMNS,):
        raise ValueError(f"Manifest neutral row must have 38 values, got {baseline.shape}.")
    active_ids = [
        index
        for index, name in enumerate(joint_names)
        if name.startswith("waist_") or any(token in name for token in ("shoulder", "elbow", "wrist"))
    ]
    invariant_ids = [index for index in range(29) if index not in active_ids]
    failures: list[str] = []
    extractor = KeypointExtractor(robot_type="g1")

    for file_info in manifest["files"]:
        path = REFERENCE_ROOT / file_info["name"]
        data = np.loadtxt(path, delimiter=",")
        raw_q = data[:, 7:36]
        q = raw_q[:, reindex]
        if data.shape != (int(file_info["frames"]), EXPECTED_COLUMNS):
            failures.append(f"{path.name}: unexpected shape {data.shape}")
            continue
        if not np.isfinite(data).all():
            failures.append(f"{path.name}: contains non-finite values")
            continue
        if not np.allclose(data[:, :7], baseline[:7], atol=1.0e-8):
            failures.append(f"{path.name}: root pose changed")
        baseline_q = baseline[7:36][reindex]
        if not np.allclose(q[:, invariant_ids], baseline_q[invariant_ids], atol=1.0e-8):
            failures.append(f"{path.name}: lower-body invariant changed")
        if not np.allclose(data[:, 36:38], 1.0, atol=1.0e-8):
            failures.append(f"{path.name}: both feet must remain in contact")
        if not np.allclose(data[[0, -1]], baseline, atol=1.0e-8):
            failures.append(f"{path.name}: endpoints must be the neutral stand row")
        # new change
        # Validate the exact 100% plateaus, not only aggregate ranges.  This
        # catches accidental scaling or a reversed canonical/CSV permutation.
        for label, sample_time, target in (
            ("A", 6.0, file_info["target_a_delta_rad"]),
            ("B", 16.0, file_info["target_b_delta_rad"]),
        ):
            expected = baseline_q.copy()
            for joint_name, delta in target.items():
                expected[joint_names.index(joint_name)] += float(delta)
            actual = q[int(round(sample_time * fps))]
            if not np.allclose(actual, expected, atol=1.0e-8):
                failures.append(f"{path.name}: target {label} is not the exact 100% plateau")
        if np.any(q < limits[:, 0] - 1.0e-8) or np.any(q > limits[:, 1] + 1.0e-8):
            failures.append(f"{path.name}: exceeds MuJoCo joint limits")

        velocity = np.diff(q, axis=0) * fps
        acceleration = np.diff(velocity, axis=0) * fps
        max_speed = float(np.abs(velocity).max(initial=0.0))
        max_accel = float(np.abs(acceleration).max(initial=0.0))
        if max_speed > MAX_JOINT_SPEED_RAD_S:
            failures.append(f"{path.name}: max joint speed {max_speed:.3f}rad/s")
        if max_accel > MAX_JOINT_ACCEL_RAD_S2:
            failures.append(f"{path.name}: max joint acceleration {max_accel:.3f}rad/s^2")

        sample_ids = np.arange(0, data.shape[0], 5)
        keypoints, _, _ = extractor.compute_keypoints_batch(
            torch.from_numpy(data[sample_ids, :3]).float(),
            torch.from_numpy(data[sample_ids, 3:7]).float(),
            torch.from_numpy(q[sample_ids]).float(),
        )
        keypoint_step = torch.linalg.vector_norm(keypoints[1:] - keypoints[:-1], dim=-1)
        max_keypoint_speed = float(keypoint_step.max().item() * fps / 5.0)
        if max_keypoint_speed > MAX_KEYPOINT_SPEED_M_S:
            failures.append(f"{path.name}: max keypoint speed {max_keypoint_speed:.3f}m/s")
        print(
            f"[OK] {path.name:37s} frames={len(data):3d} "
            f"q_span={np.ptp(q, axis=0).max():.3f}rad "
            f"q_speed={max_speed:.3f}rad/s q_accel={max_accel:.3f}rad/s^2 "
            f"kp_speed={max_keypoint_speed:.3f}m/s"
        )

    if failures:
        raise SystemExit("\n".join(f"[ERROR] {failure}" for failure in failures))
    print(
        f"[OK] StandCoordinationV1: {len(manifest['files'])} unscaled 100% references, "
        "strict 29DOF order, static root/lower body, continuous MuJoCo FK."
    )


if __name__ == "__main__":
    main()
