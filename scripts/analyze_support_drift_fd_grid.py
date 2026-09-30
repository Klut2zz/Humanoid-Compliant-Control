#!/usr/bin/env python3
"""Build a centered finite-difference response report from strict pairs."""

# new change
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


AXES = ("hip_pitch", "ankle_pitch", "hip_roll", "ankle_roll")


def load_pair(root: Path, name: str) -> dict:
    data = torch.load(root / name / "paired.pt", map_location="cpu", weights_only=False)
    if not data.get("strict_pairing", False):
        raise RuntimeError(f"Not a strict pair: {name}")
    if max(map(float, data["snapshot_deltas"].values()), default=0.0) != 0.0:
        raise RuntimeError(f"Snapshot mismatch: {name}")
    if max(map(float, data["baseline_replay_deltas"].values()), default=0.0) != 0.0:
        raise RuntimeError(f"Replay mismatch: {name}")
    return data


def treatment(data: dict, key: str, final: bool = False) -> torch.Tensor:
    candidate = data["candidate"][key]
    baseline = data["baseline"][key]
    delta = candidate - baseline
    return delta[:, -1] if final else delta.mean(dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.grid / "manifest.json").read_text())
    amplitude = float(manifest["amplitude_rad"])
    response = {}
    for axis in AXES:
        positive = load_pair(args.grid, f"{axis}_pos")
        negative = load_pair(args.grid, f"{axis}_neg")
        valid = positive["valid"].bool() & negative["valid"].bool()
        root_pos = treatment(positive, "root_state_trajectory", final=True)[:, :2]
        root_neg = treatment(negative, "root_state_trajectory", final=True)[:, :2]
        velocity_pos = treatment(positive, "root_state_trajectory", final=True)[:, 7:9]
        velocity_neg = treatment(negative, "root_state_trajectory", final=True)[:, 7:9]
        com_pos = treatment(positive, "com_support_xy_distance_m_trajectory")
        com_neg = treatment(negative, "com_support_xy_distance_m_trajectory")
        tilt_pos = treatment(positive, "torso_tilt_rad_trajectory")
        tilt_neg = treatment(negative, "torso_tilt_rad_trajectory")

        def centered(pos, neg):
            return (pos - neg) / 2.0

        root = centered(root_pos, root_neg)[valid]
        velocity = centered(velocity_pos, velocity_neg)[valid]
        com = centered(com_pos, com_neg)[valid]
        tilt = centered(tilt_pos, tilt_neg)[valid]
        failure_pos = positive["paired_failure_delta"][valid]
        failure_neg = negative["paired_failure_delta"][valid]
        response[axis] = {
            "valid": int(valid.sum()),
            "root_xy_centered_mean_m": root.mean(0).tolist(),
            "root_xy_centered_std_m": root.std(0, unbiased=False).tolist(),
            "root_xy_sensitivity_m_per_rad": (root.mean(0) / amplitude).tolist(),
            "root_xy_velocity_centered_mean_m_s": velocity.mean(0).tolist(),
            "root_xy_velocity_sensitivity_m_s_per_rad": (velocity.mean(0) / amplitude).tolist(),
            "com_centered_mean_m": float(com.mean()),
            "tilt_centered_mean_rad": float(tilt.mean()),
            "candidate_caused_failures_pos": int((failure_pos > 0).sum()),
            "candidate_caused_failures_neg": int((failure_neg > 0).sum()),
            "mean_root_response_norm_m": float(torch.linalg.vector_norm(root, dim=1).mean()),
            "mean_root_response_to_baseline_drift_ratio": float(
                torch.linalg.vector_norm(root, dim=1).mean()
                / positive["baseline"]["root_state_trajectory"][:, -1, :2]
                .sub(positive["baseline"]["root_state_trajectory"][:, 0, :2])
                .norm(dim=1)[valid]
                .mean()
                .clamp_min(1.0e-6)
            ),
        }

    report = {
        "schema": "SupportDriftFiniteDifferenceResponseV1",
        "grid": str(args.grid.resolve()),
        "amplitude_rad": amplitude,
        "pulse_steps": int(manifest.get("pulse_steps", 5)),
        "strict_pairing": True,
        "ppo_updates": False,
        "response": response,
    }
    (args.grid / "response_matrix.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Support drift finite-difference response",
        "",
        f"- Amplitude: {amplitude * 1000:.1f} mrad",
        f"- Pulse: {report['pulse_steps']} steps",
        "- Strict snapshot/replay audit: PASS",
        "- PPO updates: none",
        "",
        "| axis | dx sensitivity (m/rad) | dy sensitivity (m/rad) | mean response (mm) | response / baseline drift | caused failures (+/-) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for axis, values in response.items():
        sensitivity = values["root_xy_sensitivity_m_per_rad"]
        lines.append(
            f"| {axis} | {sensitivity[0]:.4f} | {sensitivity[1]:.4f} | "
            f"{1000 * values['mean_root_response_norm_m']:.3f} | "
            f"{values['mean_root_response_to_baseline_drift_ratio']:.4f} | "
            f"{values['candidate_caused_failures_pos']}/{values['candidate_caused_failures_neg']} |"
        )
    lines.extend(
        [
            "",
            "A five-step pulse is considered too weak when its centered response is small relative to baseline drift and no stable axis-dominant mapping emerges. Test a longer bounded pulse before designing the feedback law.",
        ]
    )
    (args.grid / "RESPONSE_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
