#!/usr/bin/env python3
"""Collect strict finite-difference support-chain responses for root drift."""

# new change
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch


PROJECT_ROOT = Path(__file__).absolute().parents[1]
DEFAULT_POLICY = PROJECT_ROOT / (
    "logs/rsl_rl/g1_stand_coordination_v1/"
    "2026-08-20_18-57-37_stand_coordination_v1_full_v1/model_1375.jit"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "logs/stand_coordination_v1/stationary_support_control_v2/"
    "drift_controllability_fd/grid_v1"
)
TASK = (
    "Isaac-G1-StandCoordination-V2FrozenInference-ZeroForce-"
    "100Pct-29DOF-v0"
)
DEFAULT_AMPLITUDE = 0.004


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--jit_policy", type=Path, default=DEFAULT_POLICY)
parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--outcome_horizon_s", type=float, default=2.0)
parser.add_argument("--amplitude_rad", type=float, default=DEFAULT_AMPLITUDE)
parser.add_argument("--pulse_steps", type=int, default=5)
parser.add_argument("--stop_after_patterns", type=int, default=None)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()


def vector_text(values) -> str:
    return ",".join(str(value) for value in values)


def run_logged(command: list[str], log_path: Path) -> None:
    print(json.dumps({"launch": command, "log": str(log_path)}), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            if any(token in line for token in ("'output':", "Traceback", "RuntimeError", "ValueError")):
                print(line.rstrip(), flush=True)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"Child exited {code}; see {log_path}.")


def branch_command(branch: str, output: Path, vector) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts/collect_support_long_horizon_branch.py"),
        "--task", TASK,
        "--jit_policy", str(args.jit_policy),
        "--branch", branch,
        "--output", str(output),
        "--num_envs", str(args.num_envs),
        "--seed", str(args.seed),
        "--phase_offset", "0.0",
        "--warmup_steps", str(args.warmup_steps),
        "--outcome_horizon_s", str(args.outcome_horizon_s),
        "--pulse_steps", str(args.pulse_steps),
        "--teacher_limit_rad", "0.006",
        # new change: the equals form keeps a leading negative vector value
        # from being parsed as a new argparse option.
        f"--candidate_vector_rad={vector_text(vector)}",
        "--device", args.device,
        "--headless",
    ]


def merge_command(group: Path) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts/merge_support_long_horizon_branches.py"),
        "--baseline", str(group / "baseline.pt"),
        "--baseline_replay", str(group / "baseline_replay.pt"),
        "--candidate", str(group / "candidate.pt"),
        "--output", str(group / "paired.pt"),
    ]


def paired_complete(path: Path, vector) -> bool:
    if not path.exists():
        return False
    data = torch.load(path, map_location="cpu", weights_only=False)
    target = data["candidate_target_rad"]
    expected = torch.tensor(vector, dtype=target.dtype).expand_as(target)
    return bool(
        data.get("strict_pairing", False)
        and not data.get("ppo_updates", True)
        and torch.equal(target, expected)
        and max(map(float, data["snapshot_deltas"].values()), default=0.0) == 0.0
        and max(map(float, data["baseline_replay_deltas"].values()), default=0.0) == 0.0
        and int(data.get("baseline_replay_failure_mismatch", -1)) == 0
    )


def summarize(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    valid = data["valid"].bool()
    baseline = data["baseline"]
    candidate = data["candidate"]
    root_delta = (
        candidate["root_state_trajectory"][:, -1, :2]
        - baseline["root_state_trajectory"][:, -1, :2]
    )
    root_velocity_delta = (
        candidate["root_state_trajectory"][:, -1, 7:9]
        - baseline["root_state_trajectory"][:, -1, 7:9]
    )
    com_delta = (
        candidate["com_support_xy_distance_m_trajectory"].mean(1)
        - baseline["com_support_xy_distance_m_trajectory"].mean(1)
    )
    tilt_delta = (
        candidate["torso_tilt_rad_trajectory"].mean(1)
        - baseline["torso_tilt_rad_trajectory"].mean(1)
    )
    return {
        "valid": int(valid.sum()),
        "baseline_failures": int((baseline["failure"] & valid).sum()),
        "candidate_failures": int((candidate["failure"] & valid).sum()),
        "root_xy_final_delta_mean_m": root_delta[valid].mean(0).tolist(),
        "root_xy_final_delta_std_m": root_delta[valid].std(0, unbiased=False).tolist(),
        "root_xy_velocity_final_delta_mean_m_s": root_velocity_delta[valid].mean(0).tolist(),
        "com_distance_candidate_minus_baseline_mean_m": float(com_delta[valid].mean()),
        "tilt_candidate_minus_baseline_mean_rad": float(tilt_delta[valid].mean()),
        "score_candidate_minus_baseline_mean": float(data["paired_score_delta"][valid].mean()),
    }


def main() -> None:
    if args.num_envs % 8 != 0:
        raise ValueError("--num_envs must be divisible by eight motion files.")
    if not 0.0 < args.amplitude_rad <= 0.006:
        raise ValueError("--amplitude_rad must be inside (0, 0.006].")
    if args.pulse_steps < 1:
        raise ValueError("--pulse_steps must be positive.")
    # new change
    amplitude = args.amplitude_rad
    patterns = {
        "hip_pitch_pos": (amplitude, 0, amplitude, 0, 0, 0, 0, 0, 0, 0),
        "hip_pitch_neg": (-amplitude, 0, -amplitude, 0, 0, 0, 0, 0, 0, 0),
        "ankle_pitch_pos": (0, 0, 0, 0, 0, 0, amplitude, 0, amplitude, 0),
        "ankle_pitch_neg": (0, 0, 0, 0, 0, 0, -amplitude, 0, -amplitude, 0),
        "hip_roll_pos": (0, amplitude, 0, amplitude, 0, 0, 0, 0, 0, 0),
        "hip_roll_neg": (0, -amplitude, 0, -amplitude, 0, 0, 0, 0, 0, 0),
        "ankle_roll_pos": (0, 0, 0, 0, 0, 0, 0, amplitude, 0, amplitude),
        "ankle_roll_neg": (0, 0, 0, 0, 0, 0, 0, -amplitude, 0, -amplitude),
    }
    # new change
    # The legacy vectors above followed an interleaved left-body label list,
    # while collect_support_long_horizon_branch uses the canonical order
    # [L/R hip pitch, L/R knee, L/R ankle pitch, L/R hip roll,
    #  L/R ankle roll].  Keep the old table visible for provenance, then
    # replace it with correctly grouped bilateral probes.  Knee pitch is now
    # measured explicitly instead of being accidentally mixed into hip pitch.
    patterns = {
        "hip_pitch_pos": (amplitude, amplitude, 0, 0, 0, 0, 0, 0, 0, 0),
        "hip_pitch_neg": (-amplitude, -amplitude, 0, 0, 0, 0, 0, 0, 0, 0),
        "knee_pitch_pos": (0, 0, amplitude, amplitude, 0, 0, 0, 0, 0, 0),
        "knee_pitch_neg": (0, 0, -amplitude, -amplitude, 0, 0, 0, 0, 0, 0),
        "ankle_pitch_pos": (0, 0, 0, 0, amplitude, amplitude, 0, 0, 0, 0),
        "ankle_pitch_neg": (0, 0, 0, 0, -amplitude, -amplitude, 0, 0, 0, 0),
        "hip_roll_pos": (0, 0, 0, 0, 0, 0, amplitude, amplitude, 0, 0),
        "hip_roll_neg": (0, 0, 0, 0, 0, 0, -amplitude, -amplitude, 0, 0),
        "ankle_roll_pos": (0, 0, 0, 0, 0, 0, 0, 0, amplitude, amplitude),
        "ankle_roll_neg": (0, 0, 0, 0, 0, 0, 0, 0, -amplitude, -amplitude),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "SupportDriftFiniteDifferenceGridV1",
        "task": TASK,
        "jit_policy": str(args.jit_policy.resolve()),
        "num_envs": args.num_envs,
        "seed": args.seed,
        "warmup_steps": args.warmup_steps,
        "outcome_horizon_s": args.outcome_horizon_s,
        "amplitude_rad": amplitude,
        "pulse_steps": args.pulse_steps,
        "strict_three_process_reconstruction": True,
        "ppo_updates": False,
        "patterns": {},
    }
    completed = 0
    # new change
    # Baseline physics and actions are independent of candidate_target_rad;
    # the branch fingerprint and merge audit deliberately exclude only that
    # intervention tensor. Reuse one exactly replayed baseline pair across all
    # corrected +/- probes, then retain the normal zero-delta merge checks.
    shared_baseline_paths: dict[str, Path] = {}
    for name, vector in patterns.items():
        group = args.output_dir / name
        group.mkdir(parents=True, exist_ok=True)
        for branch, filename in (
            ("baseline", "baseline.pt"),
            ("baseline", "baseline_replay.pt"),
            ("candidate", "candidate.pt"),
        ):
            output = group / filename
            if not output.exists():
                # new change
                if branch == "baseline" and filename in shared_baseline_paths:
                    shutil.copy2(shared_baseline_paths[filename], output)
                    (group / filename.replace(".pt", ".console.log")).write_text(
                        "Reused strict baseline from "
                        f"{shared_baseline_paths[filename]}\n",
                        encoding="utf-8",
                    )
                else:
                    run_logged(
                        branch_command(branch, output, vector),
                        group / filename.replace(".pt", ".console.log"),
                    )
            # new change
            if branch == "baseline" and filename not in shared_baseline_paths:
                shared_baseline_paths[filename] = output
        paired = group / "paired.pt"
        if not paired_complete(paired, vector):
            run_logged(merge_command(group), group / "merge.console.log")
        if not paired_complete(paired, vector):
            raise RuntimeError(f"Strict pair audit failed: {paired}")
        manifest["patterns"][name] = {
            "vector_rad": vector,
            "paired": str(paired.resolve()),
            "summary": summarize(paired),
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"pattern_complete": name, "summary": manifest["patterns"][name]["summary"]}), flush=True)
        completed += 1
        if args.stop_after_patterns is not None and completed >= args.stop_after_patterns:
            return
    manifest["completed"] = True
    manifest["pattern_count"] = len(manifest["patterns"])
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
