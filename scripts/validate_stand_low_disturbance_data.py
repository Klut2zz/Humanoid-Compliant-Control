"""Validate integrity, coverage, tracking quality, stability, and recovery of low-disturbance data."""

# new change
from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass, field

import torch


# new change
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset_dir", type=pathlib.Path, required=True)
parser.add_argument("--clean_success_min", type=float, default=0.98)
parser.add_argument("--perturbed_success_min", type=float, default=0.95)
parser.add_argument("--clean_wrist_mae_max_m", type=float, default=0.012)
parser.add_argument("--clean_com_mae_max_m", type=float, default=0.015)
parser.add_argument("--clean_support_q_mae_max_deg", type=float, default=2.0)
parser.add_argument("--perturbed_degradation_max", type=float, default=1.25)
parser.add_argument("--recovery_degradation_max", type=float, default=1.35)
args = parser.parse_args()


# new change
WAIST_IDS = torch.tensor([2, 5, 8], dtype=torch.long)
SUPPORT_IDS = torch.tensor([0, 1, 13, 14, 3, 4, 17, 18], dtype=torch.long)
ARM_IDS = torch.tensor(
    [11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
    dtype=torch.long,
)
WRIST_KEYPOINT_IDS = torch.tensor([2, 5], dtype=torch.long)


# new change
@dataclass
class GroupAccumulator:
    sample_count: int = 0
    full_q_ref_sum_rad: float = 0.0
    waist_q_ref_sum_rad: float = 0.0
    support_q_ref_sum_rad: float = 0.0
    support_q_target_sum_rad: float = 0.0
    arm_q_ref_sum_rad: float = 0.0
    wrist_keypoint_sum_m: float = 0.0
    com_distance_sum_m: float = 0.0
    torso_tilt_sum_rad: float = 0.0
    torso_tilt_max_rad: float = 0.0
    foot_speed_sum_mps: float = 0.0
    force_magnitude_sum_n: float = 0.0
    force_magnitude_max_n: float = 0.0

    def update(self, data: dict[str, torch.Tensor], mask: torch.Tensor) -> None:
        count = int(mask.sum().item())
        if count == 0:
            return
        q_error = (data["joint_q"] - data["reference_q"]).abs()[mask]
        target_error = (
            data["joint_q"] - data["joint_position_target"]
        ).abs()[mask]
        wrist_error = data["arm_keypoint_error_m"][mask].index_select(
            1, WRIST_KEYPOINT_IDS
        )
        com_distance = torch.linalg.vector_norm(
            data["com_support_offset_local_m"][mask], dim=-1
        )
        tilt = data["torso_tilt_rad"][mask]
        foot_speed = data["foot_xy_speed_mps"][mask]
        force_magnitude = torch.linalg.vector_norm(data["force_w"][mask], dim=-1)

        self.sample_count += count
        self.full_q_ref_sum_rad += float(q_error.mean(dim=1).sum())
        self.waist_q_ref_sum_rad += float(
            q_error.index_select(1, WAIST_IDS).mean(dim=1).sum()
        )
        self.support_q_ref_sum_rad += float(
            q_error.index_select(1, SUPPORT_IDS).mean(dim=1).sum()
        )
        self.support_q_target_sum_rad += float(
            target_error.index_select(1, SUPPORT_IDS).mean(dim=1).sum()
        )
        self.arm_q_ref_sum_rad += float(
            q_error.index_select(1, ARM_IDS).mean(dim=1).sum()
        )
        self.wrist_keypoint_sum_m += float(wrist_error.mean(dim=1).sum())
        self.com_distance_sum_m += float(com_distance.sum())
        self.torso_tilt_sum_rad += float(tilt.sum())
        self.torso_tilt_max_rad = max(
            self.torso_tilt_max_rad, float(tilt.max())
        )
        self.foot_speed_sum_mps += float(foot_speed.mean(dim=1).sum())
        self.force_magnitude_sum_n += float(force_magnitude.sum())
        self.force_magnitude_max_n = max(
            self.force_magnitude_max_n, float(force_magnitude.max())
        )

    def summary(self) -> dict[str, float | int | None]:
        if self.sample_count == 0:
            return {"sample_count": 0}
        inv = 1.0 / self.sample_count
        rad_to_deg = 180.0 / math.pi
        return {
            "sample_count": self.sample_count,
            "full_q_ref_mae_deg": self.full_q_ref_sum_rad * inv * rad_to_deg,
            "waist_q_ref_mae_deg": self.waist_q_ref_sum_rad * inv * rad_to_deg,
            "support_q_ref_mae_deg": self.support_q_ref_sum_rad * inv * rad_to_deg,
            "support_q_target_mae_deg": self.support_q_target_sum_rad
            * inv
            * rad_to_deg,
            "arm_q_ref_mae_deg": self.arm_q_ref_sum_rad * inv * rad_to_deg,
            "wrist_keypoint_mae_m": self.wrist_keypoint_sum_m * inv,
            "com_support_mae_m": self.com_distance_sum_m * inv,
            "torso_tilt_mae_deg": self.torso_tilt_sum_rad * inv * rad_to_deg,
            "torso_tilt_max_deg": self.torso_tilt_max_rad * rad_to_deg,
            "foot_xy_speed_mae_mps": self.foot_speed_sum_mps * inv,
            "force_magnitude_mean_n": self.force_magnitude_sum_n * inv,
            "force_magnitude_max_n": self.force_magnitude_max_n,
        }


# new change
def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 1.0e-12:
        return None
    return numerator / denominator


# new change
def main() -> None:
    metadata_path = args.dataset_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    chunk_paths = sorted(args.dataset_dir.glob("transitions_*.pt"))
    if not chunk_paths:
        raise RuntimeError(f"No transition chunks found in {args.dataset_dir}")

    groups = {
        "clean_cohort": GroupAccumulator(),
        "perturbed_cohort": GroupAccumulator(),
        "active_force": GroupAccumulator(),
        "recovery": GroupAccumulator(),
    }
    observed_episodes: dict[str, set[int]] = {
        "clean_cohort": set(),
        "perturbed_cohort": set(),
    }
    failed_episodes: dict[str, set[int]] = {
        "clean_cohort": set(),
        "perturbed_cohort": set(),
    }
    force_body_counts = [0 for _ in metadata["force_body_names"]]
    nonfinite_by_field: dict[str, int] = {}
    total_samples = 0
    active_samples = 0
    bad_orientation_samples = 0
    base_contact_samples = 0

    required_fields = {
        "env_id",
        "episode_index",
        "clean_cohort",
        "force_active",
        "force_body_slot",
        "force_w",
        "recovery_age_steps",
        "joint_q",
        "reference_q",
        "joint_position_target",
        "arm_keypoint_error_m",
        "com_support_offset_local_m",
        "torso_tilt_rad",
        "foot_xy_speed_mps",
        "bad_orientation",
        "base_contact",
    }

    for chunk_path in chunk_paths:
        payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1:
            raise RuntimeError(f"Unsupported schema in {chunk_path}")
        data = payload["data"]
        missing = sorted(required_fields - set(data))
        if missing:
            raise RuntimeError(f"{chunk_path} is missing fields: {missing}")
        sample_count = int(data["env_id"].shape[0])
        inconsistent = {
            name: int(value.shape[0])
            for name, value in data.items()
            if int(value.shape[0]) != sample_count
        }
        if inconsistent:
            raise RuntimeError(
                f"Inconsistent leading dimensions in {chunk_path}: {inconsistent}"
            )

        total_samples += sample_count
        clean = data["clean_cohort"].bool()
        perturbed = ~clean
        active = data["force_active"].bool()
        recovery = (
            perturbed
            & (~active)
            & (data["recovery_age_steps"] >= 0)
            & (
                data["recovery_age_steps"]
                < int(metadata["recovery_window_steps"])
            )
        )
        groups["clean_cohort"].update(data, clean)
        groups["perturbed_cohort"].update(data, perturbed)
        groups["active_force"].update(data, active)
        groups["recovery"].update(data, recovery)
        active_samples += int(active.sum())
        bad_orientation_samples += int(data["bad_orientation"].sum())
        base_contact_samples += int(data["base_contact"].sum())

        episode_keys = (
            data["env_id"].to(torch.int64) * 1_000_000
            + data["episode_index"].to(torch.int64)
        )
        for group_name, mask in (
            ("clean_cohort", clean),
            ("perturbed_cohort", perturbed),
        ):
            observed_episodes[group_name].update(
                int(value)
                for value in torch.unique(episode_keys[mask]).tolist()
            )
            failed_mask = mask & (
                data["bad_orientation"].bool() | data["base_contact"].bool()
            )
            failed_episodes[group_name].update(
                int(value)
                for value in torch.unique(episode_keys[failed_mask]).tolist()
            )

        active_slots = data["force_body_slot"][active].to(torch.int64)
        for body_slot in range(len(force_body_counts)):
            force_body_counts[body_slot] += int((active_slots == body_slot).sum())

        for name, value in data.items():
            if value.is_floating_point():
                nonfinite = int((~torch.isfinite(value)).sum())
                if nonfinite:
                    nonfinite_by_field[name] = (
                        nonfinite_by_field.get(name, 0) + nonfinite
                    )

    summaries = {name: accumulator.summary() for name, accumulator in groups.items()}
    episode_summary = {}
    for group_name in ("clean_cohort", "perturbed_cohort"):
        observed = len(observed_episodes[group_name])
        failed = len(failed_episodes[group_name])
        episode_summary[group_name] = {
            "observed_episode_count": observed,
            "failed_episode_count": failed,
            "success_rate": 1.0 - failed / observed if observed else None,
        }

    clean_summary = summaries["clean_cohort"]
    perturbed_summary = summaries["perturbed_cohort"]
    recovery_summary = summaries["recovery"]
    expected_force = float(metadata["force_max_n"]) > 0.0 and int(
        metadata["perturbed_env_count"]
    ) > 0
    integrity_pass = not nonfinite_by_field and total_samples == int(
        metadata["total_transitions"]
    )
    force_coverage_pass = (active_samples > 0) if expected_force else True
    clean_success = episode_summary["clean_cohort"]["success_rate"]
    perturbed_success = episode_summary["perturbed_cohort"]["success_rate"]

    clean_tracking_pass = True
    if clean_summary.get("sample_count", 0):
        clean_tracking_pass = (
            clean_summary["wrist_keypoint_mae_m"]
            <= args.clean_wrist_mae_max_m
            and clean_summary["com_support_mae_m"] <= args.clean_com_mae_max_m
            and clean_summary["support_q_ref_mae_deg"]
            <= args.clean_support_q_mae_max_deg
        )
    clean_success_pass = clean_success is None or clean_success >= args.clean_success_min
    perturbed_success_pass = (
        perturbed_success is None
        or perturbed_success >= args.perturbed_success_min
    )

    wrist_degradation = None
    com_degradation = None
    recovery_wrist_degradation = None
    recovery_com_degradation = None
    if clean_summary.get("sample_count", 0) and perturbed_summary.get(
        "sample_count", 0
    ):
        wrist_degradation = _safe_ratio(
            perturbed_summary["wrist_keypoint_mae_m"],
            clean_summary["wrist_keypoint_mae_m"],
        )
        com_degradation = _safe_ratio(
            perturbed_summary["com_support_mae_m"],
            clean_summary["com_support_mae_m"],
        )
    if clean_summary.get("sample_count", 0) and recovery_summary.get(
        "sample_count", 0
    ):
        recovery_wrist_degradation = _safe_ratio(
            recovery_summary["wrist_keypoint_mae_m"],
            clean_summary["wrist_keypoint_mae_m"],
        )
        recovery_com_degradation = _safe_ratio(
            recovery_summary["com_support_mae_m"],
            clean_summary["com_support_mae_m"],
        )

    degradation_pass = all(
        value is None or value <= args.perturbed_degradation_max
        for value in (wrist_degradation, com_degradation)
    )
    recovery_pass = all(
        value is None or value <= args.recovery_degradation_max
        for value in (recovery_wrist_degradation, recovery_com_degradation)
    )
    gates = {
        "integrity_pass": integrity_pass,
        "force_coverage_pass": force_coverage_pass,
        "clean_tracking_pass": clean_tracking_pass,
        "clean_success_pass": clean_success_pass,
        "perturbed_success_pass": perturbed_success_pass,
        "perturbed_degradation_pass": degradation_pass,
        "recovery_pass": recovery_pass,
    }
    overall_pass = all(gates.values())

    summary = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "metadata": metadata,
        "chunk_count": len(chunk_paths),
        "total_samples": total_samples,
        "active_force_samples": active_samples,
        "active_force_fraction": active_samples / total_samples,
        "bad_orientation_samples": bad_orientation_samples,
        "base_contact_samples": base_contact_samples,
        "nonfinite_by_field": nonfinite_by_field,
        "force_body_sample_counts": {
            name: force_body_counts[index]
            for index, name in enumerate(metadata["force_body_names"])
        },
        "groups": summaries,
        "episodes": episode_summary,
        "degradation_ratios": {
            "perturbed_wrist_vs_clean": wrist_degradation,
            "perturbed_com_vs_clean": com_degradation,
            "recovery_wrist_vs_clean": recovery_wrist_degradation,
            "recovery_com_vs_clean": recovery_com_degradation,
        },
        "thresholds": {
            "clean_success_min": args.clean_success_min,
            "perturbed_success_min": args.perturbed_success_min,
            "clean_wrist_mae_max_m": args.clean_wrist_mae_max_m,
            "clean_com_mae_max_m": args.clean_com_mae_max_m,
            "clean_support_q_mae_max_deg": args.clean_support_q_mae_max_deg,
            "perturbed_degradation_max": args.perturbed_degradation_max,
            "recovery_degradation_max": args.recovery_degradation_max,
        },
        "gates": gates,
        "overall_pass": overall_pass,
    }
    (args.dataset_dir / "quality_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "# Low-disturbance data quality report",
        "",
        f"- Verdict: **{'PASS' if overall_pass else 'FAIL'}**",
        f"- Transitions: {total_samples}",
        f"- Active-force fraction: {100.0 * active_samples / total_samples:.2f}%",
        f"- Non-finite values: {sum(nonfinite_by_field.values())}",
        "",
        "## Episode stability",
        "",
        "| cohort | observed | failed | success |",
        "|---|---:|---:|---:|",
    ]
    for group_name in ("clean_cohort", "perturbed_cohort"):
        values = episode_summary[group_name]
        success = values["success_rate"]
        report.append(
            f"| {group_name} | {values['observed_episode_count']} | "
            f"{values['failed_episode_count']} | "
            f"{'n/a' if success is None else f'{100.0 * success:.2f}%'} |"
        )
    report.extend(
        [
            "",
            "## Tracking and recovery",
            "",
            "| group | wrist mm | COM mm | support q/ref deg | arm q/ref deg | torso tilt deg | foot XY m/s |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group_name, values in summaries.items():
        if not values.get("sample_count", 0):
            report.append(f"| {group_name} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        report.append(
            f"| {group_name} | {1000.0 * values['wrist_keypoint_mae_m']:.3f} | "
            f"{1000.0 * values['com_support_mae_m']:.3f} | "
            f"{values['support_q_ref_mae_deg']:.3f} | "
            f"{values['arm_q_ref_mae_deg']:.3f} | "
            f"{values['torso_tilt_mae_deg']:.3f} | "
            f"{values['foot_xy_speed_mae_mps']:.3f} |"
        )
    report.extend(["", "## Quality gates", ""])
    report.extend(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in gates.items()
    )
    report.extend(
        [
            "",
            "The clean cohort is retained in the same rollout to detect nominal-policy regression. "
            "The perturbed cohort includes force and no-force intervals; `active_force` isolates direct response, "
            "while `recovery` covers the configured post-force window.",
        ]
    )
    (args.dataset_dir / "quality_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(
        f"[LOW_DIST_VALIDATE] verdict={'PASS' if overall_pass else 'FAIL'} "
        f"samples={total_samples} active={active_samples} "
        f"output={args.dataset_dir / 'quality_report.md'}"
    )


if __name__ == "__main__":
    main()
