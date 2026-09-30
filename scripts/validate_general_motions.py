"""Validate the raw motion contract used by GeneralTrackingV1."""

from __future__ import annotations

import pathlib

import numpy as np


# new change
PROJECT_ROOT = pathlib.Path(__file__).absolute().parents[1]
MOTION_NAMES = ("stand.csv", "walk.csv", "tpose.csv", "boxpick.csv", "pour.csv")
EXPECTED_COLUMNS = 7 + 29 + 2


# new change
def main() -> None:
    motion_root = PROJECT_ROOT / "datasets" / "motions_csv"
    failures: list[str] = []
    total_frames = 0
    for name in MOTION_NAMES:
        path = motion_root / name
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        data = np.genfromtxt(path, delimiter=",")
        if data.ndim != 2 or data.shape[1] != EXPECTED_COLUMNS:
            failures.append(
                f"{name}: expected (*, {EXPECTED_COLUMNS}), got {data.shape}"
            )
            continue
        if not np.isfinite(data).all():
            failures.append(f"{name}: contains non-finite values")
            continue
        total_frames += data.shape[0]
        print(
            f"[OK] {name:12s} frames={data.shape[0]:5d} "
            f"duration={(data.shape[0] - 1) / 30.0:7.2f}s "
            f"max_abs_q={np.abs(data[:, 7:36]).max():.3f}rad"
        )
    if failures:
        raise SystemExit("\n".join(f"[ERROR] {failure}" for failure in failures))
    print(f"[OK] GeneralTrackingV1 motion contract: 5 files, {total_frames} frames, 38 columns.")


if __name__ == "__main__":
    main()
