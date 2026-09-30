"""Name-safe keypoint resolution helpers."""

from __future__ import annotations

from collections.abc import Sequence


# new change
def resolve_keypoint_body_ids(asset, keypoint_body_ids: Sequence[int | str] | None) -> list[int]:
    """Resolve body names in the configured order, retaining legacy integer ids."""

    if not keypoint_body_ids:
        return []
    if isinstance(keypoint_body_ids[0], str):
        body_ids, body_names = asset.find_bodies(list(keypoint_body_ids), preserve_order=True)
        if list(body_names) != list(keypoint_body_ids):
            raise ValueError(
                "Keypoint body mapping did not resolve exactly: "
                f"requested={list(keypoint_body_ids)}, resolved={list(body_names)}."
            )
        return [int(body_id) for body_id in body_ids]
    return [int(body_id) for body_id in keypoint_body_ids]
