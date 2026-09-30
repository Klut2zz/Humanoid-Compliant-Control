#new change
"""Stateless curriculum transform; no RNG consumption in the original sampler."""
import hashlib
import math
import torch


def curriculum_cohort(env_ids, mode):
    if mode == 'mixed':
        return torch.where(env_ids % 4 == 0, 0, torch.where(env_ids % 4 == 3, 2, 1))
    return torch.full_like(env_ids, {'zero': 0, 'low': 1, 'medium': 2}[mode])


def rear_draw(seed, env_id, event_index):
    value = hashlib.sha256(f'stand-recovery-v1:{seed}:{env_id}:{event_index}'.encode()).digest()
    return (int.from_bytes(value[:8], 'little') / 2**64,
            int.from_bytes(value[8:16], 'little') / 2**64)


def transform_peaks(peaks, stiffness, max_displacement, env_ids, event_indices, seed, mode):
    """Return sampled peaks and rear-biased flags, retaining the original IK cap.

    Half the hard-group events replace ONLY horizontal azimuth by a fresh draw
    within +/-30 degrees of world -X (nominal backward in the fixed neutral CSV).
    Other directions, vertical component, hands, timing and stiffness are retained.
    Pure medium evaluation NEVER uses the training-only direction bias.
    """
    result = peaks.clone()
    cohorts = curriculum_cohort(env_ids, mode)
    rear = torch.zeros(len(env_ids), dtype=torch.bool, device=peaks.device)
    hard = torch.where(cohorts == 2)[0]
    if len(hard):
        original = peaks[hard]
        magnitude = original.norm(dim=-1, keepdim=True)
        assert torch.all(magnitude > 1e-6), 'Base hard cohort must not be zero-force masked.'
        direction = original / magnitude
        if mode == 'mixed':
            draws = torch.tensor([rear_draw(seed, int(e), int(k)) for e, k in
                zip(env_ids[hard].cpu().tolist(), event_indices[hard].cpu().tolist())],
                device=peaks.device, dtype=peaks.dtype)
            biased = draws[:, 0] < .5
            theta = (draws[:, 1] * 2 - 1) * (math.pi / 6)
            horizontal = torch.sqrt((1-direction[:, 2].square()).clamp(min=0))
            new_direction = torch.stack((-horizontal*torch.cos(theta), horizontal*torch.sin(theta), direction[:, 2]), dim=-1)
            direction = torch.where(biased[:, None], new_direction, direction)
            rear[hard] = biased
        # Same uniform quantile: [2,4] -> [4,7]. Keep the old displacement cap.
        desired = 1.5*magnitude + 1.
        capped = torch.minimum(desired, stiffness[hard].reshape(-1, 1)*max_displacement)
        result[hard] = direction*capped
    result[cohorts == 0] = 0
    return result, rear
