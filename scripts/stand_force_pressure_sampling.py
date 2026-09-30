#new change
"""Pressure-only peak remapping with read-only, pre-cap RNG telemetry."""
import torch


class PressureTelemetryMixin:
    """Wrap the existing medium sampler without additional random draws.

    The parent's independent per-env sampler draws 2--4 N and remaps to 4--7 N.
    Capture the ORIGINAL magnitude draw, before either displacement/IK cap.
    """
    pressure_force_range_n = (7., 10.)

    def _uniform(self, count, value_range):
        value = super()._uniform(count, value_range)
        draws = getattr(self, '_pressure_capture_draws', None)
        if draws is not None and tuple(value_range) == (2., 4.):
            draws.append(value.detach().clone())
        return value

    def _sample_events(self, env_ids):
        if env_ids.numel() == 0:
            return super()._sample_events(env_ids)
        assert getattr(self, '_pressure_capture_draws', None) is None
        self._pressure_capture_draws = []
        try:
            super()._sample_events(env_ids)
            draws = self._pressure_capture_draws
        finally:
            self._pressure_capture_draws = None
        raw = torch.cat(draws).reshape(-1)
        assert raw.numel() == env_ids.numel(), 'Exactly one force magnitude draw per event.'
        if not hasattr(self, 'pressure_requested_peak_world'):
            self.pressure_requested_peak_world = torch.zeros_like(self._sampled_peak_force_world)
            self.pressure_requested_peak_n = torch.zeros_like(self._sampled_peak_force_world[:, 0])
            self.pressure_displacement_cap_scale = torch.zeros_like(self.pressure_requested_peak_n)
        low, high = self.pressure_force_range_n
        assert (low, high) in ((4., 7.), (7., 10.))
        # Match the exact parent's arithmetic for the equivalence mode.
        desired = 1.5*raw + (1. if low == 4. else 4.)
        original = self._sampled_peak_force_world[env_ids]
        magnitude = original.norm(dim=-1)
        assert torch.all(magnitude > 0), 'Pressure branch must contain no zero-force cohort.'
        direction = original / magnitude[:, None]
        capped = torch.minimum(desired, self._event_stiffness[env_ids].reshape(-1)*float(self.cfg.max_displacement_m))
        self.pressure_requested_peak_n[env_ids] = desired
        self.pressure_requested_peak_world[env_ids] = direction*desired[:, None]
        self.pressure_displacement_cap_scale[env_ids] = capped/desired
        if low == 4.:
            # Telemetry must leave original medium-force bytes untouched.
            torch.testing.assert_close(magnitude, capped, atol=3e-5, rtol=1e-5)
        else:
            self._sampled_peak_force_world[env_ids] = direction*capped[:, None]

