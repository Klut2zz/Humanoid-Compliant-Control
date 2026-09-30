"""General-tracking policy extensions."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import unpad_trajectories


# new change
class StandCoordinationWaistSupportMaskedGaussianDistribution(
    GaussianDistribution
):
    """PPO distribution whose stochastic/log-prob space is only V3 rows."""

    def __init__(self, output_dim: int) -> None:
        if output_dim != 29:
            raise ValueError(
                f"V3 masked distribution requires 29 actions, got {output_dim}."
            )
        super().__init__(output_dim, init_std=0.005, std_type="scalar")
        active_ids = torch.tensor(
            [0, 1, 2, 3, 4, 5, 8, 13, 14, 17, 18], dtype=torch.long
        )
        active_mask = torch.zeros(output_dim, dtype=torch.bool)
        active_mask[active_ids] = True
        self.register_buffer("active_action_mask", active_mask, persistent=False)

    def sample(self) -> torch.Tensor:
        sampled = self._distribution.sample()  # type: ignore[union-attr]
        return torch.where(self.active_action_mask, sampled, self.mean)

    @property
    def entropy(self) -> torch.Tensor:
        per_action = self._distribution.entropy()  # type: ignore[union-attr]
        return per_action[..., self.active_action_mask].sum(dim=-1)

    # new change
    # Retained on the V3 parent so the support-only subclass and existing
    # waist-support tasks share the same masked PPO semantics.
    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        per_action = self._distribution.log_prob(outputs)  # type: ignore[union-attr]
        return per_action[..., self.active_action_mask].sum(dim=-1)

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        per_action = torch.distributions.kl_divergence(
            Normal(old_mean, old_std), Normal(new_mean, new_std)
        )
        return per_action[..., self.active_action_mask].sum(dim=-1)


# new change
class StandCoordinationSupportOnlyMaskedGaussianDistribution(
    StandCoordinationWaistSupportMaskedGaussianDistribution
):
    """PPO distribution restricted to the eight support-adapter action rows."""

    def __init__(self, output_dim: int) -> None:
        super().__init__(output_dim)
        active_ids = torch.tensor(
            [0, 1, 3, 4, 13, 14, 17, 18], dtype=torch.long
        )
        active_mask = torch.zeros(output_dim, dtype=torch.bool)
        active_mask[active_ids] = True
        self.active_action_mask.copy_(active_mask)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        per_action = self._distribution.log_prob(outputs)  # type: ignore[union-attr]
        return per_action[..., self.active_action_mask].sum(dim=-1)


# new change
class StandCoordinationSupportControlV2MaskedGaussianDistribution(
    StandCoordinationSupportOnlyMaskedGaussianDistribution
):
    """PPO distribution restricted to the complete 10DOF support chain."""

    def __init__(self, output_dim: int) -> None:
        super().__init__(output_dim)
        active_ids = torch.tensor(
            [0, 1, 3, 4, 9, 10, 13, 14, 17, 18], dtype=torch.long
        )
        active_mask = torch.zeros(output_dim, dtype=torch.bool)
        active_mask[active_ids] = True
        self.active_action_mask.copy_(active_mask)

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        per_action = torch.distributions.kl_divergence(
            Normal(old_mean, old_std), Normal(new_mean, new_std)
        )
        return per_action[..., self.active_action_mask].sum(dim=-1)


# new change
class StandCoordinationOnlineComplianceMaskedGaussianDistribution(
    GaussianDistribution
):
    """Explore and score only waist/arm rows owned by the compliance head."""

    def __init__(self, output_dim: int) -> None:
        if output_dim != 29:
            raise ValueError(
                f"Online-compliance distribution requires 29 actions, got {output_dim}."
            )
        super().__init__(output_dim, init_std=0.01, std_type="scalar")
        active_ids = torch.tensor(
            [2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
            dtype=torch.long,
        )
        active_mask = torch.zeros(output_dim, dtype=torch.bool)
        active_mask[active_ids] = True
        self.register_buffer("active_action_mask", active_mask, persistent=False)

    def sample(self) -> torch.Tensor:
        sampled = self._distribution.sample()  # type: ignore[union-attr]
        return torch.where(self.active_action_mask, sampled, self.mean)

    @property
    def entropy(self) -> torch.Tensor:
        values = self._distribution.entropy()  # type: ignore[union-attr]
        return values[..., self.active_action_mask].sum(dim=-1)


# new change
class StandCoordinationOnlineComplianceArmOnlyMaskedGaussianDistribution(
    StandCoordinationOnlineComplianceMaskedGaussianDistribution
):
    """First compliance stage: explore arms while waist/support stay nominal.

    The online compliance model still contains a zero-initialized waist head so
    checkpoints remain forward-compatible with the later coordination stage.
    This distribution excludes all waist and lower-body action rows from PPO,
    preventing force adaptation from being achieved by translating the base.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__(output_dim)
        active_ids = torch.tensor(
            [11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
            dtype=torch.long,
        )
        active_mask = torch.zeros(output_dim, dtype=torch.bool)
        active_mask[active_ids] = True
        self.active_action_mask.copy_(active_mask)


# new change
class StandCoordinationOnlineComplianceWaistOnlyMaskedGaussianDistribution(
    StandCoordinationOnlineComplianceMaskedGaussianDistribution
):
    """Coordination stage: explore only waist yaw/roll/pitch action rows."""

    def __init__(self, output_dim: int) -> None:
        super().__init__(output_dim)
        active_ids = torch.tensor([2, 5, 8], dtype=torch.long)
        active_mask = torch.zeros(output_dim, dtype=torch.bool)
        active_mask[active_ids] = True
        self.active_action_mask.copy_(active_mask)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        values = self._distribution.log_prob(outputs)  # type: ignore[union-attr]
        return values[..., self.active_action_mask].sum(dim=-1)

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        values = torch.distributions.kl_divergence(
            Normal(old_mean, old_std), Normal(new_mean, new_std)
        )
        return values[..., self.active_action_mask].sum(dim=-1)

# new change
class GeneralBilateralStructuredArmResidualAdapterMLPModel(MLPModel):
    """Frozen-compatible whole-body actor plus two generic 7DOF arm adapters.

    Each side consumes q, q_ref, q-q_ref, dq, dq_ref and phase sin/cos (37
    values).  The three independent heads prevent shoulder, elbow and wrist
    credit from being averaged into one correction.  Their final layers are
    exactly zero, so a legacy model_1598 checkpoint is behaviorally identical
    immediately after loading.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        left_observation_group: str = "left_arm_adapter",
        right_observation_group: str = "right_arm_adapter",
        analytic_error_gain: float = 0.0,
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        for group_name in (left_observation_group, right_observation_group):
            if group_name not in obs:
                raise ValueError(f"Missing structured observation group '{group_name}'.")
            if int(obs[group_name].shape[-1]) != 37:
                raise ValueError(
                    f"Structured group '{group_name}' must contain 37 values, "
                    f"got {int(obs[group_name].shape[-1])}."
                )
        self.left_observation_group = left_observation_group
        self.right_observation_group = right_observation_group
        # new change
        # Non-persistent keeps legacy and trained adapter checkpoints compatible.
        self.register_buffer(
            "analytic_error_gain",
            torch.tensor(float(analytic_error_gain)),
            persistent=False,
        )

        # Actor units are converted by the environment to target radians.  Arm
        # action scale is 0.15, so these caps correspond to 0.03 rad at each
        # shoulder and 0.0375 rad at elbow/wrist.
        limits = torch.tensor([0.20, 0.20, 0.20, 0.25, 0.25, 0.25, 0.25])
        self.register_buffer("left_arm_adapter_limits", limits.clone(), persistent=True)
        self.register_buffer("right_arm_adapter_limits", limits.clone(), persistent=True)
        self.register_buffer(
            "left_arm_adapter_action_ids",
            torch.tensor([11, 15, 19, 21, 23, 25, 27], dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "right_arm_adapter_action_ids",
            torch.tensor([12, 16, 20, 22, 24, 26, 28], dtype=torch.long),
            persistent=True,
        )
        left_projection = torch.zeros(7, output_dim)
        right_projection = torch.zeros(7, output_dim)
        left_projection[torch.arange(7), self.left_arm_adapter_action_ids] = 1.0
        right_projection[torch.arange(7), self.right_arm_adapter_action_ids] = 1.0
        self.register_buffer("left_arm_adapter_projection", left_projection, persistent=True)
        self.register_buffer("right_arm_adapter_projection", right_projection, persistent=True)

        self.left_arm_residual_adapter = self._make_adapter()
        self.right_arm_residual_adapter = self._make_adapter()

    @staticmethod
    def _zero_head(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
        head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    @classmethod
    def _make_adapter(cls) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                "shoulder": cls._zero_head(37, 64, 3),
                "elbow": cls._zero_head(37, 32, 1),
                "wrist": cls._zero_head(37, 64, 3),
            }
        )

    @staticmethod
    def _raw_adapter_output(adapter: nn.ModuleDict, features: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                adapter["shoulder"](features),
                adapter["elbow"](features),
                adapter["wrist"](features),
            ),
            dim=-1,
        )

    def _adapter_output(
        self, adapter: nn.ModuleDict, features: torch.Tensor, limits: torch.Tensor
    ) -> torch.Tensor:
        return torch.tanh(self._raw_adapter_output(adapter, features)) * limits

    # new change
    def _analytic_error_output(
        self, features: torch.Tensor, limits: torch.Tensor
    ) -> torch.Tensor:
        """Map q-q_ref to bounded actor units for an inference-only probe."""

        q_error = features[..., 14:21]
        raw_actor_correction = -self.analytic_error_gain * q_error / 0.15
        return torch.maximum(
            torch.minimum(raw_actor_correction, limits), -limits
        )

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        output = self.mlp(latent)
        left = self._adapter_output(
            self.left_arm_residual_adapter,
            obs[self.left_observation_group],
            self.left_arm_adapter_limits,
        )
        right = self._adapter_output(
            self.right_arm_residual_adapter,
            obs[self.right_observation_group],
            self.right_arm_adapter_limits,
        )
        # new change
        left = left + self._analytic_error_output(
            obs[self.left_observation_group], self.left_arm_adapter_limits
        )
        right = right + self._analytic_error_output(
            obs[self.right_observation_group], self.right_arm_adapter_limits
        )
        output = output + left @ self.left_arm_adapter_projection
        output = output + right @ self.right_arm_adapter_projection
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output

    def as_jit(self) -> nn.Module:
        return _GeneralBilateralStructuredArmAdapterExport(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _GeneralBilateralStructuredArmAdapterOnnxExport(self, verbose)


# new change
class GeneralBilateralArmErrorFeedbackAlpha010MLPModel(
    GeneralBilateralStructuredArmResidualAdapterMLPModel
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, analytic_error_gain=0.10, **kwargs)


# new change
class GeneralBilateralArmErrorFeedbackAlpha025MLPModel(
    GeneralBilateralStructuredArmResidualAdapterMLPModel
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, analytic_error_gain=0.25, **kwargs)


# new change
class GeneralBilateralArmErrorFeedbackAlpha050MLPModel(
    GeneralBilateralStructuredArmResidualAdapterMLPModel
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, analytic_error_gain=0.50, **kwargs)


# new change
class StandCoordinationBilateralStructuredArmResidualAdapterMLPModel(
    GeneralBilateralStructuredArmResidualAdapterMLPModel
):
    """Stand-specific 7DOF-per-side adapter with deployment-safe limits.

    The four proximal joints are limited to 0.04 rad target correction and
    the three wrist joints to 0.08 rad.  This contains the supervised
    alpha=0.50 targets observed from model_1375 without granting the adapter
    control over the waist, legs, or legacy actor trunk.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, analytic_error_gain=0.0, **kwargs)
        target_limits_rad = torch.tensor(
            [0.04, 0.04, 0.04, 0.04, 0.08, 0.08, 0.08],
            device=self.left_arm_adapter_limits.device,
            dtype=self.left_arm_adapter_limits.dtype,
        )
        actor_limits = target_limits_rad / 0.15
        self.left_arm_adapter_limits.copy_(actor_limits)
        self.right_arm_adapter_limits.copy_(actor_limits)


# new change
class StandCoordinationOnlineComplianceResidualAdapterMLPModel(
    StandCoordinationBilateralStructuredArmResidualAdapterMLPModel
):
    """Accepted stand controller plus a zero-init force-conditioned head.

    The legacy actor and both accepted arm adapters remain byte-for-byte
    frozen.  A 54D online context controls only three waist and fourteen arm
    residual rows, keeping all twelve lower-body actions under model_1375.
    """

    def __init__(
        self,
        *args,
        compliance_observation_group: str = "online_compliance",
        compliance_feature_dim: int = 54,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        obs = args[0] if args else kwargs.get("obs")
        if obs is None or compliance_observation_group not in obs:
            raise ValueError(
                f"Missing online compliance group '{compliance_observation_group}'."
            )
        actual_dim = int(obs[compliance_observation_group].shape[-1])
        if actual_dim != compliance_feature_dim:
            raise ValueError(
                f"Online compliance group must contain {compliance_feature_dim} values, "
                f"got {actual_dim}."
            )
        self.compliance_observation_group = compliance_observation_group
        self.compliance_feature_dim = int(compliance_feature_dim)
        self.compliance_residual_adapter = nn.ModuleDict(
            {
                "waist": self._zero_head(self.compliance_feature_dim, 48, 3),
                "left_arm": self._zero_head(self.compliance_feature_dim, 96, 7),
                "right_arm": self._zero_head(self.compliance_feature_dim, 96, 7),
            }
        )
        action_ids = torch.tensor(
            [2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
            dtype=torch.long,
        )
        self.register_buffer(
            "compliance_adapter_action_ids", action_ids, persistent=True
        )
        # 4 mrad waist and 20 mrad arm authority leave substantial physical
        # compliance to the later stiffness-learning stage.
        target_limits_rad = torch.tensor([0.004, 0.004, 0.004] + [0.020] * 14)
        action_scales = torch.tensor([0.20, 0.20, 0.20] + [0.15] * 14)
        self.register_buffer(
            "compliance_adapter_limits",
            target_limits_rad / action_scales,
            persistent=True,
        )
        projection = torch.zeros(17, 29)
        projection[torch.arange(17), action_ids] = 1.0
        self.register_buffer(
            "compliance_adapter_projection", projection, persistent=True
        )

    def _compliance_output(self, features: torch.Tensor) -> torch.Tensor:
        raw = torch.cat(
            (
                self.compliance_residual_adapter["waist"](features),
                self.compliance_residual_adapter["left_arm"](features),
                self.compliance_residual_adapter["right_arm"](features),
            ),
            dim=-1,
        )
        return torch.tanh(raw) * self.compliance_adapter_limits

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        output = self.mlp(latent)
        left = self._adapter_output(
            self.left_arm_residual_adapter,
            obs[self.left_observation_group],
            self.left_arm_adapter_limits,
        )
        right = self._adapter_output(
            self.right_arm_residual_adapter,
            obs[self.right_observation_group],
            self.right_arm_adapter_limits,
        )
        compliance = self._compliance_output(obs[self.compliance_observation_group])
        output = output + left @ self.left_arm_adapter_projection
        output = output + right @ self.right_arm_adapter_projection
        output = output + compliance @ self.compliance_adapter_projection
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output

    def as_jit(self) -> nn.Module:
        return _StandCoordinationOnlineComplianceExport(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _StandCoordinationOnlineComplianceOnnxExport(self, verbose)


# new change
class StandCoordinationForceGatedOnlineComplianceResidualAdapterMLPModel(
    StandCoordinationOnlineComplianceResidualAdapterMLPModel
):
    """Make the learned compliance residual exactly zero without force.

    The 54D feature layout is q-error17, dq17 and context20.  The context's
    active-force flag is element 14, hence flattened index 48.  This hard gate
    protects nominal model_1375 behavior while retaining the already learned
    arm response during synchronized physical-force events.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "compliance_force_active_index",
            torch.tensor(48, dtype=torch.long),
            persistent=True,
        )

    def _compliance_output(self, features: torch.Tensor) -> torch.Tensor:
        residual = super()._compliance_output(features)
        active = features[..., 48:49].clamp(0.0, 1.0)
        return residual * active


# new change
class StandCoordinationAnalyticArmFeedbackOnlineComplianceMLPModel(
    StandCoordinationForceGatedOnlineComplianceResidualAdapterMLPModel
):
    """Inference-only bilateral q/dq feedback around frozen model_1375.

    Physical target correction is
    clip(-alpha*(q-q_ref)-beta*(dq-dq_ref), +/-20 mrad).  It is converted to
    actor units only after physical clipping and is hard-gated by the accepted
    online force flag, so nominal no-force behavior remains exactly unchanged.
    """

    def __init__(
        self,
        *args,
        analytic_position_gain: float,
        analytic_velocity_gain_s: float = 0.01,
        compliance_feature_dim: int = 71,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            compliance_feature_dim=compliance_feature_dim,
            **kwargs,
        )
        self.register_buffer(
            "analytic_position_gain",
            torch.tensor(float(analytic_position_gain)),
            persistent=True,
        )
        self.register_buffer(
            "analytic_velocity_gain_s",
            torch.tensor(float(analytic_velocity_gain_s)),
            persistent=True,
        )
        self.register_buffer(
            "analytic_arm_target_limit_rad",
            torch.tensor(0.020),
            persistent=True,
        )
        self.register_buffer(
            "analytic_left_feature_ids",
            torch.tensor([3, 5, 7, 9, 11, 13, 15], dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "analytic_right_feature_ids",
            torch.tensor([4, 6, 8, 10, 12, 14, 16], dtype=torch.long),
            persistent=True,
        )
        # q-error17 + dq17 + dq-ref17 + context active flag at index 14.
        self.compliance_force_active_index.fill_(65)

    def _arm_feedback_actor_units(
        self, features: torch.Tensor, feature_ids: torch.Tensor
    ) -> torch.Tensor:
        q_error = features[..., :17].index_select(-1, feature_ids)
        dq = features[..., 17:34].index_select(-1, feature_ids)
        dq_ref = features[..., 34:51].index_select(-1, feature_ids)
        correction_rad = (
            -self.analytic_position_gain * q_error
            - self.analytic_velocity_gain_s * (dq - dq_ref)
        )
        correction_rad = torch.clamp(
            correction_rad,
            min=-self.analytic_arm_target_limit_rad,
            max=self.analytic_arm_target_limit_rad,
        )
        return correction_rad / 0.15

    def _compliance_output(self, features: torch.Tensor) -> torch.Tensor:
        learned = StandCoordinationOnlineComplianceResidualAdapterMLPModel._compliance_output(
            self, features
        )
        analytic = torch.zeros_like(learned)
        analytic[..., 3:10] = self._arm_feedback_actor_units(
            features, self.analytic_left_feature_ids
        )
        analytic[..., 10:17] = self._arm_feedback_actor_units(
            features, self.analytic_right_feature_ids
        )
        combined = torch.maximum(
            torch.minimum(learned + analytic, self.compliance_adapter_limits),
            -self.compliance_adapter_limits,
        )
        active = features[..., 65:66].clamp(0.0, 1.0)
        return combined * active

    def as_jit(self) -> nn.Module:
        return _StandCoordinationAnalyticArmFeedbackOnlineComplianceExport(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _StandCoordinationAnalyticArmFeedbackOnlineComplianceOnnxExport(
            self, verbose
        )


# new change
class StandCoordinationAnalyticArmFeedbackAlpha010MLPModel(
    StandCoordinationAnalyticArmFeedbackOnlineComplianceMLPModel
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, analytic_position_gain=0.10, **kwargs)


# new change
class StandCoordinationAnalyticArmFeedbackAlpha025MLPModel(
    StandCoordinationAnalyticArmFeedbackOnlineComplianceMLPModel
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, analytic_position_gain=0.25, **kwargs)


# new change
class StandCoordinationAnalyticArmFeedbackAlpha050MLPModel(
    StandCoordinationAnalyticArmFeedbackOnlineComplianceMLPModel
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, analytic_position_gain=0.50, **kwargs)


# new change
class StandCoordinationSupervisedArmFeedbackOnlineComplianceMLPModel(
    StandCoordinationForceGatedOnlineComplianceResidualAdapterMLPModel
):
    """Learned 71D replacement for the accepted alpha=0.50 analytic teacher.

    The legacy actor, bilateral stand adapters, waist and support actions are
    unchanged.  Only the two online arm heads reproduce the bounded teacher;
    the inherited force gate makes the no-force output exactly zero.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, compliance_feature_dim=71, **kwargs)
        # q-error17 + dq17 + dq-ref17 + context active flag at index 14.
        self.compliance_force_active_index.fill_(65)


# new change
class StandCoordinationArmWaistCOMResidualAdapterMLPModel(
    StandCoordinationBilateralStructuredArmResidualAdapterMLPModel
):
    """Protected arm-accuracy actor with a tiny zero-init waist coordinator.

    The accepted bilateral arm adapters are loaded unchanged from the
    supervised model_1375 checkpoint.  A separate 17D waist head sees
    q/q-ref/dq/reference velocity/phase and can add at most 0.005 rad yaw or
    0.01 rad roll/pitch.  It never modifies the legacy actor or leg actions.
    """

    def __init__(
        self,
        *args,
        waist_observation_group: str = "waist_adapter",
        waist_feature_dim: int = 17,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        obs = args[0] if args else kwargs.get("obs")
        if obs is None or waist_observation_group not in obs:
            raise ValueError(
                f"Missing structured observation group '{waist_observation_group}'."
            )
        if int(obs[waist_observation_group].shape[-1]) != waist_feature_dim:
            raise ValueError(
                f"Waist group '{waist_observation_group}' must contain "
                f"{waist_feature_dim} values, "
                f"got {int(obs[waist_observation_group].shape[-1])}."
            )
        self.waist_observation_group = waist_observation_group
        self.waist_feature_dim = int(waist_feature_dim)
        self.waist_residual_adapter = nn.ModuleDict(
            {
                "yaw": self._zero_head(self.waist_feature_dim, 16, 1),
                "roll_pitch": self._zero_head(self.waist_feature_dim, 32, 2),
            }
        )
        # Waist action scale is 0.20 rad per actor unit.  These caps preserve
        # substantial authority for later compliant/disturbance training.
        waist_target_limits_rad = torch.tensor([0.005, 0.010, 0.010])
        self.register_buffer(
            "waist_adapter_limits",
            waist_target_limits_rad / 0.20,
            persistent=True,
        )
        self.register_buffer(
            "waist_adapter_action_ids",
            torch.tensor([2, 5, 8], dtype=torch.long),
            persistent=True,
        )
        waist_projection = torch.zeros(
            3, self.left_arm_adapter_projection.shape[1]
        )
        waist_projection[torch.arange(3), self.waist_adapter_action_ids] = 1.0
        self.register_buffer(
            "waist_adapter_projection", waist_projection, persistent=True
        )

    def _waist_output(self, features: torch.Tensor) -> torch.Tensor:
        raw = torch.cat(
            (
                self.waist_residual_adapter["yaw"](features),
                self.waist_residual_adapter["roll_pitch"](features),
            ),
            dim=-1,
        )
        return torch.tanh(raw) * self.waist_adapter_limits

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        output = self.mlp(latent)
        left = self._adapter_output(
            self.left_arm_residual_adapter,
            obs[self.left_observation_group],
            self.left_arm_adapter_limits,
        )
        right = self._adapter_output(
            self.right_arm_residual_adapter,
            obs[self.right_observation_group],
            self.right_arm_adapter_limits,
        )
        waist = self._waist_output(obs[self.waist_observation_group])
        output = output + left @ self.left_arm_adapter_projection
        output = output + right @ self.right_arm_adapter_projection
        output = output + waist @ self.waist_adapter_projection
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output

    def as_jit(self) -> nn.Module:
        return _StandCoordinationArmWaistCOMExport(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _StandCoordinationArmWaistCOMOnnxExport(self, verbose)


# new change
class StandCoordinationStateAwareArmWaistCOMResidualAdapterMLPModel(
    StandCoordinationArmWaistCOMResidualAdapterMLPModel
):
    """V2 waist coordinator with explicit COM, torso, and angular state."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, waist_feature_dim=25, **kwargs)


# new change
class StandCoordinationStateAwareWaistSupportResidualAdapterMLPModel(
    StandCoordinationStateAwareArmWaistCOMResidualAdapterMLPModel
):
    """V3: frozen legacy/arms plus bounded waist and support-chain heads."""

    def __init__(
        self,
        *args,
        support_observation_group: str = "support_adapter",
        support_feature_dim: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        obs = args[0] if args else kwargs.get("obs")
        if obs is None or support_observation_group not in obs:
            raise ValueError(
                f"Missing structured observation group '{support_observation_group}'."
            )
        if int(obs[support_observation_group].shape[-1]) != support_feature_dim:
            raise ValueError(
                f"Support group '{support_observation_group}' must contain "
                f"{support_feature_dim} values, "
                f"got {int(obs[support_observation_group].shape[-1])}."
            )
        self.support_observation_group = support_observation_group
        self.support_feature_dim = int(support_feature_dim)
        self.support_residual_adapter = nn.ModuleDict(
            {
                "sagittal": self._zero_head(self.support_feature_dim, 32, 4),
                "lateral": self._zero_head(self.support_feature_dim, 32, 4),
            }
        )
        # Target-space order: bilateral hip pitch, ankle pitch, hip roll,
        # ankle roll. Hip authority is 4 mrad and ankle authority is 6 mrad.
        support_target_limits_rad = torch.tensor(
            [0.004, 0.004, 0.006, 0.006, 0.004, 0.004, 0.006, 0.006]
        )
        self.register_buffer(
            "support_adapter_limits",
            support_target_limits_rad / 0.30,
            persistent=True,
        )
        self.register_buffer(
            "support_adapter_action_ids",
            torch.tensor([0, 1, 13, 14, 3, 4, 17, 18], dtype=torch.long),
            persistent=True,
        )
        support_projection = torch.zeros(
            8, self.left_arm_adapter_projection.shape[1]
        )
        support_projection[
            torch.arange(8), self.support_adapter_action_ids
        ] = 1.0
        self.register_buffer(
            "support_adapter_projection", support_projection, persistent=True
        )

    def _support_output(self, features: torch.Tensor) -> torch.Tensor:
        raw = torch.cat(
            (
                self.support_residual_adapter["sagittal"](features),
                self.support_residual_adapter["lateral"](features),
            ),
            dim=-1,
        )
        return torch.tanh(raw) * self.support_adapter_limits

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        output = self.mlp(latent)
        left = self._adapter_output(
            self.left_arm_residual_adapter,
            obs[self.left_observation_group],
            self.left_arm_adapter_limits,
        )
        right = self._adapter_output(
            self.right_arm_residual_adapter,
            obs[self.right_observation_group],
            self.right_arm_adapter_limits,
        )
        waist = self._waist_output(obs[self.waist_observation_group])
        support = self._support_output(obs[self.support_observation_group])
        output = output + left @ self.left_arm_adapter_projection
        output = output + right @ self.right_arm_adapter_projection
        output = output + waist @ self.waist_adapter_projection
        output = output + support @ self.support_adapter_projection
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output

    def as_jit(self) -> nn.Module:
        return _StandCoordinationWaistSupportExport(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _StandCoordinationWaistSupportOnnxExport(self, verbose)


# new change
class StandCoordinationStationarySupportResidualAdapterMLPModel(
    StandCoordinationStateAwareWaistSupportResidualAdapterMLPModel
):
    """Support-only successor with translational and resolved-contact state."""

    def __init__(self, *args, **kwargs) -> None:
        # Original support50 plus root-position2, base-velocity3,
        # two foot velocities6, resolved force2, and resolved contact2.
        super().__init__(*args, support_feature_dim=65, **kwargs)
        # Physical target authority: 15 mrad at hips and 30 mrad at ankles.
        # This is below the unsafe 60 mrad inference probe while exceeding the
        # ineffective 4/6 mrad V3 range.
        target_limits_rad = torch.tensor(
            [0.015, 0.015, 0.030, 0.030, 0.015, 0.015, 0.030, 0.030],
            device=self.support_adapter_limits.device,
            dtype=self.support_adapter_limits.dtype,
        )
        self.support_adapter_limits.copy_(target_limits_rad / 0.30)


# new change
class StandCoordinationSupportControlV2ResidualAdapterMLPModel(
    StandCoordinationStateAwareWaistSupportResidualAdapterMLPModel
):
    """Complete hip/knee/ankle support head with fixed-foot state feedback."""

    def __init__(self, *args, **kwargs) -> None:
        # 10 joints x (q, qref, error, dq, dqref) + phase2 + COM2 +
        # gravity3 + angular velocity3 + root error2 + base velocity3 +
        # foot velocity6 + forces2 + contacts2 + fixed-foot error4 = 79.
        super().__init__(*args, support_feature_dim=79, **kwargs)
        self.support_residual_adapter = nn.ModuleDict(
            {
                "sagittal": self._zero_head(79, 48, 6),
                "lateral": self._zero_head(79, 32, 4),
            }
        )
        # Finite differences showed +5 mrad knee correction was beneficial
        # without failures while +10 mrad caused failures.  Keep learned knee
        # authority at a conservative +/-6 mrad.  Hip and ankle limits remain
        # those already verified by the StationarySupportAdapter task.
        target_limits_rad = torch.tensor(
            [
                0.015,
                0.015,
                0.006,
                0.006,
                0.030,
                0.030,
                0.015,
                0.015,
                0.030,
                0.030,
            ],
            device=self.support_adapter_limits.device,
            dtype=self.support_adapter_limits.dtype,
        )
        self.support_adapter_limits = target_limits_rad / 0.30
        self.support_adapter_action_ids = torch.tensor(
            [0, 1, 9, 10, 13, 14, 3, 4, 17, 18],
            device=self.support_adapter_limits.device,
            dtype=torch.long,
        )
        support_projection = torch.zeros(
            10,
            self.left_arm_adapter_projection.shape[1],
            device=self.left_arm_adapter_projection.device,
            dtype=self.left_arm_adapter_projection.dtype,
        )
        support_projection[
            torch.arange(10, device=support_projection.device),
            self.support_adapter_action_ids,
        ] = 1.0
        self.support_adapter_projection = support_projection


# new change
class StandCoordinationSupportControlV2SupervisedInitMLPModel(
    StandCoordinationSupportControlV2ResidualAdapterMLPModel
):
    """Supervised support bootstrap with strict +/-6 mrad authority on 10DOF."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.support_adapter_limits = torch.full_like(
            self.support_adapter_limits, 0.006 / 0.30
        )


# new change
class StandCoordinationStationaryFoundationV2MLPModel(
    StandCoordinationStateAwareWaistSupportResidualAdapterMLPModel
):
    """Direct state-aware 10DOF support residual for the stationary foundation.

    Unlike the rejected RecoveryAwareTeacherV3 path, this head does not gate a
    joint-error teacher.  It directly maps support geometry, COM, contacts,
    loads, velocities and failure-trend state to bounded target corrections.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, support_feature_dim=87, **kwargs)
        self.support_residual_adapter = nn.ModuleDict(
            {
                "sagittal": self._zero_head(87, 64, 6),
                "lateral": self._zero_head(87, 48, 4),
            }
        )
        # Preserve the finite-difference safety envelope: hips +/-15 mrad,
        # knees +/-6 mrad, ankles +/-30 mrad.
        target_limits_rad = torch.tensor(
            [
                0.015,
                0.015,
                0.006,
                0.006,
                0.030,
                0.030,
                0.015,
                0.015,
                0.030,
                0.030,
            ],
            device=self.left_arm_adapter_limits.device,
            dtype=self.left_arm_adapter_limits.dtype,
        )
        self.support_adapter_limits = target_limits_rad / 0.30
        self.support_adapter_action_ids = torch.tensor(
            [0, 1, 9, 10, 13, 14, 3, 4, 17, 18],
            device=self.left_arm_adapter_limits.device,
            dtype=torch.long,
        )
        support_projection = torch.zeros(
            10,
            self.left_arm_adapter_projection.shape[1],
            device=self.left_arm_adapter_projection.device,
            dtype=self.left_arm_adapter_projection.dtype,
        )
        support_projection[
            torch.arange(10, device=support_projection.device),
            self.support_adapter_action_ids,
        ] = 1.0
        self.support_adapter_projection = support_projection
        # new change
        # Offline local-response labels mix radians, velocities, forces,
        # contacts and scaled geometry.  Keep normalization local to the new
        # head so the frozen actor and all legacy observations stay untouched.
        # Defaults preserve exact semantics for old checkpoints.
        self.register_buffer(
            "support_feature_mean", torch.zeros(87), persistent=True
        )
        self.register_buffer(
            "support_feature_scale", torch.ones(87), persistent=True
        )
        # new change
        # Frozen-inference safety A/B can reduce the distilled local teacher
        # without changing targets, actor parameters, actuators or PPO state.
        self.register_buffer(
            "support_output_scale", torch.tensor(1.0), persistent=True
        )


# new change
class StandCoordinationStationarySupportContainmentV3MLPModel(
    StandCoordinationStationaryFoundationV2MLPModel
):
    """Uniform six-milliradian support head for containment training."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Stored in actor-action units; the G1 lower-body target scale is 0.30.
        self.support_adapter_limits = torch.full_like(
            self.support_adapter_limits, 0.006 / 0.30
        )

    # new change
    def _support_output(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (
            features - self.support_feature_mean
        ) / self.support_feature_scale.clamp_min(1.0e-6)
        raw = torch.cat(
            (
                self.support_residual_adapter["sagittal"](normalized),
                self.support_residual_adapter["lateral"](normalized),
            ),
            dim=-1,
        )
        return (
            torch.tanh(raw)
            * self.support_adapter_limits
            * self.support_output_scale
        )


# new change
class StandCoordinationStationarySupportStateResidualV5MLPModel(
    StandCoordinationStationaryFoundationV2MLPModel
):
    """Full-state support residual with unit local gradient and 30 mrad bound.

    V3 multiplied the zero-initialized raw head by a 0.02 actor-unit limit,
    attenuating PPO gradients by fifty times.  This head treats its raw output
    directly as actor residual units and only clamps at the verified physical
    envelope.  Zero remains exactly equivalent to the legacy actor.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Lower-body action scale is 0.30 rad: 0.10 actor units = 30 mrad.
        self.support_adapter_limits = torch.full_like(
            self.support_adapter_limits, 0.030 / 0.30
        )

    # new change
    def _support_output(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (
            features - self.support_feature_mean
        ) / self.support_feature_scale.clamp_min(1.0e-6)
        raw_actor_residual = torch.cat(
            (
                self.support_residual_adapter["sagittal"](normalized),
                self.support_residual_adapter["lateral"](normalized),
            ),
            dim=-1,
        )
        return torch.clamp(
            raw_actor_residual,
            min=-self.support_adapter_limits,
            max=self.support_adapter_limits,
        ) * self.support_output_scale


# new change
class StandCoordinationStationaryBaseStateAwareV6MLPModel(
    StandCoordinationStationarySupportStateResidualV5MLPModel
):
    """Trainable stationary-base head after the corrected 10DOF audit.

    V5 proved that 30 mrad was not the active bound: ten ultra-conservative
    PPO iterations changed the physical output by only 0.003 mrad.  The
    corrected paired response matrix and the 12 mrad analytic probe establish
    a safe initial direction, while cancelling the observed 0.2 m/s drift may
    require materially more authority.  V6 therefore preserves the exact
    state interface and unit local gradient, but exposes a 60 mrad bound for
    the independently trainable support head only.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.support_adapter_limits = torch.full_like(
            self.support_adapter_limits, 0.060 / 0.30
        )


# new change
class StandCoordinationSupportRecoveryAwareTeacherV3MLPModel(
    StandCoordinationStateAwareWaistSupportResidualAdapterMLPModel
):
    """87D outcome-filtered support adapter with strict 6-mrad authority."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, support_feature_dim=87, **kwargs)
        self.support_residual_adapter = nn.ModuleDict(
            {
                "sagittal": self._zero_head(87, 64, 6),
                "lateral": self._zero_head(87, 48, 4),
            }
        )
        # The heads predict only a safety gate. A strongly negative default
        # is float32-equivalent to zero residual before offline initialization.
        for head in self.support_residual_adapter.values():
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, -20.0)
        self.support_adapter_limits = torch.full(
            (10,),
            0.006 / 0.30,
            device=self.left_arm_adapter_limits.device,
            dtype=self.left_arm_adapter_limits.dtype,
        )
        self.support_adapter_action_ids = torch.tensor(
            [0, 1, 9, 10, 13, 14, 3, 4, 17, 18],
            device=self.left_arm_adapter_limits.device,
            dtype=torch.long,
        )
        support_projection = torch.zeros(
            10,
            self.left_arm_adapter_projection.shape[1],
            device=self.left_arm_adapter_projection.device,
            dtype=self.left_arm_adapter_projection.dtype,
        )
        support_projection[
            torch.arange(10, device=support_projection.device),
            self.support_adapter_action_ids,
        ] = 1.0
        self.support_adapter_projection = support_projection
        self.register_buffer(
            "support_candidate_gain", torch.tensor(0.10), persistent=True
        )
        self.register_buffer(
            "support_candidate_limit_rad", torch.tensor(0.006), persistent=True
        )
        self.register_buffer(
            "support_action_scale_rad", torch.tensor(0.30), persistent=True
        )

    def _support_output(self, features: torch.Tensor) -> torch.Tensor:
        gate_logits = torch.cat(
            (
                self.support_residual_adapter["sagittal"](features),
                self.support_residual_adapter["lateral"](features),
            ),
            dim=-1,
        )
        gate = torch.sigmoid(gate_logits)
        candidate_rad = torch.clamp(
            -self.support_candidate_gain * features[..., 20:30],
            min=-float(self.support_candidate_limit_rad),
            max=float(self.support_candidate_limit_rad),
        )
        return candidate_rad * gate / self.support_action_scale_rad


# new change
class _GeneralBilateralStructuredArmAdapterExport(nn.Module):
    """Deploy with policy observations followed by left37 and right37."""

    def __init__(self, model: GeneralBilateralStructuredArmResidualAdapterMLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.left_arm_residual_adapter = copy.deepcopy(model.left_arm_residual_adapter)
        self.right_arm_residual_adapter = copy.deepcopy(model.right_arm_residual_adapter)
        self.register_buffer("left_arm_adapter_limits", model.left_arm_adapter_limits.detach().clone())
        self.register_buffer("right_arm_adapter_limits", model.right_arm_adapter_limits.detach().clone())
        self.register_buffer("left_arm_adapter_projection", model.left_arm_adapter_projection.detach().clone())
        self.register_buffer("right_arm_adapter_projection", model.right_arm_adapter_projection.detach().clone())
        # new change
        self.register_buffer(
            "analytic_error_gain", model.analytic_error_gain.detach().clone()
        )
        self.policy_obs_dim = model.obs_dim
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        policy_x = x[..., : self.policy_obs_dim]
        left_features = x[..., self.policy_obs_dim : self.policy_obs_dim + 37]
        right_features = x[..., self.policy_obs_dim + 37 : self.policy_obs_dim + 74]
        output = self.mlp(self.obs_normalizer(policy_x))
        # new change
        # Spell out the fixed ModuleDict keys: TorchScript cannot index a
        # ModuleDict passed through a helper argument.
        left_raw = torch.cat(
            (
                self.left_arm_residual_adapter["shoulder"](left_features),
                self.left_arm_residual_adapter["elbow"](left_features),
                self.left_arm_residual_adapter["wrist"](left_features),
            ),
            dim=-1,
        )
        right_raw = torch.cat(
            (
                self.right_arm_residual_adapter["shoulder"](right_features),
                self.right_arm_residual_adapter["elbow"](right_features),
                self.right_arm_residual_adapter["wrist"](right_features),
            ),
            dim=-1,
        )
        left = torch.tanh(left_raw) * self.left_arm_adapter_limits
        right = torch.tanh(right_raw) * self.right_arm_adapter_limits
        # new change
        left_error = left_features[..., 14:21]
        right_error = right_features[..., 14:21]
        left_feedback = -self.analytic_error_gain * left_error / 0.15
        right_feedback = -self.analytic_error_gain * right_error / 0.15
        left = left + torch.maximum(
            torch.minimum(left_feedback, self.left_arm_adapter_limits),
            -self.left_arm_adapter_limits,
        )
        right = right + torch.maximum(
            torch.minimum(right_feedback, self.right_arm_adapter_limits),
            -self.right_arm_adapter_limits,
        )
        output = output + left @ self.left_arm_adapter_projection
        output = output + right @ self.right_arm_adapter_projection
        return self.deterministic_output(output)

    @torch.jit.export
    def reset(self) -> None:
        pass


# new change
class _StandCoordinationArmWaistCOMExport(
    _GeneralBilateralStructuredArmAdapterExport
):
    """Deploy with policy705 + left37 + right37 + waist17 observations."""

    def __init__(self, model: StandCoordinationArmWaistCOMResidualAdapterMLPModel) -> None:
        super().__init__(model)
        self.waist_residual_adapter = copy.deepcopy(model.waist_residual_adapter)
        self.register_buffer(
            "waist_adapter_limits", model.waist_adapter_limits.detach().clone()
        )
        self.register_buffer(
            "waist_adapter_projection", model.waist_adapter_projection.detach().clone()
        )
        self.waist_feature_dim = model.waist_feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        policy_x = x[..., : self.policy_obs_dim]
        left_features = x[..., self.policy_obs_dim : self.policy_obs_dim + 37]
        right_features = x[..., self.policy_obs_dim + 37 : self.policy_obs_dim + 74]
        waist_features = x[
            ...,
            self.policy_obs_dim + 74 : self.policy_obs_dim + 74 + self.waist_feature_dim,
        ]
        output = self.mlp(self.obs_normalizer(policy_x))
        left_raw = torch.cat(
            (
                self.left_arm_residual_adapter["shoulder"](left_features),
                self.left_arm_residual_adapter["elbow"](left_features),
                self.left_arm_residual_adapter["wrist"](left_features),
            ),
            dim=-1,
        )
        right_raw = torch.cat(
            (
                self.right_arm_residual_adapter["shoulder"](right_features),
                self.right_arm_residual_adapter["elbow"](right_features),
                self.right_arm_residual_adapter["wrist"](right_features),
            ),
            dim=-1,
        )
        waist_raw = torch.cat(
            (
                self.waist_residual_adapter["yaw"](waist_features),
                self.waist_residual_adapter["roll_pitch"](waist_features),
            ),
            dim=-1,
        )
        left = torch.tanh(left_raw) * self.left_arm_adapter_limits
        right = torch.tanh(right_raw) * self.right_arm_adapter_limits
        waist = torch.tanh(waist_raw) * self.waist_adapter_limits
        output = output + left @ self.left_arm_adapter_projection
        output = output + right @ self.right_arm_adapter_projection
        output = output + waist @ self.waist_adapter_projection
        return self.deterministic_output(output)


# new change
class _StandCoordinationOnlineComplianceExport(
    _GeneralBilateralStructuredArmAdapterExport
):
    """Deploy policy705 + left37 + right37 + online-compliance54."""

    def __init__(
        self, model: StandCoordinationOnlineComplianceResidualAdapterMLPModel
    ) -> None:
        super().__init__(model)
        self.compliance_residual_adapter = copy.deepcopy(
            model.compliance_residual_adapter
        )
        self.register_buffer(
            "compliance_adapter_limits",
            model.compliance_adapter_limits.detach().clone(),
        )
        self.register_buffer(
            "compliance_adapter_projection",
            model.compliance_adapter_projection.detach().clone(),
        )
        self.compliance_feature_dim = model.compliance_feature_dim
        # new change
        self.compliance_force_active_index = int(
            getattr(model, "compliance_force_active_index", -1)
        )

    # new change
    def _learned_compliance_output(
        self, compliance_features: torch.Tensor
    ) -> torch.Tensor:
        compliance_raw = torch.cat(
            (
                self.compliance_residual_adapter["waist"](compliance_features),
                self.compliance_residual_adapter["left_arm"](compliance_features),
                self.compliance_residual_adapter["right_arm"](compliance_features),
            ),
            dim=-1,
        )
        return torch.tanh(compliance_raw) * self.compliance_adapter_limits

    # new change
    def _compliance_output(
        self, compliance_features: torch.Tensor
    ) -> torch.Tensor:
        compliance = self._learned_compliance_output(compliance_features)
        if self.compliance_force_active_index >= 0:
            active_start = self.compliance_force_active_index
            compliance = compliance * compliance_features[
                ..., active_start : active_start + 1
            ].clamp(0.0, 1.0)
        return compliance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        policy_x = x[..., : self.policy_obs_dim]
        left_features = x[..., self.policy_obs_dim : self.policy_obs_dim + 37]
        right_features = x[..., self.policy_obs_dim + 37 : self.policy_obs_dim + 74]
        compliance_features = x[
            ...,
            self.policy_obs_dim
            + 74 : self.policy_obs_dim
            + 74
            + self.compliance_feature_dim,
        ]
        output = self.mlp(self.obs_normalizer(policy_x))
        left_raw = torch.cat(
            (
                self.left_arm_residual_adapter["shoulder"](left_features),
                self.left_arm_residual_adapter["elbow"](left_features),
                self.left_arm_residual_adapter["wrist"](left_features),
            ),
            dim=-1,
        )
        right_raw = torch.cat(
            (
                self.right_arm_residual_adapter["shoulder"](right_features),
                self.right_arm_residual_adapter["elbow"](right_features),
                self.right_arm_residual_adapter["wrist"](right_features),
            ),
            dim=-1,
        )
        left = torch.tanh(left_raw) * self.left_arm_adapter_limits
        right = torch.tanh(right_raw) * self.right_arm_adapter_limits
        compliance = self._compliance_output(compliance_features)
        output = output + left @ self.left_arm_adapter_projection
        output = output + right @ self.right_arm_adapter_projection
        output = output + compliance @ self.compliance_adapter_projection
        return self.deterministic_output(output)


# new change
class _StandCoordinationAnalyticArmFeedbackOnlineComplianceExport(
    _StandCoordinationOnlineComplianceExport
):
    """JIT/export equivalent of the 71D analytic bilateral arm probe."""

    def __init__(
        self, model: StandCoordinationAnalyticArmFeedbackOnlineComplianceMLPModel
    ) -> None:
        super().__init__(model)
        self.register_buffer(
            "analytic_position_gain",
            model.analytic_position_gain.detach().clone(),
        )
        self.register_buffer(
            "analytic_velocity_gain_s",
            model.analytic_velocity_gain_s.detach().clone(),
        )
        self.register_buffer(
            "analytic_arm_target_limit_rad",
            model.analytic_arm_target_limit_rad.detach().clone(),
        )
        self.register_buffer(
            "analytic_left_feature_ids",
            model.analytic_left_feature_ids.detach().clone(),
        )
        self.register_buffer(
            "analytic_right_feature_ids",
            model.analytic_right_feature_ids.detach().clone(),
        )

    def _arm_feedback_actor_units(
        self, features: torch.Tensor, feature_ids: torch.Tensor
    ) -> torch.Tensor:
        q_error = features[..., :17].index_select(-1, feature_ids)
        dq = features[..., 17:34].index_select(-1, feature_ids)
        dq_ref = features[..., 34:51].index_select(-1, feature_ids)
        correction_rad = (
            -self.analytic_position_gain * q_error
            - self.analytic_velocity_gain_s * (dq - dq_ref)
        )
        correction_rad = torch.clamp(
            correction_rad,
            min=-self.analytic_arm_target_limit_rad,
            max=self.analytic_arm_target_limit_rad,
        )
        return correction_rad / 0.15

    def _compliance_output(
        self, compliance_features: torch.Tensor
    ) -> torch.Tensor:
        learned = self._learned_compliance_output(compliance_features)
        analytic = torch.zeros_like(learned)
        analytic[..., 3:10] = self._arm_feedback_actor_units(
            compliance_features, self.analytic_left_feature_ids
        )
        analytic[..., 10:17] = self._arm_feedback_actor_units(
            compliance_features, self.analytic_right_feature_ids
        )
        combined = torch.maximum(
            torch.minimum(learned + analytic, self.compliance_adapter_limits),
            -self.compliance_adapter_limits,
        )
        return combined * compliance_features[..., 65:66].clamp(0.0, 1.0)


# new change
class _GeneralBilateralStructuredArmAdapterOnnxExport(
    _GeneralBilateralStructuredArmAdapterExport
):
    is_recurrent: bool = False

    def __init__(self, model: GeneralBilateralStructuredArmResidualAdapterMLPModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim + 74

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs_policy_plus_left37_plus_right37"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


# new change
class _StandCoordinationArmWaistCOMOnnxExport(
    _StandCoordinationArmWaistCOMExport
):
    is_recurrent: bool = False

    def __init__(
        self,
        model: StandCoordinationArmWaistCOMResidualAdapterMLPModel,
        verbose: bool,
    ) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim + 74 + model.waist_feature_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs_policy_plus_left37_plus_right37_plus_waist_state"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


# new change
class _StandCoordinationOnlineComplianceOnnxExport(
    _StandCoordinationOnlineComplianceExport
):
    is_recurrent: bool = False

    def __init__(
        self,
        model: StandCoordinationOnlineComplianceResidualAdapterMLPModel,
        verbose: bool,
    ) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim + 74 + model.compliance_feature_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs_policy_plus_bilateral_arm_plus_online_compliance"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


# new change
class _StandCoordinationAnalyticArmFeedbackOnlineComplianceOnnxExport(
    _StandCoordinationAnalyticArmFeedbackOnlineComplianceExport
):
    is_recurrent: bool = False

    def __init__(
        self,
        model: StandCoordinationAnalyticArmFeedbackOnlineComplianceMLPModel,
        verbose: bool,
    ) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim + 74 + model.compliance_feature_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs_policy_plus_bilateral_arm_plus_online_feedback71"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


# new change
class _StandCoordinationWaistSupportExport(nn.Module):
    """Deploy V3 with policy705 + left37 + right37 + waist25 + support50."""

    def __init__(
        self, model: StandCoordinationStateAwareWaistSupportResidualAdapterMLPModel
    ) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.left_arm_residual_adapter = copy.deepcopy(
            model.left_arm_residual_adapter
        )
        self.right_arm_residual_adapter = copy.deepcopy(
            model.right_arm_residual_adapter
        )
        self.waist_residual_adapter = copy.deepcopy(model.waist_residual_adapter)
        self.support_residual_adapter = copy.deepcopy(
            model.support_residual_adapter
        )
        for buffer_name in (
            "left_arm_adapter_limits",
            "right_arm_adapter_limits",
            "left_arm_adapter_projection",
            "right_arm_adapter_projection",
            "waist_adapter_limits",
            "waist_adapter_projection",
            "support_adapter_limits",
            "support_adapter_projection",
        ):
            self.register_buffer(
                buffer_name,
                getattr(model, buffer_name).detach().clone(),
            )
        self.policy_obs_dim = model.obs_dim
        self.waist_feature_dim = model.waist_feature_dim
        self.support_feature_dim = model.support_feature_dim
        # new change
        self.normalized_direct_support = hasattr(model, "support_feature_mean")
        if self.normalized_direct_support:
            self.register_buffer(
                "support_feature_mean",
                model.support_feature_mean.detach().clone(),
            )
            self.register_buffer(
                "support_feature_scale",
                model.support_feature_scale.detach().clone(),
            )
            self.register_buffer(
                "support_output_scale",
                model.support_output_scale.detach().clone(),
            )
        else:
            self.register_buffer(
                "support_feature_mean", torch.zeros(self.support_feature_dim)
            )
            self.register_buffer(
                "support_feature_scale", torch.ones(self.support_feature_dim)
            )
            self.register_buffer("support_output_scale", torch.tensor(1.0))
        # new change
        self.recovery_aware_support = hasattr(model, "support_candidate_gain")
        if self.recovery_aware_support:
            self.register_buffer(
                "support_candidate_gain",
                model.support_candidate_gain.detach().clone(),
            )
            self.register_buffer(
                "support_candidate_limit_rad",
                model.support_candidate_limit_rad.detach().clone(),
            )
            self.register_buffer(
                "support_action_scale_rad",
                model.support_action_scale_rad.detach().clone(),
            )
        # new change
        # TorchScript compiles both sides of the branch below.  Register
        # shape-compatible defaults for direct-residual models as well so JIT
        # export does not reference missing RecoveryAware-only attributes.
        else:
            self.register_buffer(
                "support_candidate_gain", torch.tensor(0.0)
            )
            self.register_buffer(
                "support_candidate_limit_rad", torch.tensor(0.0)
            )
            self.register_buffer(
                "support_action_scale_rad", torch.tensor(0.30)
            )
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        policy_end = self.policy_obs_dim
        left_end = policy_end + 37
        right_end = left_end + 37
        waist_end = right_end + self.waist_feature_dim
        support_end = waist_end + self.support_feature_dim
        policy_x = x[..., :policy_end]
        left_features = x[..., policy_end:left_end]
        right_features = x[..., left_end:right_end]
        waist_features = x[..., right_end:waist_end]
        support_features = x[..., waist_end:support_end]
        # new change
        if self.normalized_direct_support:
            support_head_features = (
                support_features - self.support_feature_mean
            ) / self.support_feature_scale.clamp_min(1.0e-6)
        else:
            support_head_features = support_features

        output = self.mlp(self.obs_normalizer(policy_x))
        left_raw = torch.cat(
            (
                self.left_arm_residual_adapter["shoulder"](left_features),
                self.left_arm_residual_adapter["elbow"](left_features),
                self.left_arm_residual_adapter["wrist"](left_features),
            ),
            dim=-1,
        )
        right_raw = torch.cat(
            (
                self.right_arm_residual_adapter["shoulder"](right_features),
                self.right_arm_residual_adapter["elbow"](right_features),
                self.right_arm_residual_adapter["wrist"](right_features),
            ),
            dim=-1,
        )
        waist_raw = torch.cat(
            (
                self.waist_residual_adapter["yaw"](waist_features),
                self.waist_residual_adapter["roll_pitch"](waist_features),
            ),
            dim=-1,
        )
        support_raw = torch.cat(
            (
                self.support_residual_adapter["sagittal"](support_head_features),
                self.support_residual_adapter["lateral"](support_head_features),
            ),
            dim=-1,
        )
        left = torch.tanh(left_raw) * self.left_arm_adapter_limits
        right = torch.tanh(right_raw) * self.right_arm_adapter_limits
        waist = torch.tanh(waist_raw) * self.waist_adapter_limits
        # new change
        if self.recovery_aware_support:
            support_gate = torch.sigmoid(support_raw)
            support_candidate_rad = torch.clamp(
                -self.support_candidate_gain * support_features[..., 20:30],
                min=-float(self.support_candidate_limit_rad),
                max=float(self.support_candidate_limit_rad),
            )
            support = (
                support_candidate_rad
                * support_gate
                / self.support_action_scale_rad
            )
        else:
            support = (
                torch.tanh(support_raw)
                * self.support_adapter_limits
                * self.support_output_scale
            )
        output = output + left @ self.left_arm_adapter_projection
        output = output + right @ self.right_arm_adapter_projection
        output = output + waist @ self.waist_adapter_projection
        output = output + support @ self.support_adapter_projection
        return self.deterministic_output(output)

    @torch.jit.export
    def reset(self) -> None:
        pass


# new change
class _StandCoordinationWaistSupportOnnxExport(
    _StandCoordinationWaistSupportExport
):
    is_recurrent: bool = False

    def __init__(
        self,
        model: StandCoordinationStateAwareWaistSupportResidualAdapterMLPModel,
        verbose: bool,
    ) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = (
            model.obs_dim + 74 + model.waist_feature_dim + model.support_feature_dim
        )

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs_policy_plus_arms_plus_waist_plus_support"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
