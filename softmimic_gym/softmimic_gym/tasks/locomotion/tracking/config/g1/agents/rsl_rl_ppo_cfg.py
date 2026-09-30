# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

# from softmimic_gym.rsl_rl_extensions.rl_cfg import RslRlPpoStudentTeacherCfg, RslRlDistillAlgorithmCfg, RslRlPpoActorCriticPULSEHierarchicalCfg, RslRlPpoStudentTeacherPULSECfg, RslRlSymmetryCfg, RslRlPpoSymmetryAugAlgorithmCfg, RslRlPpoAMPAlgorithmCfg
import pathlib

# new change
from softmimic_gym.assets.robots.g1_compat import (
    G1_SOFTMIMIC_BILATERAL_ARM_ACTION_IDS,
    G1_SOFTMIMIC_STATIONARY_UPPER_ACTION_IDS,
    G1_SOFTMIMIC_WAIST_ACTION_IDS,
)

@configclass
class G1WBCRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 500
    experiment_name = "g1_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 512, 512],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002, #0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class G1StuntRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 500
    experiment_name = "g1_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 512, 512],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002, #0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        # learning_rate=3.0e-4,
        schedule="adaptive",
        # schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# new change
@configclass
class G1GeneralTrackingV1RunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO baseline for scratch multi-motion tracking."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "g1_general_tracking_v1"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.30,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 512, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# new change
@configclass
class G1StandCoordinationV1RunnerCfg(G1GeneralTrackingV1RunnerCfg):
    """Full-29DOF PPO configuration for the dedicated coordination model."""

    num_steps_per_env = 48
    max_iterations = 1000
    save_interval = 25
    experiment_name = "g1_stand_coordination_v1"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.08,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 512, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.10,
        entropy_coef=0.0,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1StandCoordinationV1NeutralBootstrapRunnerCfg(G1GeneralTrackingV1RunnerCfg):
    """Scratch active-balance stage sharing the final policy interface."""

    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 25
    experiment_name = "g1_stand_coordination_v1"


# new change
@configclass
class G1StandCoordinationAccuracyV1RunnerCfg(G1StandCoordinationV1RunnerCfg):
    """Twenty-five-iteration waist/wrist-only refinement from model_1375."""

    num_steps_per_env = 48
    max_iterations = 25
    save_interval = 5
    experiment_name = "g1_stand_coordination_v1"
    # new change
    # Strict 29DOF action rows: waist yaw/roll/pitch and bilateral wrist
    # roll/pitch/yaw.  The trunk, shoulders, elbows, and all leg outputs stay
    # frozen to preserve the accepted coordination/balance solution.
    actor_output_finetune_action_ids = sorted(
        list(G1_SOFTMIMIC_WAIST_ACTION_IDS) + [23, 24, 25, 26, 27, 28]
    )
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.02,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 512, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1StandCoordinationAccuracyV2RunnerCfg(G1StandCoordinationAccuracyV1RunnerCfg):
    """Supervised-bootstrap then adapter-only StandCoordination refinement."""

    # Keep the experiment root shared so model_1375 can be loaded directly;
    # the task id and run name still isolate every V2 artifact.
    experiment_name = "g1_stand_coordination_v1"
    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationBilateralStructuredArmResidualAdapterMLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.02,
            std_type="scalar",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 512, 512, 256],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
    )
    # The legacy actor, waist rows, and all lower-body rows stay immutable.
    actor_output_finetune_action_ids = None
    bilateral_structured_arm_adapter_only = True
    num_steps_per_env = 48
    max_iterations = 15
    save_interval = 3
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1StandCoordinationOnlineCompliantRunnerCfg(
    G1StandCoordinationAccuracyV2RunnerCfg
):
    """Adapter-only PPO over online-generated compliant rollouts."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationOnlineComplianceResidualAdapterMLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.DistributionCfg(
            class_name=(
                "softmimic_gym.rsl_rl_extensions.models:"
                "StandCoordinationOnlineComplianceMaskedGaussianDistribution"
            )
        ),
    )
    online_compliance_adapter_only = True
    bilateral_structured_arm_adapter_only = False
    num_steps_per_env = 48
    max_iterations = 20
    save_interval = 2
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.03,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1StandCoordinationCompliantStandLowForceArmsV1RunnerCfg(
    G1StandCoordinationOnlineCompliantRunnerCfg
):
    """Stage C1: train force-conditioned arm heads; keep waist/support frozen."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationOnlineComplianceResidualAdapterMLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.DistributionCfg(
            class_name=(
                "softmimic_gym.rsl_rl_extensions.models:"
                "StandCoordinationOnlineComplianceArmOnlyMaskedGaussianDistribution"
            )
        ),
    )
    online_compliance_adapter_only = False
    online_compliance_arm_only = True
    num_steps_per_env = 48
    max_iterations = 20
    save_interval = 2


# new change
@configclass
class G1StandCoordinationCompliantStandForceGatedArmsV1RunnerCfg(
    G1StandCoordinationCompliantStandLowForceArmsV1RunnerCfg
):
    """Inference/continuation wrapper with exact F=0 residual suppression."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationForceGatedOnlineComplianceResidualAdapterMLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.DistributionCfg(
            class_name=(
                "softmimic_gym.rsl_rl_extensions.models:"
                "StandCoordinationOnlineComplianceArmOnlyMaskedGaussianDistribution"
            )
        ),
    )


# new change
@configclass
class G1StandCoordinationAnalyticArmFeedbackAlpha010RunnerCfg(
    G1StandCoordinationCompliantStandForceGatedArmsV1RunnerCfg
):
    """Inference-only alpha=0.10, beta=0.01 s bilateral feedback probe."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationAnalyticArmFeedbackAlpha010MLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.DistributionCfg(
            class_name=(
                "softmimic_gym.rsl_rl_extensions.models:"
                "StandCoordinationOnlineComplianceArmOnlyMaskedGaussianDistribution"
            )
        ),
    )
    online_compliance_arm_only = False
    max_iterations = 0


# new change
@configclass
class G1StandCoordinationAnalyticArmFeedbackAlpha025RunnerCfg(
    G1StandCoordinationAnalyticArmFeedbackAlpha010RunnerCfg
):
    """Inference-only alpha=0.25, beta=0.01 s bilateral feedback probe."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationAnalyticArmFeedbackAlpha025MLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.DistributionCfg(
            class_name=(
                "softmimic_gym.rsl_rl_extensions.models:"
                "StandCoordinationOnlineComplianceArmOnlyMaskedGaussianDistribution"
            )
        ),
    )


# new change
@configclass
class G1StandCoordinationAnalyticArmFeedbackAlpha050RunnerCfg(
    G1StandCoordinationAnalyticArmFeedbackAlpha010RunnerCfg
):
    """Inference-only alpha=0.50, beta=0.01 s bilateral feedback probe."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationAnalyticArmFeedbackAlpha050MLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.DistributionCfg(
            class_name=(
                "softmimic_gym.rsl_rl_extensions.models:"
                "StandCoordinationOnlineComplianceArmOnlyMaskedGaussianDistribution"
            )
        ),
    )


# new change
@configclass
class G1StandCoordinationSupervisedArmFeedbackRunnerCfg(
    G1StandCoordinationAnalyticArmFeedbackAlpha010RunnerCfg
):
    """Frozen inference for the offline-distilled 71D bilateral arm heads."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationSupervisedArmFeedbackOnlineComplianceMLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.DistributionCfg(
            class_name=(
                "softmimic_gym.rsl_rl_extensions.models:"
                "StandCoordinationOnlineComplianceArmOnlyMaskedGaussianDistribution"
            )
        ),
    )
    max_iterations = 0


# new change
@configclass
class G1StandCoordinationSupervisedArmFeedbackShortPPORunnerCfg(
    G1StandCoordinationSupervisedArmFeedbackRunnerCfg
):
    """Five-iteration safety micro-tune of the two distilled arm heads only."""

    online_compliance_arm_only = True
    num_steps_per_env = 48
    max_iterations = 5
    save_interval = 1
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.03,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1StandCoordinationCompliantStandWaistCoordinationC2RunnerCfg(
    G1StandCoordinationCompliantStandForceGatedArmsV1RunnerCfg
):
    """Stage C2: preserve learned arms and train only force-gated waist 2 mrad."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationForceGatedOnlineComplianceResidualAdapterMLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.DistributionCfg(
            class_name=(
                "softmimic_gym.rsl_rl_extensions.models:"
                "StandCoordinationOnlineComplianceWaistOnlyMaskedGaussianDistribution"
            )
        ),
    )
    online_compliance_arm_only = False
    online_compliance_waist_only = True
    max_iterations = 10
    save_interval = 2


# new change
@configclass
class G1StandArmWaistCOMCoordinationRunnerCfg(
    G1StandCoordinationAccuracyV2RunnerCfg
):
    """Short protected co-optimization of proximal arms and tiny waist head."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationArmWaistCOMResidualAdapterMLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.01,
            std_type="scalar",
        ),
    )
    bilateral_structured_arm_adapter_only = False
    proximal_arm_waist_adapter_only = True
    actor_output_finetune_action_ids = None
    num_steps_per_env = 48
    max_iterations = 20
    save_interval = 2
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.03,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1StandArmWaistCOMCoordinationV2RunnerCfg(
    G1StandArmWaistCOMCoordinationRunnerCfg
):
    """State-aware V2 with enough update scale to test the new capacity."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationStateAwareArmWaistCOMResidualAdapterMLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.01,
            std_type="scalar",
        ),
    )
    max_iterations = 20
    save_interval = 2
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.03,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1StandArmWaistCOMCoordinationV3RunnerCfg(
    G1StandArmWaistCOMCoordinationV2RunnerCfg
):
    """V3 trains only bounded waist/support heads with structured noise."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationStateAwareWaistSupportResidualAdapterMLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.005,
            std_type="scalar",
        ),
    )
    proximal_arm_waist_adapter_only = False
    waist_support_adapter_only = True
    # Strict-29DOF waist plus bilateral hip/ankle pitch/roll rows.
    structured_exploration_action_ids = (
        0,
        1,
        2,
        3,
        4,
        5,
        8,
        13,
        14,
        17,
        18,
    )
    structured_exploration_active_std = 0.005
    structured_exploration_frozen_std = 1.0e-4
    max_iterations = 20
    save_interval = 2
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.03,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1StandArmWaistCOMCoordinationV3MaskedRunnerCfg(
    G1StandArmWaistCOMCoordinationV3RunnerCfg
):
    """Numerically isolated PPO over only the eleven V3 control rows."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationStateAwareWaistSupportResidualAdapterMLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.DistributionCfg(
            class_name=(
                "softmimic_gym.rsl_rl_extensions.models:"
                "StandCoordinationWaistSupportMaskedGaussianDistribution"
            )
        ),
    )
    # All Gaussian entries remain well-conditioned. The custom distribution
    # makes protected rows exactly deterministic and excludes them from PPO
    # log-probability, entropy, and KL calculations.
    structured_exploration_active_std = 0.005
    structured_exploration_frozen_std = 0.005


# new change
@configclass
class G1StandCoordinationStationarySupportAdapterRunnerCfg(
    G1StandArmWaistCOMCoordinationV3RunnerCfg
):
    """Train only the translational/contact-aware support head from model_1375."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationStationarySupportResidualAdapterMLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.DistributionCfg(
            class_name=(
                "softmimic_gym.rsl_rl_extensions.models:"
                "StandCoordinationSupportOnlyMaskedGaussianDistribution"
            )
        ),
    )
    waist_support_adapter_only = False
    stationary_support_adapter_only = True
    structured_exploration_action_ids = (0, 1, 3, 4, 13, 14, 17, 18)
    structured_exploration_active_std = 0.005
    structured_exploration_frozen_std = 0.005
    num_steps_per_env = 48
    max_iterations = 25
    save_interval = 2
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.03,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1StandCoordinationStationarySupportControlV2RunnerCfg(
    G1StandCoordinationStationarySupportAdapterRunnerCfg
):
    """Train only the complete fixed-foot 10DOF support head from model_1375."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationSupportControlV2ResidualAdapterMLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.DistributionCfg(
            class_name=(
                "softmimic_gym.rsl_rl_extensions.models:"
                "StandCoordinationSupportControlV2MaskedGaussianDistribution"
            )
        ),
    )
    structured_exploration_action_ids = (0, 1, 3, 4, 9, 10, 13, 14, 17, 18)
    # 0.01 raw action = 3 mrad physical exploration on the 0.30-rad
    # lower-body action scale.  This is large enough to expose support credit
    # while remaining below the safe +5 mrad knee finite-difference probe.
    structured_exploration_active_std = 0.01
    structured_exploration_frozen_std = 0.005
    num_steps_per_env = 48
    max_iterations = 20
    save_interval = 2
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.03,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1StandCoordinationStationarySupportControlV2SupervisedInitRunnerCfg(
    G1StandCoordinationStationarySupportControlV2RunnerCfg
):
    """Strict 6-mrad support model; PPO settings are prepared but not auto-run."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationSupportControlV2SupervisedInitMLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.DistributionCfg(
            class_name=(
                "softmimic_gym.rsl_rl_extensions.models:"
                "StandCoordinationSupportControlV2MaskedGaussianDistribution"
            )
        ),
    )
    structured_exploration_active_std = 0.005
    max_iterations = 10
    save_interval = 1
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.02,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1StandCoordinationStationaryFoundationV2RunnerCfg(
    G1StandCoordinationStationarySupportControlV2RunnerCfg
):
    """Direct 87D support residual while model_1375 and upper body stay frozen."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationStationaryFoundationV2MLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.DistributionCfg(
            class_name=(
                "softmimic_gym.rsl_rl_extensions.models:"
                "StandCoordinationSupportControlV2MaskedGaussianDistribution"
            )
        ),
    )
    structured_exploration_active_std = 0.005
    structured_exploration_frozen_std = 0.005
    num_steps_per_env = 48
    max_iterations = 50
    save_interval = 2
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.02,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1StandCoordinationStationarySupportContainmentV3RunnerCfg(
    G1StandCoordinationStationaryFoundationV2RunnerCfg
):
    """Short support-only stage after fixing stationary containment credit."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationStationarySupportContainmentV3MLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.DistributionCfg(
            class_name=(
                "softmimic_gym.rsl_rl_extensions.models:"
                "StandCoordinationSupportControlV2MaskedGaussianDistribution"
            )
        ),
    )
    # 0.002 action std corresponds to 0.6 mrad target exploration.
    structured_exploration_active_std = 0.002
    structured_exploration_frozen_std = 0.002
    num_steps_per_env = 48
    max_iterations = 15
    save_interval = 1
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.02,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1StandCoordinationStationarySupportActorRowsV4RunnerCfg(
    G1StandCoordinationStationarySupportContainmentV3RunnerCfg
):
    """Fine-tune only model_1375's ten legacy support output rows."""

    # Use the original MLP actor parameterization so model_1375 loads exactly;
    # only the selected final-layer rows are optimizer-visible.
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.DistributionCfg(
            class_name=(
                "softmimic_gym.rsl_rl_extensions.models:"
                "StandCoordinationSupportControlV2MaskedGaussianDistribution"
            )
        ),
    )
    stationary_support_adapter_only = False
    actor_output_finetune_action_ids = (
        0,
        1,
        3,
        4,
        9,
        10,
        13,
        14,
        17,
        18,
    )
    structured_exploration_action_ids = actor_output_finetune_action_ids
    # 0.005 actor units = 1.5 mrad on the 0.30-rad support scale.
    structured_exploration_active_std = 0.005
    structured_exploration_frozen_std = 1.0e-5
    num_steps_per_env = 48
    max_iterations = 25
    save_interval = 1
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.02,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1StandCoordinationStationarySupportStateResidualV5RunnerCfg(
    G1StandCoordinationStationarySupportContainmentV3RunnerCfg
):
    """Train only the state-visible, unit-gradient bounded support head."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationStationarySupportStateResidualV5MLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.DistributionCfg(
            class_name=(
                "softmimic_gym.rsl_rl_extensions.models:"
                "StandCoordinationSupportControlV2MaskedGaussianDistribution"
            )
        ),
    )
    stationary_support_adapter_only = True
    actor_output_finetune_action_ids = None
    structured_exploration_action_ids = (
        0,
        1,
        3,
        4,
        9,
        10,
        13,
        14,
        17,
        18,
    )
    # Three-milliradian exploration, while deterministic residuals remain
    # independently bounded to +/-30 mrad.
    structured_exploration_active_std = 0.01
    structured_exploration_frozen_std = 1.0e-5
    num_steps_per_env = 48
    max_iterations = 25
    save_interval = 1
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.02,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1StandCoordinationStationaryBaseStateAwareV6RunnerCfg(
    G1StandCoordinationStationarySupportStateResidualV5RunnerCfg
):
    """Learn the state-aware base head from a supervised safe direction."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationStationaryBaseStateAwareV6MLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.DistributionCfg(
            class_name=(
                "softmimic_gym.rsl_rl_extensions.models:"
                "StandCoordinationSupportControlV2MaskedGaussianDistribution"
            )
        ),
    )
    stationary_support_adapter_only = True
    actor_output_finetune_action_ids = None
    structured_exploration_action_ids = (
        0,
        1,
        3,
        4,
        9,
        10,
        13,
        14,
        17,
        18,
    )
    # 0.02 actor units equals 6 mrad of physical target exploration.
    structured_exploration_active_std = 0.02
    structured_exploration_frozen_std = 1.0e-5
    num_steps_per_env = 48
    max_iterations = 100
    save_interval = 5
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.10,
        entropy_coef=0.0,
        num_learning_epochs=2,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1StandCoordinationStationaryCriticAwareV7RunnerCfg(
    G1StandCoordinationStationaryBaseStateAwareV6RunnerCfg
):
    """V6 support actor with a failure-observable critic and 2.56 s rollout."""

    # Median containment occurs near 2.2 s.  A 48-step (0.96 s) rollout never
    # contained that event; 128 steps lets GAE observe the delayed outcome.
    num_steps_per_env = 128
    max_iterations = 50
    save_interval = 5


# new change
@configclass
class G1StandCoordinationSupportRecoveryAwareTeacherV3RunnerCfg(
    G1StandCoordinationStationarySupportControlV2SupervisedInitRunnerCfg
):
    """Recovery-aware 87D head; PPO remains conditional on multi-seed acceptance."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "StandCoordinationSupportRecoveryAwareTeacherV3MLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.DistributionCfg(
            class_name=(
                "softmimic_gym.rsl_rl_extensions.models:"
                "StandCoordinationSupportControlV2MaskedGaussianDistribution"
            )
        ),
    )
    # No PPO command is issued until all three 512x1500 inference gates pass.
    max_iterations = 0
    save_interval = 1


# new change
@configclass
class G1GeneralTrackingV1StationarySafeFineTuneArmsRunnerCfg(
    G1GeneralTrackingV1RunnerCfg
):
    """Stage S1: adapt only fourteen arm output rows from model_1598."""

    num_steps_per_env = 24
    max_iterations = 50
    save_interval = 5
    actor_output_finetune_action_ids = list(G1_SOFTMIMIC_BILATERAL_ARM_ACTION_IDS)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.05,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 512, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1GeneralTrackingV1StationarySafeFineTuneWaistArmsRunnerCfg(
    G1GeneralTrackingV1StationarySafeFineTuneArmsRunnerCfg
):
    """Stage S2: reopen the three waist rows after the arm-only gate passes."""

    actor_output_finetune_action_ids = list(G1_SOFTMIMIC_STATIONARY_UPPER_ACTION_IDS)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1GeneralTrackingV1StationaryArmsCreditIsolatedRunnerCfg(
    G1GeneralTrackingV1StationarySafeFineTuneArmsRunnerCfg
):
    """Short S1b probe with isolated arm credit and unchanged row mask."""

    max_iterations = 25
    save_interval = 5
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.02,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 512, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.5,
    )


# new change
@configclass
class G1GeneralTrackingV1StationaryStructuredArmsAdapterRunnerCfg(
    G1GeneralTrackingV1StationaryArmsCreditIsolatedRunnerCfg
):
    """Train only two zero-initialized 7DOF arm feedback adapters."""

    actor = RslRlMLPModelCfg(
        class_name=(
            "softmimic_gym.rsl_rl_extensions.models:"
            "GeneralBilateralStructuredArmResidualAdapterMLPModel"
        ),
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.02,
            std_type="scalar",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 512, 512, 256],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
    )
    bilateral_structured_arm_adapter_only = True
    # new change
    # Disable the inherited output-row mask: only the two new adapters train.
    actor_output_finetune_action_ids = None
    num_steps_per_env = 24
    max_iterations = 50
    save_interval = 5
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0,
        num_learning_epochs=1,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.5,
    )


# new change
def _analytic_probe_actor(model_class_name: str) -> RslRlMLPModelCfg:
    return RslRlMLPModelCfg(
        class_name=f"softmimic_gym.rsl_rl_extensions.models:{model_class_name}",
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.02, std_type="scalar"
        ),
    )


# new change
@configclass
class G1GeneralTrackingV1ArmErrorFeedbackAlpha010RunnerCfg(
    G1GeneralTrackingV1StationaryStructuredArmsAdapterRunnerCfg
):
    actor = _analytic_probe_actor("GeneralBilateralArmErrorFeedbackAlpha010MLPModel")
    bilateral_structured_arm_adapter_only = False
    max_iterations = 0


# new change
@configclass
class G1GeneralTrackingV1ArmErrorFeedbackAlpha025RunnerCfg(
    G1GeneralTrackingV1StationaryStructuredArmsAdapterRunnerCfg
):
    actor = _analytic_probe_actor("GeneralBilateralArmErrorFeedbackAlpha025MLPModel")
    bilateral_structured_arm_adapter_only = False
    max_iterations = 0


# new change
@configclass
class G1GeneralTrackingV1ArmErrorFeedbackAlpha050RunnerCfg(
    G1GeneralTrackingV1StationaryStructuredArmsAdapterRunnerCfg
):
    actor = _analytic_probe_actor("GeneralBilateralArmErrorFeedbackAlpha050MLPModel")
    bilateral_structured_arm_adapter_only = False
    max_iterations = 0

@configclass
class G1CSEStuntRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 500
    experiment_name = "g1_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 512, 512],
        activation="elu",
        class_name="ActorCriticCSE",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002, #0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        class_name="PPOCSE",
    )
