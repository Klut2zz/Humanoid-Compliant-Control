"""Registration of deployable SoftMimic tasks."""

import gymnasium as gym

from . import agents, g1_force_control

# new change
from .general_tracking_v1 import (
    G1GeneralTrackingV1StationaryArmsCreditIsolatedEnvCfg,
    G1GeneralTrackingV1StationaryStructuredArmsAdapterEnvCfg,
    G1GeneralTrackingV1CoreEnvCfg,
    G1GeneralTrackingV1EnvCfg,
    G1GeneralTrackingV1LocomotionBootstrapEnvCfg,
    G1GeneralTrackingV1StationaryCoreEnvCfg,
    G1GeneralTrackingV1StandBootstrapEnvCfg,
    G1GeneralTrackingV1WalkBootstrapEnvCfg,
    # new change
    G1StandCoordinationAccuracyV1EnvCfg,
    # new change
    G1StandCoordinationAccuracyV2EnvCfg,
    # new change
    G1StandCoordinationLowDisturbanceDataEnvCfg,
    # new change
    G1StandCoordinationOnlineCompliantEnvCfg,
    # new change
    G1StandCoordinationOnlineCompliantZeroForceEnvCfg,
    # new change
    G1StandCoordinationOnlineCompliantVisualEnvCfg,
    # new change
    G1StandCoordinationOnlineArmLocalIKV2EnvCfg,
    # new change
    G1StandCoordinationCompliantStandLowForceArmsV1EnvCfg,
    # new change
    G1StandCoordinationCompliantStandLowForceArmsV1ZeroForceEnvCfg,
    # new change
    G1StandCoordinationFrictionABZeroForceBaselineEnvCfg,
    # new change
    G1StandCoordinationFrictionABZeroForceMediumEnvCfg,
    # new change
    G1StandCoordinationFrictionABZeroForceHighEnvCfg,
    # new change
    G1StandCoordinationFrictionABLowForceBaselineEnvCfg,
    # new change
    G1StandCoordinationFrictionABLowForceMediumEnvCfg,
    # new change
    G1StandCoordinationFrictionABLowForceHighEnvCfg,
    # new change
    G1StandCoordinationForceEventABLowForce2to4NEnvCfg,
    # new change
    G1StandCoordinationForceEventABSmooth2to4NEnvCfg,
    # new change
    G1StandCoordinationForceEventABSmooth4to7NEnvCfg,
    # new change
    G1StandCoordinationForceEventABSlopeMatchedSmooth2to4NEnvCfg,
    # new change
    G1StandCoordinationCompliantStandSmooth2to4NArmsV2EnvCfg,
    # new change
    G1StandCoordinationCompliantStandSmooth2to4NAnalyticArmFeedbackEnvCfg,
    # new change
    G1StandCoordinationCompliantStandSmooth4to7NArmFeedbackStressEnvCfg,
    # new change
    G1StandCoordinationCompliantStandSmooth4to7NArmFeedbackStressVisualEnvCfg,
    # new change
    G1StandCoordinationCompliantStandSmooth4to7NArmFeedbackFriction10x08EnvCfg,
    # new change
    G1StandCoordinationCompliantStandSmooth4to7NArmFeedbackFriction12x10EnvCfg,
    # new change
    G1StandCoordinationCompliantStandSmooth4to7NRecoveryFDEnvCfg,
    # new change
    G1StandCoordinationOnlineArmLocalIKV2BaseActorEnvCfg,
    # new change
    G1StandCoordinationOnlineArmLocalIKV2ZeroForceBaseActorEnvCfg,
    # new change
    G1StandCoordinationOnlineArmLocalIKV2ReferenceVisualEnvCfg,
    # new change
    G1StandCoordinationOnlineArmLocalIKV2ReferenceBatchEnvCfg,
    # new change
    G1StandCoordinationOnlineArmLocalIKV2ReleaseAmplitudeVisualEnvCfg,
    # new change
    G1StandCoordinationOnlineArmLocalIKV2ReleaseAmplitudeBatchEnvCfg,
    # new change
    G1StandCoordinationOnlineArmLocalIKV2LowVisualEnvCfg,
    # new change
    G1StandCoordinationOnlineArmLocalIKV2MediumVisualEnvCfg,
    # new change
    G1StandCoordinationOnlineArmLocalIKV2HighVisualEnvCfg,
    # new change
    G1StandCoordinationFrozenControllerLowForceClosedLoopEnvCfg,
    # new change
    G1StandCoordinationFrozenControllerZeroForceClosedLoopEnvCfg,
    # new change
    G1StandCoordinationStationarySupportDiagnosisEnvCfg,
    # new change
    G1StandCoordinationStationarySupportAdapterEnvCfg,
    # new change
    G1StandCoordinationStationarySupportControlV2EnvCfg,
    # new change
    G1StandCoordinationStationarySupportControlV2SupervisedInitEnvCfg,
    # new change
    G1StandCoordinationStationaryFoundationV2EnvCfg,
    # new change
    G1StandCoordinationStationarySupportContainmentV3EnvCfg,
    # new change
    G1StandCoordinationStationarySupportActorRowsV4EnvCfg,
    # new change
    G1StandCoordinationStationarySupportStateResidualV5EnvCfg,
    # new change
    G1StandCoordinationStationaryBaseStateAwareV6EnvCfg,
    # new change
    G1StandCoordinationStationaryCriticAwareV7EnvCfg,
    # new change
    G1StandCoordinationSupportRecoveryAwareTeacherV3EnvCfg,
    # new change
    G1StandCoordinationFrozenNeighborGateZeroForceEnvCfg,
    # new change
    G1StandCoordinationFrozenNeighborGateLowForceEnvCfg,
    # new change
    G1StandCoordinationV2FrozenInferenceZeroForceEnvCfg,
    # new change
    G1StandCoordinationV2FrozenInferenceLowForceEnvCfg,
    # new change
    G1StandArmWaistCOMCoordinationEnvCfg,
    # new change
    G1StandArmWaistCOMCoordinationV2EnvCfg,
    # new change
    G1StandArmWaistCOMCoordinationV3EnvCfg,
    G1StandCoordinationV1EnvCfg,
    G1StandCoordinationV1NeutralBootstrapEnvCfg,
)

_ENV_PREFIX = "Isaac-G1-AugmentedReference-ForceTorque-Control-VariableStiffness"
REGISTERED_TASK_IDS = []


def _compose_task_id(motion_meta: dict, mode_meta: dict) -> str:
    parts = [_ENV_PREFIX]
    motion_segment = motion_meta.get("id_segment")
    if motion_segment:
        parts.append(motion_segment)
    parts.append(mode_meta["id_segment"])
    return "-".join(parts) + "-Deployable-Mimic-v0"


for (motion_key, mode_key), env_cfg_cls in g1_force_control.AVAILABLE_DEPLOYABLE_ENV_CFGS.items():
    motion_meta = g1_force_control.SUPPORTED_MOTION_VARIANTS[motion_key]
    mode_meta = g1_force_control.SUPPORTED_MODE_VARIANTS[mode_key]
    task_id = _compose_task_id(motion_meta, mode_meta)
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_cls,
            "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.G1StuntRunnerCfg,
        },
    )
    REGISTERED_TASK_IDS.append(task_id)


# new change
GENERAL_TRACKING_V1_TASK_ID = "Isaac-G1-GeneralTrackingV1-29DOF-v0"
gym.register(
    id=GENERAL_TRACKING_V1_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1GeneralTrackingV1EnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.G1GeneralTrackingV1RunnerCfg,
    },
)
REGISTERED_TASK_IDS.append(GENERAL_TRACKING_V1_TASK_ID)


# new change
STAND_COORDINATION_V1_TASK_ID = "Isaac-G1-StandCoordinationV1-100Pct-29DOF-v0"
gym.register(
    id=STAND_COORDINATION_V1_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationV1EnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.G1StandCoordinationV1RunnerCfg,
    },
)
REGISTERED_TASK_IDS.append(STAND_COORDINATION_V1_TASK_ID)

# new change
STAND_COORDINATION_ACCURACY_V1_TASK_ID = (
    "Isaac-G1-StandCoordinationAccuracyV1-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_ACCURACY_V1_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationAccuracyV1EnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationAccuracyV1RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(STAND_COORDINATION_ACCURACY_V1_TASK_ID)

# new change
STAND_COORDINATION_ACCURACY_V2_TASK_ID = (
    "Isaac-G1-StandCoordinationAccuracyV2-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_ACCURACY_V2_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationAccuracyV2EnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationAccuracyV2RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(STAND_COORDINATION_ACCURACY_V2_TASK_ID)

# new change
STAND_COORDINATION_LOW_DISTURBANCE_DATA_TASK_ID = (
    "Isaac-G1-StandCoordinationLowDisturbanceData-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_LOW_DISTURBANCE_DATA_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationLowDisturbanceDataEnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationAccuracyV2RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(STAND_COORDINATION_LOW_DISTURBANCE_DATA_TASK_ID)

# new change
STAND_COORDINATION_ONLINE_COMPLIANT_TASK_ID = (
    "Isaac-G1-StandCoordinationOnlineCompliant-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_ONLINE_COMPLIANT_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationOnlineCompliantEnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationOnlineCompliantRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(STAND_COORDINATION_ONLINE_COMPLIANT_TASK_ID)

# new change
STAND_COORDINATION_ONLINE_COMPLIANT_ZERO_FORCE_TASK_ID = (
    "Isaac-G1-StandCoordinationOnlineCompliant-ZeroForce-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_ONLINE_COMPLIANT_ZERO_FORCE_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationOnlineCompliantZeroForceEnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationOnlineCompliantRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_ONLINE_COMPLIANT_ZERO_FORCE_TASK_ID
)

# new change
STAND_COORDINATION_ONLINE_COMPLIANT_VISUAL_TASK_ID = (
    "Isaac-G1-StandCoordinationOnlineCompliant-VisualQuality-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_ONLINE_COMPLIANT_VISUAL_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationOnlineCompliantVisualEnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationOnlineCompliantRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(STAND_COORDINATION_ONLINE_COMPLIANT_VISUAL_TASK_ID)

# new change
STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_TASK_ID = (
    "Isaac-G1-StandCoordinationOnlineCompliant-ArmLocalIKV2-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationOnlineArmLocalIKV2EnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationOnlineCompliantRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_TASK_ID)

# new change
STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_BASE_ACTOR_TASK_ID = (
    "Isaac-G1-StandCoordinationOnlineCompliant-ArmLocalIKV2-BaseActor-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_BASE_ACTOR_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationOnlineArmLocalIKV2BaseActorEnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationAccuracyV2RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_BASE_ACTOR_TASK_ID
)

# new change
STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_ZERO_FORCE_BASE_ACTOR_TASK_ID = (
    "Isaac-G1-StandCoordinationOnlineCompliant-ArmLocalIKV2-ZeroForce-BaseActor-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_ZERO_FORCE_BASE_ACTOR_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationOnlineArmLocalIKV2ZeroForceBaseActorEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationAccuracyV2RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_ZERO_FORCE_BASE_ACTOR_TASK_ID
)

# new change
STAND_COORDINATION_COMPLIANT_STAND_LOW_FORCE_ARMS_V1_TASK_ID = (
    "Isaac-G1-CompliantStand-LowForce4to7N-ArmsV1-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_COMPLIANT_STAND_LOW_FORCE_ARMS_V1_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationCompliantStandLowForceArmsV1EnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationCompliantStandLowForceArmsV1RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_COMPLIANT_STAND_LOW_FORCE_ARMS_V1_TASK_ID
)

# new change
STAND_COORDINATION_COMPLIANT_STAND_LOW_FORCE_ARMS_V1_ZERO_TASK_ID = (
    "Isaac-G1-CompliantStand-ZeroForce-ArmsV1-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_COMPLIANT_STAND_LOW_FORCE_ARMS_V1_ZERO_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationCompliantStandLowForceArmsV1ZeroForceEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationCompliantStandLowForceArmsV1RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_COMPLIANT_STAND_LOW_FORCE_ARMS_V1_ZERO_TASK_ID
)

# new change
STAND_COORDINATION_COMPLIANT_STAND_FORCE_GATED_ARMS_V1_TASK_ID = (
    "Isaac-G1-CompliantStand-LowForce4to7N-ForceGatedArmsV1-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_COMPLIANT_STAND_FORCE_GATED_ARMS_V1_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationCompliantStandLowForceArmsV1EnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationCompliantStandForceGatedArmsV1RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_COMPLIANT_STAND_FORCE_GATED_ARMS_V1_TASK_ID
)

# new change
STAND_COORDINATION_COMPLIANT_STAND_FORCE_GATED_ARMS_V1_ZERO_TASK_ID = (
    "Isaac-G1-CompliantStand-ZeroForce-ForceGatedArmsV1-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_COMPLIANT_STAND_FORCE_GATED_ARMS_V1_ZERO_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationCompliantStandLowForceArmsV1ZeroForceEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationCompliantStandForceGatedArmsV1RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_COMPLIANT_STAND_FORCE_GATED_ARMS_V1_ZERO_TASK_ID
)

# new change
_FRICTION_AB_ZERO_FORCE_TASKS = (
    (
        "Isaac-G1-FrictionAB-ZeroForce-Baseline08x06-Model1375-100Pct-29DOF-v0",
        G1StandCoordinationFrictionABZeroForceBaselineEnvCfg,
    ),
    (
        "Isaac-G1-FrictionAB-ZeroForce-Medium11x09-Model1375-100Pct-29DOF-v0",
        G1StandCoordinationFrictionABZeroForceMediumEnvCfg,
    ),
    (
        "Isaac-G1-FrictionAB-ZeroForce-High14x12-Model1375-100Pct-29DOF-v0",
        G1StandCoordinationFrictionABZeroForceHighEnvCfg,
    ),
)
for task_id, env_cfg in _FRICTION_AB_ZERO_FORCE_TASKS:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": (
                agents.rsl_rl_ppo_cfg.G1StandCoordinationAccuracyV2RunnerCfg
            ),
        },
    )
    REGISTERED_TASK_IDS.append(task_id)

# new change
_FRICTION_AB_LOW_FORCE_TASKS = (
    (
        "Isaac-G1-FrictionAB-LowForce4to7N-Baseline08x06-ForceGate-100Pct-29DOF-v0",
        G1StandCoordinationFrictionABLowForceBaselineEnvCfg,
    ),
    (
        "Isaac-G1-FrictionAB-LowForce4to7N-Medium11x09-ForceGate-100Pct-29DOF-v0",
        G1StandCoordinationFrictionABLowForceMediumEnvCfg,
    ),
    (
        "Isaac-G1-FrictionAB-LowForce4to7N-High14x12-ForceGate-100Pct-29DOF-v0",
        G1StandCoordinationFrictionABLowForceHighEnvCfg,
    ),
)
for task_id, env_cfg in _FRICTION_AB_LOW_FORCE_TASKS:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": (
                agents.rsl_rl_ppo_cfg.G1StandCoordinationCompliantStandForceGatedArmsV1RunnerCfg
            ),
        },
    )
    REGISTERED_TASK_IDS.append(task_id)

# new change
STAND_COORDINATION_FORCE_EVENT_AB_LOW_2TO4_TASK_ID = (
    "Isaac-G1-ForceEventAB-LowForce2to4N-Baseline08x06-ForceGate-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_FORCE_EVENT_AB_LOW_2TO4_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationForceEventABLowForce2to4NEnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationCompliantStandForceGatedArmsV1RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(STAND_COORDINATION_FORCE_EVENT_AB_LOW_2TO4_TASK_ID)

# new change
_FORCE_EVENT_HALF_COSINE_AB_TASKS = (
    (
        "Isaac-G1-ForceEventAB-SmoothHalfCosine-LowForce2to4N-Baseline08x06-ForceGate-100Pct-29DOF-v0",
        G1StandCoordinationForceEventABSmooth2to4NEnvCfg,
    ),
    (
        "Isaac-G1-ForceEventAB-SmoothHalfCosine-LowForce4to7N-Baseline08x06-ForceGate-100Pct-29DOF-v0",
        G1StandCoordinationForceEventABSmooth4to7NEnvCfg,
    ),
)
for task_id, env_cfg in _FORCE_EVENT_HALF_COSINE_AB_TASKS:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": (
                agents.rsl_rl_ppo_cfg.G1StandCoordinationCompliantStandForceGatedArmsV1RunnerCfg
            ),
        },
    )
    REGISTERED_TASK_IDS.append(task_id)

# new change
STAND_COORDINATION_FORCE_EVENT_AB_SLOPE_MATCHED_SMOOTH_2TO4_TASK_ID = (
    "Isaac-G1-ForceEventAB-SlopeMatchedHalfCosine-LowForce2to4N-Baseline08x06-ForceGate-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_FORCE_EVENT_AB_SLOPE_MATCHED_SMOOTH_2TO4_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationForceEventABSlopeMatchedSmooth2to4NEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationCompliantStandForceGatedArmsV1RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_FORCE_EVENT_AB_SLOPE_MATCHED_SMOOTH_2TO4_TASK_ID
)

# new change
STAND_COORDINATION_COMPLIANT_STAND_SMOOTH_2TO4_ARMS_V2_TASK_ID = (
    "Isaac-G1-CompliantStand-SmoothHalfCosine-LowForce2to4N-ForceGatedArmsV2-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_COMPLIANT_STAND_SMOOTH_2TO4_ARMS_V2_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationCompliantStandSmooth2to4NArmsV2EnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationCompliantStandForceGatedArmsV1RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_COMPLIANT_STAND_SMOOTH_2TO4_ARMS_V2_TASK_ID
)

# new change
_STAND_COORDINATION_ANALYTIC_ARM_FEEDBACK_TASKS = (
    (
        "Isaac-G1-CompliantStand-SmoothHalfCosine-LowForce2to4N-AnalyticArmFeedbackAlpha010-100Pct-29DOF-v0",
        agents.rsl_rl_ppo_cfg.G1StandCoordinationAnalyticArmFeedbackAlpha010RunnerCfg,
    ),
    (
        "Isaac-G1-CompliantStand-SmoothHalfCosine-LowForce2to4N-AnalyticArmFeedbackAlpha025-100Pct-29DOF-v0",
        agents.rsl_rl_ppo_cfg.G1StandCoordinationAnalyticArmFeedbackAlpha025RunnerCfg,
    ),
    (
        "Isaac-G1-CompliantStand-SmoothHalfCosine-LowForce2to4N-AnalyticArmFeedbackAlpha050-100Pct-29DOF-v0",
        agents.rsl_rl_ppo_cfg.G1StandCoordinationAnalyticArmFeedbackAlpha050RunnerCfg,
    ),
)
for task_id, runner_cfg in _STAND_COORDINATION_ANALYTIC_ARM_FEEDBACK_TASKS:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                G1StandCoordinationCompliantStandSmooth2to4NAnalyticArmFeedbackEnvCfg
            ),
            "rsl_rl_cfg_entry_point": runner_cfg,
        },
    )
    REGISTERED_TASK_IDS.append(task_id)

# new change
STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_TASK_ID = (
    "Isaac-G1-CompliantStand-SmoothHalfCosine-LowForce2to4N-"
    "SupervisedArmFeedback-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationCompliantStandSmooth2to4NAnalyticArmFeedbackEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationSupervisedArmFeedbackRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_TASK_ID)

# new change
STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_4TO7_STRESS_TASK_ID = (
    "Isaac-G1-CompliantStand-SmoothHalfCosine-MediumForce4to7N-"
    "SupervisedArmFeedback-Stress-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_4TO7_STRESS_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationCompliantStandSmooth4to7NArmFeedbackStressEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationSupervisedArmFeedbackRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_4TO7_STRESS_TASK_ID
)

# new change
STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_4TO7_STRESS_VISUAL_TASK_ID = (
    "Isaac-G1-CompliantStand-SmoothHalfCosine-MediumForce4to7N-"
    "SupervisedArmFeedback-StressVisual-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_4TO7_STRESS_VISUAL_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationCompliantStandSmooth4to7NArmFeedbackStressVisualEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationSupervisedArmFeedbackRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_4TO7_STRESS_VISUAL_TASK_ID
)

# new change
STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_4TO7_FRICTION10X08_TASK_ID = (
    "Isaac-G1-CompliantStand-SmoothHalfCosine-MediumForce4to7N-"
    "SupervisedArmFeedback-Friction10x08-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_4TO7_FRICTION10X08_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationCompliantStandSmooth4to7NArmFeedbackFriction10x08EnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationSupervisedArmFeedbackRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_4TO7_FRICTION10X08_TASK_ID
)

# new change
STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_4TO7_FRICTION12X10_TASK_ID = (
    "Isaac-G1-CompliantStand-SmoothHalfCosine-MediumForce4to7N-"
    "SupervisedArmFeedback-Friction12x10-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_4TO7_FRICTION12X10_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationCompliantStandSmooth4to7NArmFeedbackFriction12x10EnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationSupervisedArmFeedbackRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_4TO7_FRICTION12X10_TASK_ID
)

# new change
STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_4TO7_RECOVERY_FD_TASK_ID = (
    "Isaac-G1-CompliantStand-SmoothHalfCosine-MediumForce4to7N-"
    "SupervisedArmFeedback-RecoveryFD-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_4TO7_RECOVERY_FD_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationCompliantStandSmooth4to7NRecoveryFDEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationSupervisedArmFeedbackRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_4TO7_RECOVERY_FD_TASK_ID
)

# new change
STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_SHORT_PPO_TASK_ID = (
    "Isaac-G1-CompliantStand-SmoothHalfCosine-LowForce2to4N-"
    "SupervisedArmFeedback-ShortPPO-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_SHORT_PPO_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationCompliantStandSmooth2to4NAnalyticArmFeedbackEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationSupervisedArmFeedbackShortPPORunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_SUPERVISED_ARM_FEEDBACK_SHORT_PPO_TASK_ID
)

# new change
STAND_COORDINATION_COMPLIANT_STAND_WAIST_C2_TASK_ID = (
    "Isaac-G1-CompliantStand-LowForce4to7N-ForceGatedWaistC2-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_COMPLIANT_STAND_WAIST_C2_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationCompliantStandLowForceArmsV1EnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationCompliantStandWaistCoordinationC2RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(STAND_COORDINATION_COMPLIANT_STAND_WAIST_C2_TASK_ID)

# new change
STAND_COORDINATION_COMPLIANT_STAND_WAIST_C2_ZERO_TASK_ID = (
    "Isaac-G1-CompliantStand-ZeroForce-ForceGatedWaistC2-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_COMPLIANT_STAND_WAIST_C2_ZERO_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationCompliantStandLowForceArmsV1ZeroForceEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationCompliantStandWaistCoordinationC2RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_COMPLIANT_STAND_WAIST_C2_ZERO_TASK_ID
)

# new change
STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_REFERENCE_VISUAL_TASK_ID = (
    "Isaac-G1-StandCoordinationOnlineCompliant-ArmLocalIKV2-ReferenceVisual-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_REFERENCE_VISUAL_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationOnlineArmLocalIKV2ReferenceVisualEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationOnlineCompliantRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_REFERENCE_VISUAL_TASK_ID
)

# new change
STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_REFERENCE_BATCH_TASK_ID = (
    "Isaac-G1-StandCoordinationOnlineCompliant-ArmLocalIKV2-ReferenceBatch-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_REFERENCE_BATCH_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationOnlineArmLocalIKV2ReferenceBatchEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationOnlineCompliantRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_REFERENCE_BATCH_TASK_ID
)

# new change
STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_RELEASE_AMPLITUDE_VISUAL_TASK_ID = (
    "Isaac-G1-StandCoordinationOnlineCompliant-ArmLocalIKV2-ReleaseAmplitudeVisual-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_RELEASE_AMPLITUDE_VISUAL_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationOnlineArmLocalIKV2ReleaseAmplitudeVisualEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationOnlineCompliantRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_RELEASE_AMPLITUDE_VISUAL_TASK_ID
)

# new change
STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_RELEASE_AMPLITUDE_BATCH_TASK_ID = (
    "Isaac-G1-StandCoordinationOnlineCompliant-ArmLocalIKV2-ReleaseAmplitudeBatch-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_RELEASE_AMPLITUDE_BATCH_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationOnlineArmLocalIKV2ReleaseAmplitudeBatchEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationOnlineCompliantRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_ONLINE_ARM_LOCAL_IK_V2_RELEASE_AMPLITUDE_BATCH_TASK_ID
)

# new change
STAND_COORDINATION_FROZEN_CONTROLLER_LOW_FORCE_CLOSED_LOOP_TASK_ID = (
    "Isaac-G1-StandCoordination-FrozenController-LowForce4to7N-ClosedLoop-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_FROZEN_CONTROLLER_LOW_FORCE_CLOSED_LOOP_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationFrozenControllerLowForceClosedLoopEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationAccuracyV2RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_FROZEN_CONTROLLER_LOW_FORCE_CLOSED_LOOP_TASK_ID
)

# new change
STAND_COORDINATION_FROZEN_CONTROLLER_ZERO_FORCE_CLOSED_LOOP_TASK_ID = (
    "Isaac-G1-StandCoordination-FrozenController-ZeroForce-ClosedLoop-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_FROZEN_CONTROLLER_ZERO_FORCE_CLOSED_LOOP_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationFrozenControllerZeroForceClosedLoopEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationAccuracyV2RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_FROZEN_CONTROLLER_ZERO_FORCE_CLOSED_LOOP_TASK_ID
)

# new change
STAND_COORDINATION_STATIONARY_SUPPORT_DIAGNOSIS_TASK_ID = (
    "Isaac-G1-StandCoordination-StationarySupportDiagnosis-ZeroForce-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_STATIONARY_SUPPORT_DIAGNOSIS_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationStationarySupportDiagnosisEnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationAccuracyV2RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_STATIONARY_SUPPORT_DIAGNOSIS_TASK_ID
)

# new change
STAND_COORDINATION_STATIONARY_SUPPORT_ADAPTER_TASK_ID = (
    "Isaac-G1-StandCoordination-StationarySupportAdapter-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_STATIONARY_SUPPORT_ADAPTER_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationStationarySupportAdapterEnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationStationarySupportAdapterRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(STAND_COORDINATION_STATIONARY_SUPPORT_ADAPTER_TASK_ID)

# new change
STAND_COORDINATION_STATIONARY_SUPPORT_CONTROL_V2_TASK_ID = (
    "Isaac-G1-StandCoordination-StationarySupportControlV2-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_STATIONARY_SUPPORT_CONTROL_V2_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationStationarySupportControlV2EnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationStationarySupportControlV2RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_STATIONARY_SUPPORT_CONTROL_V2_TASK_ID
)

# new change
STAND_COORDINATION_STATIONARY_SUPPORT_CONTROL_V2_SUPERVISED_INIT_TASK_ID = (
    "Isaac-G1-StandCoordination-StationarySupportControlV2-SupervisedInit-"
    "100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_STATIONARY_SUPPORT_CONTROL_V2_SUPERVISED_INIT_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationStationarySupportControlV2SupervisedInitEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationStationarySupportControlV2SupervisedInitRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_STATIONARY_SUPPORT_CONTROL_V2_SUPERVISED_INIT_TASK_ID
)

# new change
STAND_COORDINATION_STATIONARY_FOUNDATION_V2_TASK_ID = (
    "Isaac-G1-StandCoordination-StationaryFoundationV2-"
    "100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_STATIONARY_FOUNDATION_V2_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationStationaryFoundationV2EnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationStationaryFoundationV2RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_STATIONARY_FOUNDATION_V2_TASK_ID
)

# new change
STAND_COORDINATION_STATIONARY_SUPPORT_CONTAINMENT_V3_TASK_ID = (
    "Isaac-G1-StandCoordination-StationarySupportContainmentV3-"
    "100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_STATIONARY_SUPPORT_CONTAINMENT_V3_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationStationarySupportContainmentV3EnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationStationarySupportContainmentV3RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_STATIONARY_SUPPORT_CONTAINMENT_V3_TASK_ID
)

# new change
STAND_COORDINATION_STATIONARY_SUPPORT_ACTOR_ROWS_V4_TASK_ID = (
    "Isaac-G1-StandCoordination-StationarySupportActorRowsV4-"
    "100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_STATIONARY_SUPPORT_ACTOR_ROWS_V4_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationStationarySupportActorRowsV4EnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationStationarySupportActorRowsV4RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_STATIONARY_SUPPORT_ACTOR_ROWS_V4_TASK_ID
)

# new change
STAND_COORDINATION_STATIONARY_SUPPORT_STATE_RESIDUAL_V5_TASK_ID = (
    "Isaac-G1-StandCoordination-StationarySupportStateResidualV5-"
    "100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_STATIONARY_SUPPORT_STATE_RESIDUAL_V5_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationStationarySupportStateResidualV5EnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationStationarySupportStateResidualV5RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_STATIONARY_SUPPORT_STATE_RESIDUAL_V5_TASK_ID
)

# new change
STAND_COORDINATION_STATIONARY_BASE_STATE_AWARE_V6_TASK_ID = (
    "Isaac-G1-StandCoordination-StationaryBaseStateAwareV6-"
    "100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_STATIONARY_BASE_STATE_AWARE_V6_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationStationaryBaseStateAwareV6EnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationStationaryBaseStateAwareV6RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_STATIONARY_BASE_STATE_AWARE_V6_TASK_ID
)

# new change
STAND_COORDINATION_STATIONARY_CRITIC_AWARE_V7_TASK_ID = (
    "Isaac-G1-StandCoordination-StationaryCriticAwareV7-"
    "100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_STATIONARY_CRITIC_AWARE_V7_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            G1StandCoordinationStationaryCriticAwareV7EnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationStationaryCriticAwareV7RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_STATIONARY_CRITIC_AWARE_V7_TASK_ID
)

# new change
STAND_COORDINATION_SUPPORT_RECOVERY_AWARE_TEACHER_V3_TASK_ID = (
    "Isaac-G1-StandCoordination-SupportRecoveryAwareTeacherV3-"
    "100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_SUPPORT_RECOVERY_AWARE_TEACHER_V3_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationSupportRecoveryAwareTeacherV3EnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationSupportRecoveryAwareTeacherV3RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_SUPPORT_RECOVERY_AWARE_TEACHER_V3_TASK_ID
)

# new change
STAND_COORDINATION_FROZEN_NEIGHBOR_GATE_ZERO_FORCE_TASK_ID = (
    "Isaac-G1-StandCoordination-FrozenNeighborGate-ZeroForce-"
    "100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_FROZEN_NEIGHBOR_GATE_ZERO_FORCE_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationFrozenNeighborGateZeroForceEnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationSupportRecoveryAwareTeacherV3RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_FROZEN_NEIGHBOR_GATE_ZERO_FORCE_TASK_ID
)

# new change
STAND_COORDINATION_FROZEN_NEIGHBOR_GATE_LOW_FORCE_TASK_ID = (
    "Isaac-G1-StandCoordination-FrozenNeighborGate-LowForce4to7N-"
    "100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_FROZEN_NEIGHBOR_GATE_LOW_FORCE_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationFrozenNeighborGateLowForceEnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationSupportRecoveryAwareTeacherV3RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_FROZEN_NEIGHBOR_GATE_LOW_FORCE_TASK_ID
)

# new change
STAND_COORDINATION_V2_FROZEN_INFERENCE_ZERO_FORCE_TASK_ID = (
    "Isaac-G1-StandCoordination-V2FrozenInference-ZeroForce-"
    "100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_V2_FROZEN_INFERENCE_ZERO_FORCE_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationV2FrozenInferenceZeroForceEnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationSupportRecoveryAwareTeacherV3RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_V2_FROZEN_INFERENCE_ZERO_FORCE_TASK_ID
)

# new change
STAND_COORDINATION_V2_FROZEN_INFERENCE_LOW_FORCE_TASK_ID = (
    "Isaac-G1-StandCoordination-V2FrozenInference-LowForce4to7N-"
    "100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_V2_FROZEN_INFERENCE_LOW_FORCE_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationV2FrozenInferenceLowForceEnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationSupportRecoveryAwareTeacherV3RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(
    STAND_COORDINATION_V2_FROZEN_INFERENCE_LOW_FORCE_TASK_ID
)

# new change
for _difficulty_name, _difficulty_cfg in (
    ("Low4to7N", G1StandCoordinationOnlineArmLocalIKV2LowVisualEnvCfg),
    ("Medium7to10N", G1StandCoordinationOnlineArmLocalIKV2MediumVisualEnvCfg),
    ("High10to15N", G1StandCoordinationOnlineArmLocalIKV2HighVisualEnvCfg),
):
    _task_id = (
        "Isaac-G1-StandCoordinationOnlineCompliant-ArmLocalIKV2-"
        f"{_difficulty_name}-ReferenceVisual-100Pct-29DOF-v0"
    )
    gym.register(
        id=_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _difficulty_cfg,
            "rsl_rl_cfg_entry_point": (
                agents.rsl_rl_ppo_cfg.G1StandCoordinationOnlineCompliantRunnerCfg
            ),
        },
    )
    REGISTERED_TASK_IDS.append(_task_id)

# new change
STAND_ARM_WAIST_COM_COORDINATION_TASK_ID = (
    "Isaac-G1-StandArmWaistCOMCoordination-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_ARM_WAIST_COM_COORDINATION_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandArmWaistCOMCoordinationEnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandArmWaistCOMCoordinationRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(STAND_ARM_WAIST_COM_COORDINATION_TASK_ID)

# new change
STAND_ARM_WAIST_COM_COORDINATION_V2_TASK_ID = (
    "Isaac-G1-StandArmWaistCOMCoordinationV2-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_ARM_WAIST_COM_COORDINATION_V2_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandArmWaistCOMCoordinationV2EnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandArmWaistCOMCoordinationV2RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(STAND_ARM_WAIST_COM_COORDINATION_V2_TASK_ID)

# new change
STAND_ARM_WAIST_COM_COORDINATION_V3_TASK_ID = (
    "Isaac-G1-StandArmWaistCOMCoordinationV3-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_ARM_WAIST_COM_COORDINATION_V3_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandArmWaistCOMCoordinationV3EnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandArmWaistCOMCoordinationV3RunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(STAND_ARM_WAIST_COM_COORDINATION_V3_TASK_ID)

# new change
STAND_ARM_WAIST_COM_COORDINATION_V3_MASKED_TASK_ID = (
    "Isaac-G1-StandArmWaistCOMCoordinationV3Masked-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_ARM_WAIST_COM_COORDINATION_V3_MASKED_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandArmWaistCOMCoordinationV3EnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandArmWaistCOMCoordinationV3MaskedRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(STAND_ARM_WAIST_COM_COORDINATION_V3_MASKED_TASK_ID)

# new change
STAND_COORDINATION_V1_NEUTRAL_BOOTSTRAP_TASK_ID = (
    "Isaac-G1-StandCoordinationV1-NeutralBootstrap-100Pct-29DOF-v0"
)
gym.register(
    id=STAND_COORDINATION_V1_NEUTRAL_BOOTSTRAP_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StandCoordinationV1NeutralBootstrapEnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1StandCoordinationV1NeutralBootstrapRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(STAND_COORDINATION_V1_NEUTRAL_BOOTSTRAP_TASK_ID)


# new change
STATIONARY_ARMS_CREDIT_ISOLATED_TASK_ID = (
    "Isaac-G1-GeneralTrackingV1-StationaryArmsCreditIsolated-29DOF-v0"
)
gym.register(
    id=STATIONARY_ARMS_CREDIT_ISOLATED_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1GeneralTrackingV1StationaryArmsCreditIsolatedEnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1GeneralTrackingV1StationaryArmsCreditIsolatedRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(STATIONARY_ARMS_CREDIT_ISOLATED_TASK_ID)


# new change
STATIONARY_STRUCTURED_ARMS_ADAPTER_TASK_ID = (
    "Isaac-G1-GeneralTrackingV1-StationaryStructuredArmsAdapter-29DOF-v0"
)
gym.register(
    id=STATIONARY_STRUCTURED_ARMS_ADAPTER_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1GeneralTrackingV1StationaryStructuredArmsAdapterEnvCfg,
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_ppo_cfg.G1GeneralTrackingV1StationaryStructuredArmsAdapterRunnerCfg
        ),
    },
)
REGISTERED_TASK_IDS.append(STATIONARY_STRUCTURED_ARMS_ADAPTER_TASK_ID)


# new change
for alpha_name, probe_runner_cfg in (
    ("Alpha010", agents.rsl_rl_ppo_cfg.G1GeneralTrackingV1ArmErrorFeedbackAlpha010RunnerCfg),
    ("Alpha025", agents.rsl_rl_ppo_cfg.G1GeneralTrackingV1ArmErrorFeedbackAlpha025RunnerCfg),
    ("Alpha050", agents.rsl_rl_ppo_cfg.G1GeneralTrackingV1ArmErrorFeedbackAlpha050RunnerCfg),
):
    probe_task_id = (
        f"Isaac-G1-GeneralTrackingV1-StationaryArmErrorFeedback-{alpha_name}-29DOF-v0"
    )
    gym.register(
        id=probe_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": G1GeneralTrackingV1StationaryStructuredArmsAdapterEnvCfg,
            "rsl_rl_cfg_entry_point": probe_runner_cfg,
        },
    )
    REGISTERED_TASK_IDS.append(probe_task_id)


# new change
# These two tasks share the StationaryCore environment contract but use
# different actor-row masks and conservative optimizers.  Separate task ids
# prevent an accidental full-network resume.
for safe_stage_name, safe_runner_cfg in (
    (
        "StationarySafeFineTune-Arms",
        agents.rsl_rl_ppo_cfg.G1GeneralTrackingV1StationarySafeFineTuneArmsRunnerCfg,
    ),
    (
        "StationarySafeFineTune-WaistArms",
        agents.rsl_rl_ppo_cfg.G1GeneralTrackingV1StationarySafeFineTuneWaistArmsRunnerCfg,
    ),
):
    safe_task_id = f"Isaac-G1-GeneralTrackingV1-{safe_stage_name}-29DOF-v0"
    gym.register(
        id=safe_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": G1GeneralTrackingV1StationaryCoreEnvCfg,
            "rsl_rl_cfg_entry_point": safe_runner_cfg,
        },
    )
    REGISTERED_TASK_IDS.append(safe_task_id)


# new change
for curriculum_name, env_cfg_cls in (
    ("StandBootstrap", G1GeneralTrackingV1StandBootstrapEnvCfg),
    ("WalkBootstrap", G1GeneralTrackingV1WalkBootstrapEnvCfg),
    ("LocomotionBootstrap", G1GeneralTrackingV1LocomotionBootstrapEnvCfg),
    ("Core", G1GeneralTrackingV1CoreEnvCfg),
    # new change
    ("StationaryCore", G1GeneralTrackingV1StationaryCoreEnvCfg),
):
    task_id = f"Isaac-G1-GeneralTrackingV1-{curriculum_name}-29DOF-v0"
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_cls,
            "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.G1GeneralTrackingV1RunnerCfg,
        },
    )
    REGISTERED_TASK_IDS.append(task_id)
