#new change
"""Independent frozen pressure task; all V2 controls and IK limits preserved."""
import json
import gymnasium as gym
from isaaclab.utils import configclass
from stand_force_recovery_mix_v2_cfg import StandForceRecoveryMixV2Cfg
from stand_force_pressure_sampling import PressureTelemetryMixin

TASK = 'Isaac-G1-StandForceJoint-RecoveryMixV2-Pressure7to10-Frozen-v0'


@configclass
class StandForcePressureCfg(StandForceRecoveryMixV2Cfg):
    recovery_sampling: str = 'medium'
    pressure_force_range_n: tuple[float, float] = (7., 10.)

    def __post_init__(self):
        assert self.recovery_sampling == 'medium'
        assert tuple(self.pressure_force_range_n) in ((4., 7.), (7., 10.))
        super().__post_init__()
        before = self.to_dict()
        command = self.commands.upper_body_joints
        parent = command.class_type

        class PressureEvents(PressureTelemetryMixin, parent):
            pass

        PressureEvents.pressure_force_range_n = tuple(self.pressure_force_range_n)
        command.class_type = PressureEvents
        after = self.to_dict()
        before['commands']['upper_body_joints']['class_type'] = after['commands']['upper_body_joints']['class_type']
        assert json.dumps(before, default=repr, sort_keys=True) == json.dumps(after, default=repr, sort_keys=True)
        print('PRESSURE_RANGE_ONLY', json.dumps(dict(requested_range_N=self.pressure_force_range_n,
            base_rng_range_N=command.force_range_n, stiffness_N_m=command.stiffness_range_n_per_m,
            displacement_limit_m=command.max_displacement_m, recovery_wait_s=self.recovery_wait_range_s,
            physical_force_enabled=command.physical_force_enabled, ik_limits_unchanged=True)), flush=True)


gym.register(id=TASK, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
             kwargs={'env_cfg_entry_point': StandForcePressureCfg})

