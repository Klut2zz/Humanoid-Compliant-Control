#new change
"""Frozen single-variable comparison: extend only post-release waiting."""
import gymnasium as gym
from isaaclab.utils import configclass
from dual540_wholebody_force_cfg import WholeBodyForceCfg

TASK='Isaac-G1-Dual540-WholeBodyForce-RecoveryWindow-v0'

@configclass
class RecoveryWindowCfg(WholeBodyForceCfg):
    recovery_wait_range_s:tuple[float,float]=(3.,5.)
    def __post_init__(self):
        super().__post_init__()
        parent=self.commands.upper_body_joints.class_type
        original=tuple(self.commands.upper_body_joints.wait_time_range_s)
        extended=tuple(self.recovery_wait_range_s)
        assert original==(.5,1.5),original
        assert extended==(3.,5.),extended
        class RecoveryWindowCommand(parent):
            def _advance_events(self):
                self._sampling_after_natural_release=True
                try:super()._advance_events()
                finally:self._sampling_after_natural_release=False
            def _sample_events(self,env_ids):
                super()._sample_events(env_ids)
                # No extra RNG draw: preserve each event's force, hand, ramp,
                # hold and stiffness. Initial/reset waits stay at 0.5-1.5 s.
                if getattr(self,'_sampling_after_natural_release',False):
                    wait=-self._event_time[env_ids]
                    u=((wait-original[0])/(original[1]-original[0])).clamp(0,1)
                    self._event_time[env_ids]=-(extended[0]+u*(extended[1]-extended[0]))
        self.commands.upper_body_joints.class_type=RecoveryWindowCommand
        print('RECOVERY_WINDOW_CONFIG',{'initial_and_reset_wait_s':original,
              'post_release_wait_s':extended,'force_ramp_hold_controller_unchanged':True},flush=True)

gym.register(id=TASK,entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
             kwargs={'env_cfg_entry_point':RecoveryWindowCfg})
