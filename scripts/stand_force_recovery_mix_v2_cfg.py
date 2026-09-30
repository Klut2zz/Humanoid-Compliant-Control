#new change
"""Independent V2 from V1 model50; only change mixed cohort proportions. No reward/physics/network changes."""
import json
import numpy as np
import gymnasium as gym
from isaaclab.utils import configclass
from stand_force_joint_v1_cfg import StandForceJointCfg, NEUTRAL
from stand_force_recovery_sampling_v2 import curriculum_cohort, transform_peaks

TASK = 'Isaac-G1-StandForceJoint-RecoveryMixV2-v0'


@configclass
class StandForceRecoveryMixV2Cfg(StandForceJointCfg):
    recovery_sampling: str = 'mixed'

    def __post_init__(self):
        assert self.recovery_sampling in ('mixed', 'zero', 'low', 'medium')
        self.zero_force_fraction = {'mixed': .25, 'zero': 1., 'low': 0., 'medium': 0.}[self.recovery_sampling]
        super().__post_init__()
        # The chosen backward direction must remain tied to this fixed pose.
        poses = np.loadtxt(NEUTRAL, delimiter=',')
        np.testing.assert_allclose(poses[:, 3:7], np.tile([0., 0., 0., 1.], (len(poses), 1)), atol=1e-10)
        before = self.to_dict()
        command = self.commands.upper_body_joints
        assert tuple(command.force_range_n) == (2., 4.)
        assert command.force_envelope_profile == 'half_cosine'
        assert tuple(self.recovery_wait_range_s) == (3., 5.)
        parent = command.class_type
        mode = self.recovery_sampling

        class RecoveryCurriculumCommand(parent):
            def _sample_events(self, env_ids):
                # Preserves per-env event RNG, IK resets and the 3--5 s window.
                super()._sample_events(env_ids)
                if not hasattr(self, '_recovery_cohort'):
                    self._recovery_cohort = curriculum_cohort(self._env_ids, mode)
                    self._recovery_rear_biased = self._solution_pending.clone().fill_(False)
                if env_ids.numel() == 0: return
                value, rear = transform_peaks(self._sampled_peak_force_world[env_ids],
                    self._event_stiffness[env_ids], float(self.cfg.max_displacement_m),
                    env_ids, self._independent_event_index[env_ids], int(self._env.cfg.seed), mode)
                self._sampled_peak_force_world[env_ids] = value
                self._recovery_rear_biased[env_ids] = rear

        command.class_type = RecoveryCurriculumCommand
        #new change: IsaacLab serializes callable class types to strings.
        after = self.to_dict()
        before['commands']['upper_body_joints']['class_type'] = after['commands']['upper_body_joints']['class_type']
        assert json.dumps(before, default=repr, sort_keys=True) == json.dumps(after, default=repr, sort_keys=True)
        print('RECOVERY_MIX_V2_CONFIG', json.dumps(dict(mode=mode,
            train_cohorts={'zero': .25, '2to4': .25, '4to7': .50},
            hard_rear_bias_probability=.5 if mode=='mixed' else 0.,
            reward_physics_action_observation_unchanged=True)), flush=True)


gym.register(id=TASK, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
             kwargs={'env_cfg_entry_point': StandForceRecoveryMixV2Cfg})
