#new change
"""Independent frozen 4--7 N probe. All original configurations are retained."""
import json
import gymnasium as gym
from isaaclab.utils import configclass
from stand_force_joint_v1_cfg import StandForceJointCfg

TASK = 'Isaac-G1-StandForceJointV1-Force4to7-Frozen-v0'


@configclass
class StandForceJointMediumCfg(StandForceJointCfg):
    def __post_init__(self):
        super().__post_init__()
        before = self.to_dict()
        command = self.commands.upper_body_joints
        assert tuple(command.force_range_n) == (2., 4.)
        assert command.force_envelope_profile == 'half_cosine'
        assert tuple(self.recovery_wait_range_s) == (3., 5.)
        assert command.physical_force_enabled
        command.force_range_n = (4., 7.)
        before['commands']['upper_body_joints']['force_range_n'] = (4., 7.)
        assert json.dumps(before, default=repr, sort_keys=True) == json.dumps(self.to_dict(), default=repr, sort_keys=True)
        print('ONLY_CONFIG_CHANGE force_range_n: [2,4] -> [4,7]', flush=True)


gym.register(id=TASK, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
             kwargs={'env_cfg_entry_point': StandForceJointMediumCfg})
