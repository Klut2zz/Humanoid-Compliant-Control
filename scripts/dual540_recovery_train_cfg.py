#new change
"""Independent training registration; recovery inference and old tasks retained."""
import gymnasium as gym
from isaaclab.utils import configclass
from dual540_recovery_window_cfg import RecoveryWindowCfg

TASK='Isaac-G1-Dual540-WholeBodyForce-RecoveryWindow-Train-v0'
@configclass
class RecoveryWindowTrainCfg(RecoveryWindowCfg):
    pass

gym.register(id=TASK,entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
             kwargs={'env_cfg_entry_point':RecoveryWindowTrainCfg})
