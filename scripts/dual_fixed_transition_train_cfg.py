#new change
"""Independent full-lower-network adaptation to audited fixed arm reference."""
import gymnasium as gym
from isaaclab.utils import configclass
from dual_fixed_transition_cfg import FixedTransitionCfg
from neutral_dual_adapt_cfg import NeutralDualAdaptRunnerCfg

TASK='Isaac-G1-FixedTransition476-FrozenArms-LowerAdapt-v0'

@configclass
class FixedTransitionLowerRunnerCfg(NeutralDualAdaptRunnerCfg):
    max_iterations=50
    save_interval=5
    # Same experiment root permits loading the accepted model_299 run.
    # A distinct run_name creates separate output; no checkpoint is overwritten.
    def __post_init__(self):
        super().__post_init__()
        self.save_interval=5
        self.max_iterations=50

gym.register(id=TASK,entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
    kwargs={'env_cfg_entry_point':FixedTransitionCfg,
            'rsl_rl_cfg_entry_point':FixedTransitionLowerRunnerCfg})
