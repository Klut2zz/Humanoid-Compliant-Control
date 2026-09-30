#new change
"""Reset-only curriculum; original rewards, actuators and frozen arms unchanged."""
import gymnasium as gym
import torch
from isaaclab.utils import configclass
from dual_fixed_transition_cfg import FixedTransitionCfg
from dual_fixed_transition_train_cfg import FixedTransitionLowerRunnerCfg
from softmimic_gym.tasks.locomotion.tracking.mdp.events import reset_robot_state_by_command

#new change: explicit signature required by IsaacLab event-manager validation.
def reset_x_recovery(env,env_ids,pose_range,velocity_range,jpos_range,jvel_range,command_name):
    reset_robot_state_by_command(env,env_ids,pose_range,velocity_range,jpos_range,jvel_range,command_name)
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    if len(env_ids)==0:return
    stage=min(int(env.common_step_counter)//480,2)
    bound=(.005,.010,.020)[stage]
    zero=torch.rand(len(env_ids),device=env.device)<.5
    velocity=torch.zeros((len(env_ids),6),device=env.device)
    velocity[:,0]=(2*torch.rand(len(env_ids),device=env.device)-1)*bound
    velocity[zero,0]=0
    env.scene['robot'].write_root_velocity_to_sim(velocity,env_ids=env_ids)
    assert velocity[:,0].abs().max()<=bound+1e-7
    if getattr(env,'_x_recovery_stage',None)!=stage:
        env._x_recovery_stage=stage
        print('X_RECOVERY_RESET',{'stage':stage,'bound_mps':bound,'zero_probability':.5,'batch':len(env_ids),'zero_count':int(zero.sum())},flush=True)

@configclass
class FixedTransitionVelocityRecoveryCfg(FixedTransitionCfg):
    def __post_init__(self):
        super().__post_init__()
        assert self.events.reset_robot.func is reset_robot_state_by_command
        self.events.reset_robot.func=reset_x_recovery

@configclass
class FixedTransitionVelocityRecoveryRunnerCfg(FixedTransitionLowerRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations=30
        self.save_interval=5
        assert self.num_steps_per_env==48

TASK='Isaac-G1-FixedTransition476-XVelocityRecovery-v0'
gym.register(id=TASK,entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
    kwargs={'env_cfg_entry_point':FixedTransitionVelocityRecoveryCfg,'rsl_rl_cfg_entry_point':FixedTransitionVelocityRecoveryRunnerCfg})
