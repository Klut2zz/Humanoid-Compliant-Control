#new change
"""Recovery-only curriculum extension. No reference, physics or reward changes."""
import gymnasium as gym
import torch
from isaaclab.utils import configclass
from hold75_lower_stability_cfg import Hold75LowerStabilityCfg, Hold75LowerStabilityRunnerCfg
from softmimic_gym.tasks.locomotion.tracking.mdp.events import reset_robot_state_by_command

def recovery_stage(step):
    # Local environment steps, independent of the resumed checkpoint iteration.
    return 0 if step<240 else (1 if step<480 else 2)

def reset_hold_recovery(env,env_ids,pose_range,velocity_range,jpos_range,jvel_range,command_name):
    reset_robot_state_by_command(env,env_ids,pose_range,velocity_range,jpos_range,jvel_range,command_name)
    if env_ids is None: env_ids=torch.arange(env.num_envs,device=env.device)
    if len(env_ids)==0:return
    stage=recovery_stage(int(env.common_step_counter))
    bound=(.005,.010,.020)[stage]
    moving=torch.rand(len(env_ids),device=env.device)>=.5
    velocity=torch.zeros((len(env_ids),6),device=env.device)
    velocity[:,0]=-torch.rand(len(env_ids),device=env.device)*bound*moving
    env.scene['robot'].write_root_velocity_to_sim(velocity,env_ids=env_ids)
    assert velocity[:,0].min()>=-bound-1e-7 and velocity[:,0].max()<=0
    if getattr(env,'_hold_recovery_stage',None)!=stage:
        env._hold_recovery_stage=stage
        print('HOLD_RECOVERY_STAGE',{'local_step':int(env.common_step_counter),'stage':stage,
            'vx_range_mps':[-bound,0.],'zero_probability':.5,'applies_on_reset_only':True},flush=True)

@configclass
class Hold75BackwardRecoveryCfg(Hold75LowerStabilityCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.reset_robot.func=reset_hold_recovery

@configclass
class Hold75BackwardRecoveryRunnerCfg(Hold75LowerStabilityRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        assert self.num_steps_per_env==48
        self.max_iterations=30
        self.save_interval=2

assert [recovery_stage(s) for s in (0,239,240,479,480,1439)]==[0,0,1,1,2,2]
TASK='Isaac-G1-Hold75-BackwardRecovery-LowerStability-v0'
gym.register(id=TASK,entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
    kwargs={'env_cfg_entry_point':Hold75BackwardRecoveryCfg,'rsl_rl_cfg_entry_point':Hold75BackwardRecoveryRunnerCfg})
