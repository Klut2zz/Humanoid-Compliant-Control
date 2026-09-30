#new change
"""Independent lower-only recovery curriculum; originals remain untouched."""
from pathlib import Path
import torch
from softmimic_gym.tasks.locomotion.tracking.mdp.events import reset_robot_state_by_command

def reset_backward_recovery(env,env_ids,pose_range,velocity_range,jpos_range,jvel_range,command_name):
    reset_robot_state_by_command(env,env_ids,pose_range,velocity_range,jpos_range,jvel_range,command_name)
    if env_ids is None: env_ids=torch.arange(env.num_envs,device=env.device)
    if len(env_ids)==0:return
    # 50% stationary replay; backward velocity only, no angular/lateral disturbance.
    stage=min(int(env.common_step_counter)//480,2)
    bound=(.005,.010,.020)[stage]
    moving=torch.rand(len(env_ids),device=env.device)>=.5
    velocity=torch.zeros((len(env_ids),6),device=env.device)
    velocity[:,0]=-torch.rand(len(env_ids),device=env.device)*bound*moving
    env.scene['robot'].write_root_velocity_to_sim(velocity,env_ids=env_ids)
    if getattr(env,'_backward_stage',None)!=stage:
        env._backward_stage=stage
        print('BACKWARD_RECOVERY',{'stage':stage,'bound':bound,'zero_probability':.5},flush=True)

# Reuse the audited pose/CSV checks verbatim, adding four hard-pose slots.
path=Path(__file__).with_name('multipose100_lower_balance_cfg.py')
source=path.read_text()
# Execute the original loader into a private namespace to obtain its checked source.
namespace={'__file__':str(path),'__name__':'hard_pose_source'}
# Stop before exec: no original task or class is modified.
prefix=source.rsplit("exec(compile(source,str(path),'exec'),globals())",1)[0]
exec(compile(prefix,str(path),'exec'),namespace)
s=namespace['source']
replacements={
 "'fixed100_env11_step100']":"'fixed100_env11_step100']+['fixed100_env11_step100']*4",
 'np.repeat(poses[:1,other],6,axis=0)':'np.repeat(poses[:1,other],10,axis=0)',
 ',100,11)]':',100,11)]+[(log/\'multipose_lower_balance/validation/model425_scale100_seed0_500.npz\',100,11)]*4',
 '[[0.,0.]]*6':'[[0.,0.]]*10',
 'self.motion_ids%6':'self.motion_ids%10',
 'minlength=6':'minlength=10',
 'MultiPose100LowerBalanceCfg':'HardPoseStabilityCfg',
 'MultiPose100RunnerCfg':'HardPoseStabilityRunnerCfg',
 'Isaac-G1-MultiPose100-FrozenArms-LowerBalance-v0':'Isaac-G1-HardPose100-LowerStability-v0',
 'self.events.reset_robot.func=reset_x_recovery':'self.events.reset_robot.func=reset_backward_recovery',
 'self.max_iterations=20':'self.max_iterations=20',
 'self.save_interval=5':'self.save_interval=2',
}
for old,new in replacements.items():
    assert old in s,old
    s=s.replace(old,new)
# The full-scale wrist compensation remains identical on every hard-pose slot.
action_source=source[source.index('class SixPosePitchAction'):source.index("path=Path(__file__)")]
action_source=action_source.replace('(c.motion_ids%6)==5','(c.motion_ids%10)>=5')
from neutral_dual_adapt_cfg import FrozenArmsAction
exec(compile(action_source,str(path),'exec'),globals())
exec(compile(s,str(path),'exec'),globals())
