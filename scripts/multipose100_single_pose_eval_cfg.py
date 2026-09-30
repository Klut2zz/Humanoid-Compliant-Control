#new change
"""One six-pose reference, preserving its compensation and disabling reset noise."""
import os
from pathlib import Path
POSE_INDEX=int(os.environ['EVAL_POSE_INDEX'])
assert 0<=POSE_INDEX<6
path=Path(__file__).resolve().with_name('multipose100_lower_balance_cfg.py')
source=path.read_text()
assert source.count('full=(c.motion_ids%6)==5')==1
source=source.replace('full=(c.motion_ids%6)==5','full=torch.full_like(c.motion_ids,POSE_INDEX)==5')
changes={
 'class MultiPose100LowerBalanceCfg(':'class SinglePose100EvalCfg(',
 'c.demo_recording_path=[str(p) for p in paths]':'c.demo_recording_path=[str(paths[POSE_INDEX])]',
 'c.demo_start_range=[[0.,0.]]*6':'c.demo_start_range=[[0.,0.]]',
 'groups=self.motion_ids%6':'groups=torch.full_like(self.motion_ids,POSE_INDEX)',
 'self.events.reset_robot.func=reset_x_recovery':"from softmimic_gym.tasks.locomotion.tracking.mdp.events import reset_robot_state_by_command\n        self.events.reset_robot.func=reset_robot_state_by_command\n        assert all(tuple(v)==(0.,0.) for v in self.events.reset_robot.params['velocity_range'].values())\n        print('SINGLE_POSE_ZERO_VELOCITY',POSE_INDEX,flush=True)",
 "TASK='Isaac-G1-MultiPose100-FrozenArms-LowerBalance-v0'":"TASK='Isaac-G1-MultiPose100-SinglePose-Eval-v0'",
 "'env_cfg_entry_point':MultiPose100LowerBalanceCfg":"'env_cfg_entry_point':SinglePose100EvalCfg",
}
injection=''
for old,new in changes.items():
    injection+='assert source.count('+repr(old)+')==1\nsource=source.replace('+repr(old)+','+repr(new)+')\n'
needle="exec(compile(source,str(path),'exec'),globals())"
assert source.count(needle)==1
source=source.replace(needle,injection+needle)
exec(compile(source,str(path),'exec'),globals())
