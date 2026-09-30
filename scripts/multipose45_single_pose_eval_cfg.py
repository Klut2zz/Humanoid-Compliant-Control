#new change
"""Single-pose deterministic evaluation; training curriculum explicitly disabled."""
import os
from pathlib import Path
POSE_INDEX=int(os.environ['EVAL_POSE_INDEX'])
assert 0<=POSE_INDEX<5
path=Path(__file__).resolve().with_name('multipose45_lower_balance_cfg.py')
source=path.read_text()
changes={
 'class MultiPose45LowerBalanceCfg(':'class SinglePose45EvalCfg(',
 "c.demo_recording_path=[str(p) for p in paths]":"c.demo_recording_path=[str(paths[POSE_INDEX])]",
 "c.demo_start_range=[[0.,0.]]*5":"c.demo_start_range=[[0.,0.]]",
 'groups=self.motion_ids%5':'groups=torch.full_like(self.motion_ids,POSE_INDEX)',
 'self.events.reset_robot.func=reset_x_recovery':"from softmimic_gym.tasks.locomotion.tracking.mdp.events import reset_robot_state_by_command\n        self.events.reset_robot.func=reset_robot_state_by_command\n        assert all(tuple(v)==(0.,0.) for v in self.events.reset_robot.params['velocity_range'].values())\n        print('SINGLE_POSE_ZERO_VELOCITY',POSE_INDEX,flush=True)",
 "TASK='Isaac-G1-MultiPose45-FrozenArms-LowerBalance-v0'":"TASK='Isaac-G1-MultiPose45-SinglePose-Eval-v0'",
 "'env_cfg_entry_point':MultiPose45LowerBalanceCfg":"'env_cfg_entry_point':SinglePose45EvalCfg",
}
for old,new in changes.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)
exec(compile(source,str(path),'exec'),globals())
