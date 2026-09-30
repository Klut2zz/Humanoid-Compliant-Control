#new change
"""Six-pose adaptation: keep five replay poses and add audited full-scale posture."""
from pathlib import Path
import torch
from neutral_dual_adapt_cfg import FrozenArmsAction

class SixPosePitchAction(FrozenArmsAction):
    def apply_actions(self):
        super().apply_actions()
        c=self._env.command_manager.get_term('upper_body_joints')
        j=NAMES.index('right_wrist_pitch_joint')
        # Preserve each source task's compensation, not amplitude-scaled torque/gains.
        full=(c.motion_ids%6)==5
        gain=torch.where(full,.0449389179755672,self._env.cfg.pitch_probe_gain)
        center=torch.where(full,.008737864077669903,self._env.cfg.pitch_probe_center)
        delta=(c.dof_pos[:,1,j]-center)*gain
        assert delta.abs().max()<.00501
        self._env._pitch_probe_delta=delta.detach().clone()
        index=self._asset.joint_names.index(NAMES[j])
        target=c.dof_pos[:,1,j]+self._env._dual_arm_raw[:,self.arm_ids.index(index)]*.15+delta
        self._asset.set_joint_position_target(target[:,None],joint_ids=[index])

path=Path(__file__).resolve().with_name('multipose45_lower_balance_cfg.py')
source=path.read_text()
changes={
 'MultiPose45LowerBalanceCfg':'MultiPose100LowerBalanceCfg',
 'MultiPose45RunnerCfg':'MultiPose100RunnerCfg',
 "'fixed45_paused_median_env1_step280']":"'fixed45_paused_median_env1_step280','fixed100_env11_step100']",
 'np.repeat(poses[:1,other],5,axis=0)':'np.repeat(poses[:1,other],6,axis=0)',
 "'multipose_lower_balance/validation/model406_scale045_branch250_pause_500.npz',280,1)]":"'multipose_lower_balance/validation/model406_scale045_branch250_pause_500.npz',280,1),\n                 (log/'multipose_lower_balance/validation/model425_scale100_seed0_500.npz',100,11)]",
 '[[0.,0.]]*5':'[[0.,0.]]*6',
 'self.motion_ids%5':'self.motion_ids%6',
 'minlength=5':'minlength=6',
 'FIVE_POSE_AUDIT':'SIX_POSE_AUDIT',
 'Isaac-G1-MultiPose45-FrozenArms-LowerBalance-v0':'Isaac-G1-MultiPose100-FrozenArms-LowerBalance-v0',
 'self.events.reset_robot.func=reset_x_recovery':'self.events.reset_robot.func=reset_x_recovery\n        self.actions.joint_pos.class_type=SixPosePitchAction',
}
for old,new in changes.items():
    assert old in source,old
    source=source.replace(old,new)
exec(compile(source,str(path),'exec'),globals())
