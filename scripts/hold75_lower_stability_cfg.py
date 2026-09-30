#new change
"""Add audited 75% held posture; preserve original recovery and replay tasks."""
import json
import numpy as np
import torch
from pathlib import Path
from neutral_dual_adapt_cfg import ROOT,ARM,NAMES
from hard_pose_stability_cfg import s as original_source, SixPosePitchAction, reset_backward_recovery

trace=ROOT/'logs/stand_coordination_v1/neutral_dual_adapt_v1/multipose_lower_balance/validation/model463_scale075_seed0_500.npz'
with np.load(trace) as data:
    assert data['joint_names'].tolist()==NAMES
    assert not data['terminated'][:176,11].any()
    pose=data['ref'][175,11].copy()
    assert np.max(np.abs(data['ref_vel'][175,11]))<1e-6
order=json.loads((ROOT/'datasets/motions_csv/stand_coordination_v1/manifest.json').read_text())['csv_joint_order']
neutral=np.loadtxt(ROOT/'datasets/motions_csv/neutral_stand_geometry_v1/neutral.csv',delimiter=',')
motion=np.repeat(neutral[:1],1601,axis=0)
for i in ARM: motion[:,7+order.index(NAMES[i])]=pose[i]
np.testing.assert_allclose(motion[:,7:36][:,[order.index(n) for n in NAMES]],np.broadcast_to(pose,(1601,29)),atol=1e-6,rtol=0)
folder=ROOT/'datasets/motions_csv/fixed75_model463_env11_step175'
folder.mkdir(exist_ok=True)
csv=folder/'reference.csv'
if csv.exists(): np.testing.assert_allclose(np.loadtxt(csv,delimiter=','),motion,atol=1e-11,rtol=0)
else: np.savetxt(csv,motion,delimiter=',',fmt='%.12f')
# Preserve the 75% task's exact existing phase-shaped wrist correction.
col=7+order.index('right_wrist_pitch_joint')
dynamic=np.loadtxt(ROOT/'datasets/motions_csv/dual_sagittal_csv_v2_75/reference.csv',delimiter=',')[:,col]
delta75=float((pose[NAMES.index('right_wrist_pitch_joint')]-dynamic.mean())*.005/np.max(abs(dynamic-dynamic.mean())))
assert abs(delta75)<=.00501

class Hold75PitchAction(SixPosePitchAction):
    def apply_actions(self):
        # Parent's modulo-10 classification is wrong for added slots: implement disjoint slots explicitly.
        from neutral_dual_adapt_cfg import FrozenArmsAction
        FrozenArmsAction.apply_actions(self)
        c=self._env.command_manager.get_term('upper_body_joints')
        group=c.motion_ids%15
        j=NAMES.index('right_wrist_pitch_joint')
        old=(c.dof_pos[:,1,j]-self._env.cfg.pitch_probe_center)*self._env.cfg.pitch_probe_gain
        full=(c.dof_pos[:,1,j]-.008737864077669903)*.0449389179755672
        delta=torch.where(group>=10,torch.full_like(old,delta75),torch.where(group>=5,full,old))
        assert delta.abs().max()<.00501
        self._env._pitch_probe_delta=delta.detach().clone()
        index=self._asset.joint_names.index(NAMES[j])
        target=c.dof_pos[:,1,j]+self._env._dual_arm_raw[:,self.arm_ids.index(index)]*.15+delta
        self._asset.set_joint_position_target(target[:,None],joint_ids=[index])

source=original_source
changes={
 "['fixed100_env11_step100']*4":"['fixed100_env11_step100']*4+['fixed75_model463_env11_step175']*5",
 'np.repeat(poses[:1,other],10,axis=0)':'np.repeat(poses[:1,other],15,axis=0)',
 ",100,11)]*4":",100,11)]*4+[(log/'multipose_lower_balance/validation/model463_scale075_seed0_500.npz',175,11)]*5",
 '[[0.,0.]]*10':'[[0.,0.]]*15',
 'self.motion_ids%10':'self.motion_ids%15',
 'minlength=10':'minlength=15',
 'HardPoseStabilityCfg':'Hold75LowerStabilityCfg',
 'HardPoseStabilityRunnerCfg':'Hold75LowerStabilityRunnerCfg',
 'Isaac-G1-HardPose100-LowerStability-v0':'Isaac-G1-Hold75-Replay-LowerStability-v0',
 'self.actions.joint_pos.class_type=SixPosePitchAction':'self.actions.joint_pos.class_type=Hold75PitchAction',
 'SIX_POSE_AUDIT':'HOLD75_REPLAY_AUDIT',
}
for old,new in changes.items():
    assert old in source,old
    source=source.replace(old,new)
exec(compile(source,__file__,'exec'),globals())
