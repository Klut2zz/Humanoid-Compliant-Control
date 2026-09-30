#new change
"""Add full-reference plateau coverage; preserve old replay and recovery settings."""
import torch
from hold75_lower_stability_cfg import source as previous_source, Hold75PitchAction, delta75
from hold75_backward_recovery_cfg import reset_hold_recovery
from neutral_dual_adapt_cfg import FrozenArmsAction, NAMES

class Hold100ReplayAction(Hold75PitchAction):
    def apply_actions(self):
        FrozenArmsAction.apply_actions(self)
        c=self._env.command_manager.get_term('upper_body_joints')
        group=c.motion_ids%20
        j=NAMES.index('right_wrist_pitch_joint')
        old=(c.dof_pos[:,1,j]-self._env.cfg.pitch_probe_center)*self._env.cfg.pitch_probe_gain
        full=(c.dof_pos[:,1,j]-.008737864077669903)*.0449389179755672
        delta=torch.where((group>=10)&(group<15),torch.full_like(old,delta75),torch.where(group>=5,full,old))
        assert delta.abs().max()<.00501
        self._env._pitch_probe_delta=delta.detach().clone()
        index=self._asset.joint_names.index(NAMES[j])
        target=c.dof_pos[:,1,j]+self._env._dual_arm_raw[:,self.arm_ids.index(index)]*.15+delta
        self._asset.set_joint_position_target(target[:,None],joint_ids=[index])

source=previous_source
changes={
 "['fixed75_model463_env11_step175']*5":"['fixed75_model463_env11_step175']*5+['fixed100_model511_env104_step175']*5",
 'np.repeat(poses[:1,other],15,axis=0)':'np.repeat(poses[:1,other],20,axis=0)',
 ',175,11)]*5':",175,11)]*5+[(log/'multipose_lower_balance/validation/model511_scale100_seed0_500.npz',175,104)]*5",
 '[[0.,0.]]*15':'[[0.,0.]]*20',
 'self.motion_ids%15':'self.motion_ids%20',
 'minlength=15':'minlength=20',
 'Hold75LowerStabilityCfg':'Hold100ReplayLowerCfg',
 'Hold75LowerStabilityRunnerCfg':'Hold100ReplayLowerRunnerCfg',
 'Isaac-G1-Hold75-Replay-LowerStability-v0':'Isaac-G1-Hold100-Replay-LowerStability-v0',
 'self.actions.joint_pos.class_type=Hold75PitchAction':'self.actions.joint_pos.class_type=Hold100ReplayAction',
 'self.events.reset_robot.func=reset_backward_recovery':'self.events.reset_robot.func=reset_hold_recovery',
 'self.max_iterations=20':'self.max_iterations=30',
 'HOLD75_REPLAY_AUDIT':'HOLD100_REPLAY_AUDIT',
}
for old,new in changes.items():
    assert old in source,old
    source=source.replace(old,new)
exec(compile(source,__file__,'exec'),globals())
