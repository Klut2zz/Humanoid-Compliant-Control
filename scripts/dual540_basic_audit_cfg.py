#new change
"""Training-entry safety audit only: no optimizer or policy mutation."""
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from dual540_paired_events_cfg import Dual540PairedEventsCfg,capture_paired
from dual540_online_force_cfg import OnlineForceAction
from neutral_dual_adapt_cfg import ARM
from softmimic_gym.tasks.locomotion.tracking.mdp.observations import online_compliance_context

class AuditAction(OnlineForceAction):
    def apply_actions(self):
        super().apply_actions()
        c=self._env.command_manager.get_term('upper_body_joints')
        r=self._asset
        idx=c.selected_body_ids
        # local=r._external_force_b[c._env_ids,idx]  # legacy IsaacLab API
        #new change
        local=r.permanent_wrench_composer.composed_force_as_torch[c._env_ids,idx]
        world=quat_apply(r.data.body_quat_w[c._env_ids,idx],local)
        torch.testing.assert_close(world,c.force_world,atol=2e-5,rtol=1e-5)
        assert torch.isfinite(world).all()
        assert all(not p.requires_grad for p in self.upper.parameters())
        if not hasattr(self,'_audit_printed'):
            print('AUDIT_ACTUATION_PASS','local-to-world wrench matches current command; frozen upper parameters',flush=True)
            self._audit_printed=True

@configclass
class Dual540BasicAuditCfg(Dual540PairedEventsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.class_type=AuditAction

def capture_audit(e):
    c=e.command_manager.get_term('upper_body_joints')
    context=online_compliance_context(e,'upper_body_joints')
    assert context.shape==(e.num_envs,20) and torch.isfinite(context).all()
    torch.testing.assert_close(context[:,:3]*8,c.force_world,atol=1e-6,rtol=0)
    other=[i for i in range(29) if i not in ARM]
    torch.testing.assert_close(c.dof_pos[:,:,other],c._dof_pos[:,:,other],atol=0,rtol=0)
    if not hasattr(e,'_basic_audit_printed'):
        print('AUDIT_CONTEXT_PASS',{'policy_shape':list(e.obs_buf['policy'].shape),'context_shape':list(context.shape),'lower_waist_reference_unchanged':True},flush=True)
        e._basic_audit_printed=True
    out=capture_paired(e)
    out['audit_context']=context.detach().cpu().numpy().copy()
    return out
