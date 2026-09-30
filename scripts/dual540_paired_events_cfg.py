#new change
"""Per-environment, per-event deterministic RNG; original experiments retained."""
import os
import hashlib
import torch
from isaaclab.utils import configclass
from dual540_online_force_cfg import OnlineComplianceReferenceCommand
if os.environ['DUAL_ISOLATION']=='combined':
    from dual540_online_force_cfg import Dual540OnlineForceCfg as Base,capture_online as capture_base
else:
    from dual540_isolation_cfg import Dual540IsolationCfg as Base,capture_isolation as capture_base

def event_seed(seed,env_id,event_index):
    payload=f'dual540-force-v1:{seed}:{env_id}:{event_index}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],'little')%(2**63-1)

@configclass
class Dual540PairedEventsCfg(Base):
    def __post_init__(self):
        super().__post_init__()
        parent=self.commands.upper_body_joints.class_type
        class IndependentEvents(parent):
            def _sample_events(self,env_ids):
                if not hasattr(self,'_independent_event_index'):
                    self._independent_event_index=torch.full((self.num_envs,),-1,device=self.device,dtype=torch.long)
                    self._event_counter_cpu=[-1]*self.num_envs
                original=self.torch_gen
                try:
                    for env_id in env_ids.detach().cpu().tolist():
                        self._event_counter_cpu[env_id]+=1
                        index=self._event_counter_cpu[env_id]
                        self._independent_event_index[env_id]=index
                        self.torch_gen=torch.Generator(device=self.device).manual_seed(event_seed(self._env.cfg.seed,env_id,index))
                        # Invoke the original sampler without shared-RNG wrapper.
                        OnlineComplianceReferenceCommand._sample_events(self,torch.tensor([env_id],device=self.device,dtype=torch.long))
                finally:self.torch_gen=original
        self.commands.upper_body_joints.class_type=IndependentEvents
        print('PAIRED_EVENT_RNG','sha256(global seed, environment id, local event index)',flush=True)

def capture_paired(e):
    out=capture_base(e);c=e.command_manager.get_term('upper_body_joints')
    for key,value in dict(event_index=c._independent_event_index,sampled_peak_force=c._sampled_peak_force_world,
                          event_stiffness=c._event_stiffness).items():
        out[key]=value.detach().cpu().numpy().copy()
    return out
