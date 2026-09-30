#new change
"""Independent held 35% pose: env259, pre-step150, median-failure cohort."""
import json
import numpy as np
import torch
from isaaclab.utils import configclass
from neutral_dual_adapt_cfg import NeutralDualAdaptCfg,ROOT,ARM,NAMES

@configclass
class FixedSagittal35Cfg(NeutralDualAdaptCfg):
    def __post_init__(self):
        super().__post_init__()
        with np.load(ROOT/'logs/stand_coordination_v1/neutral_dual_adapt_v1/x_velocity_recovery/timing35/seed0_500.npz') as d:
            assert d['alive'][150,259] and list(d['joint_names'])==NAMES
            pose=d['ref'][150,259].copy()
            phase=float(d['phase_time'][150,259])
        c=self.commands.upper_body_joints
        neutral=np.loadtxt(c.demo_recording_path[0],delimiter=',')
        order=json.loads((ROOT/'datasets/motions_csv/stand_coordination_v1/manifest.json').read_text())['csv_joint_order']
        motion=np.repeat(neutral[:1],1201,axis=0)
        for i in ARM:motion[:,7+order.index(NAMES[i])]=pose[i]
        folder=ROOT/'datasets/motions_csv/fixed_sagittal35_env259_step150'
        folder.mkdir(exist_ok=True)
        path=folder/'reference.csv'
        if path.exists():np.testing.assert_allclose(np.loadtxt(path,delimiter=','),motion,atol=1e-11,rtol=0)
        else:np.savetxt(path,motion,delimiter=',',fmt='%.12f')
        c.demo_recording_path=[str(path)]
        c.demo_random_init=False
        c.clip_length=None
        c.demo_start_range=[[0.,0.]]
        base=c.class_type
        class Fixed35Command(base):
            def _populate_reference_buffers(self,result,origins):
                super()._populate_reference_buffers(result,origins)
                expected=torch.as_tensor(pose,device=self.device)
                torch.testing.assert_close(self._dof_pos,expected.expand_as(self._dof_pos),atol=1e-6,rtol=0)
                self._relative_timestamp.fill_(phase)
                assert self._dof_vel.abs().max()<1e-6
        Fixed35Command.__qualname__='Fixed35Command'
        globals()['Fixed35Command']=Fixed35Command
        c.class_type=Fixed35Command
        print('FIXED35_POSE',{'env':259,'step':150,'phase':phase,'q_ref_deg':dict(zip(NAMES,np.rad2deg(pose).tolist()))},flush=True)
