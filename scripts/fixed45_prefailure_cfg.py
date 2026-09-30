#new change
"""One held pre-failure pose; preserves dynamic probe gain/center and phase."""
import json
import numpy as np
import torch
from isaaclab.utils import configclass
from dual_amplitude_50_100_cfg import AmplitudeDiagnosticCfg
from neutral_dual_adapt_cfg import ROOT,ARM,NAMES

@configclass
class Fixed45PrefailureCfg(AmplitudeDiagnosticCfg):
    def __post_init__(self):
        super().__post_init__()
        assert self.arm_motion_scale==.45
        trace=ROOT/'logs/stand_coordination_v1/neutral_dual_adapt_v1/multipose_lower_balance/validation/model406_scale045_seed0_500.npz'
        with np.load(trace) as d:
            assert d['joint_names'].tolist()==NAMES
            assert not d['terminated'][:251,18].any()
            pose=d['ref'][250,18].copy()
            phase=float(d['phase_time'][250,18])
        order=json.loads((ROOT/'datasets/motions_csv/stand_coordination_v1/manifest.json').read_text())['csv_joint_order']
        neutral=np.loadtxt(ROOT/'datasets/motions_csv/neutral_stand_geometry_v1/neutral.csv',delimiter=',')
        motion=np.repeat(neutral[:1],1601,axis=0)
        for i in ARM:motion[:,7+order.index(NAMES[i])]=pose[i]
        np.testing.assert_allclose(motion[:,7:36][:,[order.index(n) for n in NAMES]],np.broadcast_to(pose,(len(motion),29)),atol=1e-6,rtol=0)
        folder=ROOT/'datasets/motions_csv/fixed45_env18_step250'
        folder.mkdir(exist_ok=True)
        path=folder/'reference.csv'
        if path.exists():np.testing.assert_allclose(np.loadtxt(path,delimiter=','),motion,atol=1e-11,rtol=0)
        else:np.savetxt(path,motion,delimiter=',',fmt='%.12f')
        c=self.commands.upper_body_joints
        c.demo_recording_path=[str(path)]
        c.demo_start_range=[[0.,0.]]
        c.demo_random_init=False
        c.clip_length=None
        base=c.class_type
        class FixedReference(base):
            def _populate_reference_buffers(self,result,origins):
                super()._populate_reference_buffers(result,origins)
                torch.testing.assert_close(self._dof_pos,torch.as_tensor(pose,device=self.device).expand_as(self._dof_pos),atol=1e-6,rtol=0)
                assert self._dof_vel.abs().max()<1e-6
                self._relative_timestamp.fill_(phase)
        c.class_type=FixedReference
        print('FIXED45_PASS',{'env':18,'step':250,'phase':phase,'gain':self.pitch_probe_gain,'center':self.pitch_probe_center,'velocity':0},flush=True)
