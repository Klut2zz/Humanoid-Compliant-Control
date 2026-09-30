#new change
"""Correct CSV-order arms-only scaling; legacy references remain untouched."""
import json
import numpy as np
import torch
from isaaclab.utils import configclass
from neutral_dual_adapt_cfg import NeutralDualAdaptCfg, ROOT, ARM, NAMES

@configclass
class SagittalCsvV2Cfg(NeutralDualAdaptCfg):
    arm_motion_scale: float = .15

    def __post_init__(self):
        super().__post_init__()
        # assert self.arm_motion_scale in (.15,.25)
        #new change
        # assert self.arm_motion_scale in (.15,.25,.35)
        #new change: independent 50%/100% frozen diagnostics; old scales unchanged.
        # assert self.arm_motion_scale in (.15,.25,.35,.50,1.0)
        #new change: user-requested independent 45% frozen diagnostic.
        # assert self.arm_motion_scale in (.15,.25,.35,.45,.50,1.0)
        #new change: permit the independent 75% frozen reference diagnostic.
        # assert self.arm_motion_scale in (.15,.25,.35,.45,.50,.75,1.0)
        #new change: permit the requested frozen 85% diagnostic.
        assert self.arm_motion_scale in (.15,.25,.35,.45,.50,.75,.85,1.0)
        source=ROOT/'datasets/motions_csv/stand_coordination_v1'
        manifest=json.loads((source/'manifest.json').read_text())
        order=manifest['csv_joint_order']
        assert len(order)==29 and set(order)==set(NAMES)
        mapping=[order.index(n) for n in NAMES]
        assert mapping==manifest['reference_reindex']
        neutral=np.loadtxt(self.commands.upper_body_joints.demo_recording_path[0],delimiter=',')
        original=np.loadtxt(source/'01_bilateral_sagittal.csv',delimiter=',')
        motion=np.repeat(neutral[:1],len(original),axis=0)
        cols=[7+order.index(NAMES[i]) for i in ARM]
        motion[:,cols]+=self.arm_motion_scale*(original[:,cols]-neutral[0,cols])
        other=[j for j in range(motion.shape[1]) if j not in cols]
        np.testing.assert_array_equal(motion[:,other],np.repeat(neutral[:1,other],len(motion),axis=0))
        canonical=motion[:,7:36][:,mapping]
        canonical_neutral=neutral[0,7:36][mapping]
        expected=np.repeat(canonical_neutral[None],len(motion),axis=0)
        expected[:,ARM]+=self.arm_motion_scale*(original[:,7:36][:,mapping][:,ARM]-canonical_neutral[ARM])
        np.testing.assert_allclose(canonical,expected,atol=1e-12,rtol=0)
        folder=ROOT/f'datasets/motions_csv/dual_sagittal_csv_v2_{int(self.arm_motion_scale*100):02d}'
        folder.mkdir(exist_ok=True)
        path=folder/'reference.csv'
        if path.exists():np.testing.assert_allclose(np.loadtxt(path,delimiter=','),motion,atol=1e-11,rtol=0)
        else:np.savetxt(path,motion,delimiter=',',fmt='%.12f')
        c=self.commands.upper_body_joints
        c.demo_recording_path=[str(path)]
        # Keep legacy random-crop protocol for a mapping-only comparison.
        base=c.class_type
        lower=[i for i in range(29) if i not in ARM]
        class AuditedCommand(base):
            def _populate_reference_buffers(self,result,origins):
                assert self.motion_lib.reindex_mapping.tolist()==mapping
                super()._populate_reference_buffers(result,origins)
                target=torch.as_tensor(canonical_neutral[lower],device=self.device,dtype=self._dof_pos.dtype)
                torch.testing.assert_close(self._dof_pos[:,:,lower],target.expand_as(self._dof_pos[:,:,lower]),atol=1e-6,rtol=0)
        c.class_type=AuditedCommand
        print('CSV_MAPPING_PASS',json.dumps({'scale':self.arm_motion_scale,'non_arm_unchanged':True,'joint_peak_to_peak_deg':dict(zip(NAMES,np.rad2deg(np.ptp(canonical,axis=0)).tolist()))}),flush=True)

#new change
@configclass
class SagittalCsvV2_25Cfg(SagittalCsvV2Cfg):
    arm_motion_scale: float = .25

#new change
@configclass
class SagittalCsvV2_35Cfg(SagittalCsvV2Cfg):
    arm_motion_scale: float = .35
