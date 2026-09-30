#new change
"""One explicit held arm pose, no random crop; unchanged frozen controllers."""
import json
import numpy as np
from isaaclab.utils import configclass
from neutral_dual_adapt_cfg import NeutralDualAdaptCfg, ROOT, ARM, NAMES

@configclass
class FixedTransitionCfg(NeutralDualAdaptCfg):
    def __post_init__(self):
        super().__post_init__()
        with np.load(ROOT/'logs/stand_coordination_v1/neutral_dual_adapt_v1/sagittal_pause/pause.npz') as data:
            pose=data['ref'][130,476].copy()
            phase=float(data['phase_time'][130,476])
            assert list(data['joint_names'])==NAMES
        neutral=np.loadtxt(self.commands.upper_body_joints.demo_recording_path[0],delimiter=',')
        # order used joint_order previously, which is NOT the CSV order.
        #new change
        order=json.loads((ROOT/'datasets/motions_csv/stand_coordination_v1/manifest.json').read_text())['csv_joint_order']
        motion=np.repeat(neutral[:1],1201,axis=0)
        for i in ARM: motion[:,7+order.index(NAMES[i])]=pose[i]
        #new change: retain the rejected artifact, use a corrected independent path.
        folder=ROOT/'datasets/motions_csv/dual_fixed_transition476_csv_order'
        folder.mkdir(exist_ok=True)
        path=folder/'reference.csv'
        if path.exists():np.testing.assert_allclose(np.loadtxt(path,delimiter=','),motion,atol=1e-11,rtol=0)
        else:np.savetxt(path,motion,delimiter=',',fmt='%.12f')
        c=self.commands.upper_body_joints
        c.demo_recording_path=[str(path)]
        c.demo_random_init=False
        c.clip_length=None
        c.demo_start_range=[[0.,0.]]
        base=c.class_type
        class FixedPhaseCommand(base):
            def _populate_reference_buffers(self,result,origins):
                super()._populate_reference_buffers(result,origins)
                import torch
                torch.testing.assert_close(self._dof_pos,torch.as_tensor(pose,device=self.device).expand_as(self._dof_pos),atol=1e-6,rtol=0)
                self._relative_timestamp.fill_(phase)
                assert self._dof_vel.abs().max()<1e-6
        #new change: Hydra resolves module/class strings during training startup.
        FixedPhaseCommand.__qualname__='FixedPhaseCommand'
        globals()['FixedPhaseCommand']=FixedPhaseCommand
        c.class_type=FixedPhaseCommand
        print('FIXED_POSE source_env=476 source_step=130 phase=',phase,'shoulder_pitch_deg=',float(np.rad2deg(pose[11])),flush=True)
