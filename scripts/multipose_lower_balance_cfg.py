#new change
"""Three fixed references, balanced environment groups, frozen legacy arms."""
import gymnasium as gym
import numpy as np
import json
import torch
from isaaclab.utils import configclass
from neutral_dual_adapt_cfg import NeutralDualAdaptCfg,ROOT,ARM,NAMES
from fixed_transition_velocity_recovery_cfg import reset_x_recovery,FixedTransitionVelocityRecoveryRunnerCfg

@configclass
class MultiPoseLowerBalanceCfg(NeutralDualAdaptCfg):
    def __post_init__(self):
        super().__post_init__()
        c=self.commands.upper_body_joints
        files=[ROOT/'datasets/motions_csv/neutral_stand_geometry_v1/neutral.csv',
               ROOT/'datasets/motions_csv/dual_fixed_transition476_csv_order/reference.csv',
               ROOT/'datasets/motions_csv/fixed_sagittal35_env259_step150/reference.csv']
        order=json.loads((ROOT/'datasets/motions_csv/stand_coordination_v1/manifest.json').read_text())['csv_joint_order']
        mapping=[order.index(n) for n in NAMES]
        poses=[]
        for path in files:
            data=np.loadtxt(path,delimiter=',')
            np.testing.assert_allclose(data,np.repeat(data[:1],len(data),axis=0),rtol=0,atol=1e-11)
            poses.append(data[0,7:36][mapping])
        poses=np.stack(poses)
        lower=[i for i in range(29) if i not in ARM]
        np.testing.assert_allclose(poses[:,lower],np.repeat(poses[:1,lower],3,axis=0),rtol=0,atol=1e-11)
        for name,indices in [('sagittal_pause/pause.npz',(130,476)),('timing35/seed0_500.npz',(150,259))]:
            root=ROOT/'logs/stand_coordination_v1/neutral_dual_adapt_v1'
            path=root/name if name.startswith('sagittal_pause') else root/'x_velocity_recovery'/name
            with np.load(path) as d:
                group=1 if name.startswith('sagittal_pause') else 2
                np.testing.assert_allclose(poses[group],d['ref'][indices],rtol=0,atol=1e-6)
        c.demo_recording_path=[str(p) for p in files]
        c.demo_start_range=[[0.,0.]]*3
        c.demo_random_init=False
        c.clip_length=None
        base=c.class_type
        class MultiPoseCommand(base):
            def _populate_reference_buffers(self,result,origins):
                super()._populate_reference_buffers(result,origins)
                groups=self.motion_ids%3
                expected=torch.as_tensor(poses,device=self.device,dtype=self._dof_pos.dtype)[groups]
                torch.testing.assert_close(self._dof_pos,expected[:,None].expand_as(self._dof_pos),rtol=0,atol=1e-6)
                assert self._dof_vel.abs().max()<1e-6
                self._relative_timestamp[groups==1]=2.25
                self._relative_timestamp[groups==2]=2.98
                if not hasattr(self,'_multipose_audited'):
                    self._multipose_audited=True
                    print('MULTIPOSE_REFERENCE_PASS',{'counts':torch.bincount(groups,minlength=3).tolist(),'order':['neutral','fixed476','fixed35_259'],'neutral_phase':'unchanged','fixed_phases':[2.25,2.98]},flush=True)
        MultiPoseCommand.__qualname__='MultiPoseCommand'
        globals()['MultiPoseCommand']=MultiPoseCommand
        c.class_type=MultiPoseCommand
        # Retain accepted zero/x-velocity replay curriculum without reward/gain edits.
        self.events.reset_robot.func=reset_x_recovery

TASK='Isaac-G1-MultiPose-FrozenArms-LowerBalance-v0'
gym.register(id=TASK,entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
    kwargs={'env_cfg_entry_point':MultiPoseLowerBalanceCfg,'rsl_rl_cfg_entry_point':FixedTransitionVelocityRecoveryRunnerCfg})
