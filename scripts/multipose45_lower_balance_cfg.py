#new change
"""Independent five-pose lower-only adaptation from model406; no gain/reward edits."""
import json
import numpy as np
import torch
import gymnasium as gym
from isaaclab.utils import configclass
from wrist_pitch_dynamic_probe_cfg import WristPitchDynamicProbeCfg
from neutral_dual_adapt_cfg import ROOT,ARM,NAMES
from fixed_transition_velocity_recovery_cfg import reset_x_recovery,FixedTransitionVelocityRecoveryRunnerCfg

@configclass
class MultiPose45LowerBalanceCfg(WristPitchDynamicProbeCfg):
    arm_motion_scale: float = .45

    def __post_init__(self):
        super().__post_init__()
        folders=['neutral_stand_geometry_v1','dual_fixed_transition476_csv_order',
                 'fixed_sagittal35_env259_step150','fixed45_env18_step250',
                 'fixed45_paused_median_env1_step280']
        paths=[ROOT/'datasets/motions_csv'/f/('neutral.csv' if i==0 else 'reference.csv') for i,f in enumerate(folders)]
        order=json.loads((ROOT/'datasets/motions_csv/stand_coordination_v1/manifest.json').read_text())['csv_joint_order']
        poses=[]
        for path in paths:
            data=np.loadtxt(path,delimiter=',')
            np.testing.assert_allclose(data,np.repeat(data[:1],len(data),axis=0),atol=1e-11,rtol=0)
            poses.append(data[0,7:36][[order.index(n) for n in NAMES]])
        poses=np.stack(poses)
        other=[i for i in range(29) if i not in ARM]
        np.testing.assert_allclose(poses[:,other],np.repeat(poses[:1,other],5,axis=0),atol=1e-6,rtol=0)
        log=ROOT/'logs/stand_coordination_v1/neutral_dual_adapt_v1'
        sources=[(log/'sagittal_pause/pause.npz',130,476),
                 (log/'x_velocity_recovery/timing35/seed0_500.npz',150,259),
                 (log/'multipose_lower_balance/validation/model406_scale045_seed0_500.npz',250,18),
                 (log/'multipose_lower_balance/validation/model406_scale045_branch250_pause_500.npz',280,1)]
        phases=[0.]
        for group,(path,t,e) in enumerate(sources,1):
            with np.load(path) as d:
                assert d['joint_names'].tolist()==NAMES
                np.testing.assert_allclose(poses[group],d['ref'][t,e],atol=1e-6,rtol=0)
                phases.append(float(d['phase_time'][t,e]))
        c=self.commands.upper_body_joints
        c.demo_recording_path=[str(p) for p in paths]
        c.demo_start_range=[[0.,0.]]*5
        c.demo_random_init=False
        c.clip_length=None
        base=c.class_type
        class FivePoseCommand(base):
            def _populate_reference_buffers(self,result,origins):
                super()._populate_reference_buffers(result,origins)
                groups=self.motion_ids%5
                expected=torch.as_tensor(poses,device=self.device,dtype=self._dof_pos.dtype)[groups]
                torch.testing.assert_close(self._dof_pos,expected[:,None].expand_as(self._dof_pos),atol=1e-6,rtol=0)
                assert self._dof_vel.abs().max()<1e-6
                self._relative_timestamp=torch.as_tensor(phases,device=self.device,dtype=self._relative_timestamp.dtype)[groups]
                if not hasattr(self,'_five_pose_printed'):
                    print('FIVE_POSE_AUDIT',{'counts':torch.bincount(groups,minlength=5).tolist(),'phases':phases,'arm_output_dim':14,'lower_output_dim':12},flush=True)
                    self._five_pose_printed=True
        FivePoseCommand.__qualname__='FivePoseCommand'
        globals()['FivePoseCommand']=FivePoseCommand
        c.class_type=FivePoseCommand
        self.events.reset_robot.func=reset_x_recovery

@configclass
class MultiPose45RunnerCfg(FixedTransitionVelocityRecoveryRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations=20
        self.save_interval=5

TASK='Isaac-G1-MultiPose45-FrozenArms-LowerBalance-v0'
gym.register(id=TASK,entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
             kwargs={'env_cfg_entry_point':MultiPose45LowerBalanceCfg,'rsl_rl_cfg_entry_point':MultiPose45RunnerCfg})
