#new change
"""Independent 0.5s C1 arm-reference transition, frozen dynamic-wrist probe."""
import torch
from isaaclab.utils import configclass
from wrist_pitch_dynamic_probe_cfg import WristPitchDynamicProbeCfg
from neutral_dual_adapt_cfg import ARM

@configclass
class ReferenceTransitionSmoothCfg(WristPitchDynamicProbeCfg):
    def __post_init__(self):
        super().__post_init__()
        base=self.commands.upper_body_joints.class_type

        class SmoothReference(base):
            def _populate_reference_buffers(self,result,origins):
                from softmimic_deploy.src.motion_lib.keypoint_extractor import KeypointExtractor
                q=result['dof_pos'];v=result['dof_vel'];dt=float(self._env.step_dt);duration=.5
                if not hasattr(self,'_transition_start'):
                    self._transition_start=self.motion_start.clone()
                    self._transition_age=torch.full((self.num_envs,),duration+1,device=q.device)
                    self._transition_d0=torch.zeros_like(q[:,0])
                    self._transition_dv=torch.zeros_like(q[:,0])
                    self._transition_fk=KeypointExtractor('g1')
                else:
                    self._transition_age+=dt
                    reset=self._env.episode_length_buf==0
                    self._transition_age[reset]=duration+1
                    changed=(self.motion_start!=self._transition_start)&~reset
                    if changed.any():
                        self._transition_age[changed]=0
                        self._transition_d0[changed]=0
                        self._transition_dv[changed]=0
                        # Continue the previous commanded pose/velocity at the new time.
                        delta=self._transition_prev_q[:,0]+dt*self._transition_prev_v[:,0]-q[:,0]
                        dv=self._transition_prev_v[:,0]-v[:,0]
                        self._transition_d0[changed[:,None].expand_as(delta)]=delta[changed[:,None].expand_as(delta)]
                        self._transition_dv[changed[:,None].expand_as(dv)]=dv[changed[:,None].expand_as(dv)]
                        other=[i for i in range(q.shape[-1]) if i not in ARM]
                        self._transition_d0[:,other]=0;self._transition_dv[:,other]=0
                        print('SMOOTH_TRANSITION',int(changed.sum()),'duration',duration,flush=True)
                self._transition_start=self.motion_start.clone()
                t=self._transition_age[:,None]+torch.arange(q.shape[1],device=q.device)[None]*self.future_dt
                x=(t/duration).clamp(0,1)
                h00=2*x**3-3*x**2+1;h10=x**3-2*x**2+x
                dh00=(6*x**2-6*x)/duration;dh10=3*x**2-4*x+1
                correction=h00[:,:,None]*self._transition_d0[:,None]+duration*h10[:,:,None]*self._transition_dv[:,None]
                velocity=dh00[:,:,None]*self._transition_d0[:,None]+dh10[:,:,None]*self._transition_dv[:,None]
                active=t<duration
                if active.any():
                    result=dict(result)
                    result['dof_pos']=q+correction
                    result['dof_vel']=v+velocity
                    pos,rot,_=self._transition_fk.compute_keypoints_batch(result['root_pos'][active],result['root_rot'][active],result['dof_pos'][active])
                    result['keypoints']=result['keypoints'].clone()
                    result['keypoint_rotations']=result['keypoint_rotations'].clone()
                    assert pos.shape==result['keypoints'][active].shape
                    result['keypoints'][active]=pos;result['keypoint_rotations'][active]=rot
                self._transition_prev_q=result['dof_pos'].clone()
                self._transition_prev_v=result['dof_vel'].clone()
                super()._populate_reference_buffers(result,origins)

        self.commands.upper_body_joints.class_type=SmoothReference
