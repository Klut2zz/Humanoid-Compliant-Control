#new change
"""Inference-only time-warp intervention and exposed snapshot audit."""
import os
from pathlib import Path
import torch

def branch_step(e, c, obs, step):
    if step != 100:
        return
    assert not e.termination_manager.terminated.any()
    def tensors(obj):
        return {k:v.detach().cpu().clone() for k,v in vars(obj).items() if isinstance(v,torch.Tensor)}
    r=e.scene['robot']
    snapshot=dict(root=r.data.root_state_w.cpu().clone(),q=r.data.joint_pos.cpu().clone(),
        dq=r.data.joint_vel.cpu().clone(),body=r.data.body_state_w.cpu().clone(),
        obs={k:v.cpu().clone() for k,v in obs.items()},command=tensors(c),env=tensors(e),
        actions=tensors(e.action_manager),terms={k:tensors(e.action_manager.get_term(k)) for k in e.action_manager.active_terms},
        contact=e.scene['contact_forces'].data.net_forces_w.cpu().clone(),
        history={},rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),command_rng=c.torch_gen.get_state())
    for group, terms in e.observation_manager._group_obs_term_history_buffer.items():
        for name, buf in terms.items():
            snapshot['history'][group+'/'+name]=tensors(buf)
    path=Path(os.environ['TRACE_PATH']).with_suffix('.snapshot.pt')
    assert not path.exists()
    torch.save(snapshot,path)
    if os.environ['PAUSE_MODE'] != 'pause':
        return
    # First changed update occurs after this identical action/observation boundary.
    anchor=c.motion_count.clone()*c.motion_dt
    original=c.motion_lib.get_motion_state
    duration=.5
    def warp(delta):
        u=delta.clamp(0,duration)
        travel=.5*u+duration/(2*torch.pi)*torch.sin(torch.pi*u/duration)
        travel=torch.where(delta<0,delta,travel)
        rate=torch.where(delta<0,torch.ones_like(delta),.5*(1+torch.cos(torch.pi*u/duration)))
        return travel,rate
    def get_state(motion_ids,motion_times,offset=None,future_frame_dt=.02):
        frames=[]
        for j in range(c.n_future_steps+1):
            travel,rate=warp(motion_times+j*future_frame_dt-anchor)
            result=original(motion_ids,anchor+travel,offset=offset,future_frame_dt=0.)
            frame={k:v[:,:1].clone() if isinstance(v,torch.Tensor) else v for k,v in result.items()}
            for k in ('dof_vel','root_vel','root_ang_vel','root_vel_global','root_ang_vel_global'):
                if k in frame:frame[k]*=rate[:,None,None]
            frames.append(frame)
        return {k:torch.cat([f[k] for f in frames],dim=1) if isinstance(frames[0][k],torch.Tensor) else frames[0][k] for k in frames[0]}
    c.motion_lib.get_motion_state=get_state
    populate=c._populate_reference_buffers
    def populate_warped(result,origins):
        populate(result,origins)
        travel,_=warp(c.motion_count*c.motion_dt-anchor)
        c._relative_timestamp=anchor+travel-c.motion_start*c.motion_dt
    c._populate_reference_buffers=populate_warped
    print('PAUSE_INSTALLED step=100 ramp_seconds=0.5',flush=True)
