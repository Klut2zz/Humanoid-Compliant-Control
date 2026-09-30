#new change
"""Deterministic checkpoint acceptance; no optimizer steps or task changes."""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'softmimic_gym'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--output',required=True)
p.add_argument('--num_envs',type=int,default=512)
p.add_argument('--steps',type=int,default=500)
p.add_argument('--seed',type=int,default=0)
p.add_argument('--force_max',type=int,choices=[0,4],default=4)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
assert not Path(args.output).exists(),args.output
os.environ['DUAL_ISOLATION']='combined'
os.environ['DUAL_FORCE_MAX']=str(args.force_max)
os.environ['DIAGNOSTIC_ARM_SCALE']='1.0'
app=AppLauncher(args).app
import hashlib
import json
import numpy as np
import torch
import gymnasium as gym
import softmimic_gym.tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from dual540_force_train_cfg import TASK,Dual540ForceTrainCfg
from dual540_force_model import runner_config
from neutral_stand_geometry_cfg import state,NAMES
from softmimic_gym.assets.robots.g1_compat import G1_SOFTMIMIC_KEYPOINT_BODY_NAMES

def digest(module):
    h=hashlib.sha256()
    for name,value in module.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    ckpt=Path(args.checkpoint)
    provenance=json.loads((ckpt.parent/'base_controllers.json').read_text())
    for name,expected in provenance['sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    cfg=Dual540ForceTrainCfg();cfg.scene.num_envs=args.num_envs
    cfg.seed=args.seed;cfg.sim.device=args.device
    env=gym.make(TASK,cfg=cfg);e=env.unwrapped
    try:
        env.reset(seed=args.seed)
        wrapped=RslRlVecEnvWrapper(env)
        runner=OnPolicyRunner(wrapped,runner_config(args.seed),log_dir=None,device=e.device)
        actor=runner.alg.actor
        data=torch.load(ckpt,map_location=e.device,weights_only=False)
        actor.load_state_dict(data['actor_state_dict'],strict=True)
        actor.eval().requires_grad_(False)
        term=e.action_manager.get_term('joint_pos')
        modules=(actor,term.lower,term.base.upper)
        initial_hashes=[digest(m) for m in modules]
        assert all(not p.requires_grad for m in modules for p in m.parameters())
        r,c,ids,fids,feet,fe,com,forces=state(e)
        names=['left_wrist_yaw_link','right_wrist_yaw_link']
        bids=[r.body_names.index(n) for n in names]
        kids=[list(G1_SOFTMIMIC_KEYPOINT_BODY_NAMES).index(n) for n in names]
        initial_feet=feet.clone()
        alive=torch.ones(e.num_envs,device=e.device,dtype=torch.bool)
        trace=[]
        with torch.inference_mode():
            for step in range(args.steps):
                r,c,ids,fids,feet,fe,com,forces=state(e)
                action=actor(wrapped.get_observations(),stochastic_output=False)
                assert torch.isfinite(action).all()
                fields=dict(alive=alive,q=r.data.joint_pos[:,ids],ref=c.dof_pos[:,1],
                    nominal_ref=c._dof_pos[:,1],root=r.data.root_pos_w-e.scene.env_origins,
                    root_error=r.data.root_pos_w[:,:2]-c.root_state['root_pos'][:,1,:2],
                    root_vel=r.data.root_lin_vel_w,foot_error=fe,foot_drift=feet-initial_feet,
                    com=com-e.scene.env_origins,foot_load=forces[:,:,2],
                    tilt=torch.acos((-r.data.projected_gravity_b[:,2]).clamp(-1,1)),
                    wrist=r.data.body_pos_w[:,bids]-e.scene.env_origins[:,None],
                    wrist_ref=c.keypoints[:,1,kids,:3]-e.scene.env_origins[:,None],
                    residual=.02*torch.tanh(action),force=c.force_world,
                    event_index=c._independent_event_index,event_time=c._event_time,
                    ramp=c._ramp_time,hold=c._hold_time,body=c.selected_body_ids,
                    sampled_peak=c._sampled_peak_force_world,stiffness=c._event_stiffness,
                    accepted_peak=c._accepted_peak_force_world,ik_scale=c.ik_feasible_scale,
                    ik_delta=c._peak_joint_delta)
                frame={k:v.detach().cpu().numpy().copy() for k,v in fields.items()}
                obs,reward,done,truncated,_=env.step(action)
                assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
                for k,v in dict(done=done,truncated=truncated,
                    target=r.data.joint_pos_target[:,ids],
                    containment=e.termination_manager.get_term('containment')).items():
                    frame[k]=v.detach().cpu().numpy().copy()
                trace.append(frame)
                alive &= ~done
                if (step+1)%100==0:print('PROGRESS',step+1,'first_episode_failures',int((~alive).sum()),flush=True)
        assert initial_hashes==[digest(m) for m in modules]
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output,**{k:np.stack([f[k] for f in trace]) for k in trace[0]},
            dt=np.asarray(e.step_dt),seed=np.asarray(args.seed),checkpoint=np.asarray(str(ckpt)),
            joint_names=np.asarray(NAMES),force_max=np.asarray(args.force_max))
        print('CHECKPOINT_EVAL_PASS',json.dumps({'checkpoint':str(ckpt),'force_max':args.force_max,
              'seed':args.seed,'steps':args.steps,'environments':args.num_envs,
              'failed_environments':int((~alive).sum()),'all_controller_states_unchanged':True}),flush=True)
    finally:env.close()

try:main()
except BaseException:
    import traceback
    traceback.print_exc();sys.stderr.flush();raise
finally:app.close()
