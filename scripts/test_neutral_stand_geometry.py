#new change
"""Read-only frozen rollout and geometry checks; prints metrics, no report."""
import argparse
import sys
sys.path.insert(0,'/home/noah/mingkai/general_softmimic/softmimic_gym')
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--jit',default=None)
p.add_argument('--num_envs',type=int,default=512)
p.add_argument('--steps',type=int,default=1500)
p.add_argument('--seed',type=int,default=0)
p.add_argument('--reset_velocity',type=float,default=0.)
p.add_argument('--profile',default=None)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args();app=AppLauncher(args).app
import gymnasium as gym
import torch
import numpy as np
import json
import softmimic_gym.tasks
from neutral_stand_geometry_cfg import TASK,NeutralStandGeometryCfg,state,LEG_IDS,UPPER_IDS,NAMES

def main():
    cfg=NeutralStandGeometryCfg();cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device;cfg.seed=args.seed
    if args.reset_velocity:
        for key in ('x','y','roll','pitch'):
            cfg.events.reset_robot.params['velocity_range'][key]=(-args.reset_velocity,args.reset_velocity)
    env=gym.make(TASK,cfg=cfg);e=env.unwrapped
    try:
        obs,_=env.reset(seed=args.seed)
        r,c,ids,fids,feet,fe,com,forces=state(e)
        assert e.action_manager.total_action_dim==12
        assert torch.isfinite(obs['policy']).all()
        torch.testing.assert_close(r.data.joint_pos[:,ids],c.dof_pos[:,1],atol=1e-5,rtol=0)
        torch.testing.assert_close(r.data.root_pos_w[:,2],c.root_state['root_pos'][:,1,2],atol=1e-5,rtol=0)
        # Original sphere calibration: same neutral pose, original authored shape.
        # Check live collider minima rather than link-origin clearance.
        from pxr import Usd,UsdGeom,Gf
        from isaaclab.utils.math import quat_apply
        cache=UsdGeom.XformCache();mins=[]
        for side in ['left','right']:
            foot=e.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/'+side+'_ankle_roll_link')
            bid=r.body_names.index(side+'_ankle_roll_link')
            for prim in Usd.PrimRange(foot,Usd.TraverseInstanceProxies()):
                if prim.IsA(UsdGeom.Sphere) and '/spheres/' in str(prim.GetPath()):
                    transform=cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(foot).GetInverse()
                    radius=UsdGeom.Sphere(prim).GetRadiusAttr().Get()
                    points=[Gf.Vec3d(radius*np.sin(t)*np.cos(a),radius*np.sin(t)*np.sin(a),radius*np.cos(t)) for t in np.linspace(0,np.pi,25) for a in np.linspace(0,2*np.pi,49)]
                    local=torch.tensor([list(transform.Transform(v)) for v in points],device=e.device,dtype=torch.float32)
                    world=quat_apply(r.data.body_quat_w[0,bid].expand(len(points),-1),local)+r.data.body_pos_w[0,bid]
                    mins.append(float(world[:,2].min()))
        assert len(mins)==8,mins
        assert abs(min(mins)-.005)<1e-4,mins
        print('GEOMETRY_PASS',{'sphere_min_m':min(mins),'root_z':float(r.data.root_pos_w[0,2]),'obs_shape':list(obs['policy'].shape),'action_dim':12},flush=True)
        policy=torch.jit.load(args.jit,map_location=e.device).eval() if args.jit else None
        alive=torch.ones(args.num_envs,device=e.device,dtype=torch.bool)
        fail=torch.full((args.num_envs,),-1,device=e.device,dtype=torch.long)
        drift_max=torch.zeros(args.num_envs,device=e.device);foot_max=drift_max.clone();tilt_max=drift_max.clone()
        speeds=[];upper_errors=[]
        steady_speed_sum=torch.zeros(args.num_envs,device=e.device)
        steady_count=torch.zeros(args.num_envs,device=e.device)
        knee_loaded_count=torch.zeros(args.num_envs,device=e.device)
        contact_sensor=e.scene['contact_forces']
        sensor_names=contact_sensor.body_names
        knee_ids=[sensor_names.index(n) for n in ('left_knee_link','right_knee_link')]
        import cProfile
        profiler=cProfile.Profile() if args.profile else None
        if profiler:profiler.enable()
        with torch.no_grad():
            for step in range(args.steps):
                action=policy(obs['policy']) if policy else torch.zeros((args.num_envs,12),device=e.device)
                assert torch.isfinite(action).all()
                obs,reward,terminated,truncated,_=env.step(action)
                assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
                bad=terminated & alive
                fail[bad]=step+1;alive &= ~terminated
                r,c,ids,fids,feet,fe,com,forces=state(e)
                # Avoid reading auto-reset state as a terminal physical sample.
                valid=alive & ~truncated
                dr=(r.data.root_pos_w[:,:2]-c.root_state['root_pos'][:,1,:2]).norm(dim=1)
                drift_max=torch.maximum(drift_max,torch.where(valid,dr,0.))
                foot_max=torch.maximum(foot_max,torch.where(valid,fe.norm(dim=-1).amax(1),0.))
                tilt=torch.acos((-r.data.projected_gravity_b[:,2]).clamp(-1,1))
                tilt_max=torch.maximum(tilt_max,torch.where(valid,tilt,0.))
                if valid.any():speeds.append(float(r.data.root_lin_vel_w[valid,:2].norm(dim=1).mean()))
                if step>=50:
                    steady_speed_sum+=torch.where(valid,r.data.root_lin_vel_w[:,:2].norm(dim=1),0.)
                    steady_count+=valid.float()
                    knee_loaded_count+=((contact_sensor.data.net_forces_w[:,knee_ids,2].abs().amax(1)>5)&valid).float()
                upper_ids=[r.joint_names.index(NAMES[i]) for i in UPPER_IDS]
                if not (terminated|truncated).any():torch.testing.assert_close(r.data.joint_pos_target[:,upper_ids],c.dof_pos[:,1,UPPER_IDS],atol=1e-5,rtol=0)
                #new change
                # No useful first-episode evidence remains after every env fails.
                if not alive.any():
                    break
        if profiler:
            profiler.disable();profiler.dump_stats(args.profile)
        steady_mean=steady_speed_sum/steady_count.clamp(min=1)
        qualified=alive&(drift_max<.05)&(foot_max<.02)&(tilt_max<.0872665)&(steady_mean<.03)&(knee_loaded_count==0)
        print('SUPPORT_ACCEPTANCE',json.dumps({'qualified_fraction':float(qualified.float().mean()),'criteria':'survive; root max<5cm; foot max<2cm; tilt max<5deg; post1s mean speed<3cm/s; no knee load>5N after1s','root_drift_p95_m':float(torch.quantile(drift_max,.95)),'foot_error_p95_m':float(torch.quantile(foot_max,.95)),'post1s_speed_p95_mps':float(torch.quantile(steady_mean,.95)),'knee_load_env_fraction':float((knee_loaded_count>0).float().mean()),'reset_velocity':args.reset_velocity}),flush=True)
        print('EVAL_RESULT',json.dumps({'seed':args.seed,'requested_steps':args.steps,'executed_steps':step+1,'num_envs':args.num_envs,'failures':int((fail>=0).sum()),'survival':float(alive.float().mean()),'first_failure_min':int(fail[fail>=0].min()) if (fail>=0).any() else None,'first_failure_median':float(fail[fail>=0].float().median()) if (fail>=0).any() else None,'root_drift_max_m':float(drift_max.max()),'foot_error_max_m':float(foot_max.max()),'tilt_max_deg':float(tilt_max.max()*180/torch.pi),'mean_surviving_speed_mps':float(np.mean(speeds)) if speeds else None}),flush=True)
    finally:env.close()
try:main()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:app.close()
