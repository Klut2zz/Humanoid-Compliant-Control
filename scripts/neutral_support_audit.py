# new change
"""Neutral support A/B/C audit; inference only, no training or state playback."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path('/home/noah/mingkai/general_softmimic')
sys.path.insert(0, str(ROOT / 'softmimic_gym'))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--steps', type=int, default=500)
p.add_argument('--num_envs', type=int, default=8)
p.add_argument('--output', required=True)
p.add_argument('--video', action='store_true')
p.add_argument('--branch', choices=['A_reference','B_model1375','C_supervised12mrad'], required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

def main():
    import gymnasium as gym
    import numpy as np
    import torch
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    from isaaclab_tasks.utils import parse_env_cfg
    import softmimic_gym.tasks
    from softmimic_gym.assets.robots.g1_compat import G1_SOFTMIMIC_29_JOINT_NAMES

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    task = 'Isaac-G1-StandCoordination-StationaryCriticAwareV7-100Pct-29DOF-v0'
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 0
    cfg.recorders = None
    from isaaclab.sensors import ContactSensorCfg
    audit_bodies = ('left_ankle_roll_link','right_ankle_roll_link','left_knee_link','right_knee_link')
    for body in audit_bodies:
        setattr(cfg.scene, 'audit_'+body, ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+body, update_period=0., history_length=0))
    cmd = cfg.commands.upper_body_joints
    cmd.demo_recording_path = [str(ROOT / 'datasets/motions_csv/stand_coordination_v1/00_neutral.csv')]
    cmd.demo_start_range = [[0., 0.]]
    cmd.debug_vis = False
    for name in ('demo_random_init', 'demo_random_reset'):
        if hasattr(cmd, name):
            setattr(cmd, name, False)
    if hasattr(cmd, 'force_range_n'):
        cmd.force_range_n = (0., 0.)
    if hasattr(cfg.events, 'apply_online_compliance_force'):
        cfg.events.apply_online_compliance_force = None
    reset = cfg.events.reset_robot.params
    for key in ('pose_range', 'velocity_range'):
        reset[key] = {name: (0., 0.) for name in reset[key]}
    reset['jpos_range'] = (0., 0.)
    reset['jvel_range'] = (0., 0.)
    # Disable stochastic observation noise, without changing the observation layout.
    for group in vars(cfg.observations).values():
        if hasattr(group, 'enable_corruption'):
            group.enable_corruption = False
    cfg.viewer.origin_type = 'env'
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (3., 3., 2.)
    cfg.viewer.lookat = (0., 0., .65)
    env = gym.make(task, cfg=cfg, render_mode='rgb_array' if args.video else None)
    try:
        e = env.unwrapped
        robot = e.scene['robot']
        command = e.command_manager.get_term('upper_body_joints')
        sensor = e.scene['contact_forces']
        ids = torch.as_tensor(command.control_joint_ids, device=e.device)
        feet = [robot.body_names.index(n) for n in ('left_ankle_roll_link', 'right_ankle_roll_link')]
        pelvis = robot.body_names.index('pelvis')
        run = ROOT / 'logs/rsl_rl/g1_stand_coordination_v1/2026-08-20_18-57-37_stand_coordination_v1_full_v1'
        baseline = torch.jit.load(str(run / 'model_1375.jit'), map_location=e.device).eval()
        candidate = torch.jit.load(str(ROOT / 'logs/stand_coordination_v1/stationary_base_state_aware_v6/model_1375_v6_supervised12mrad.jit'), map_location=e.device).eval()
        stage = e.sim.stage
        collision = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0/Robot'), Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                owner = prim
                while owner and not owner.IsPseudoRoot() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                    owner = owner.GetParent()
                collision.append({'path': str(prim.GetPath()), 'rigid_body': str(owner.GetPath()) if owner else None,
                                  'enabled': UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()})
        meta = {'task': task, 'steps': args.steps, 'dt': e.step_dt,
                'joint_names': list(G1_SOFTMIMIC_29_JOINT_NAMES), 'body_names': robot.body_names,
                'sensor_body_names': sensor.body_names, 'colliders': collision,
                'material_properties_env0': robot.root_physx_view.get_material_properties()[0].cpu().tolist(),
                'events': str(cfg.events), 'actuators': str(cfg.scene.robot.actuators),
                'comparison': 'one branch per fresh process; compare initial_state.npy across branches before interpreting A/B/C; not a restored solver snapshot',
                'foot_velocity': 'ankle link origin velocity, not contact-point tangential velocity'}
        (out / 'metadata.json').write_text(json.dumps(meta, indent=2, default=str))
        initial = None
        summaries = {}
        for label, policy in [('A_reference', None), ('B_model1375', baseline), ('C_supervised12mrad', candidate)]:
            if label != args.branch:
                continue
            obs, _ = env.reset(seed=0)
            state = torch.cat([robot.data.root_state_w, robot.data.joint_pos, robot.data.joint_vel, command.dof_pos[:, 1]], dim=1).clone()
            np.save(out / 'initial_state.npy', state.cpu().numpy())
            if initial is None:
                initial = state
            delta = (state - initial).abs().max().item()
            if delta > 1.e-6:
                raise RuntimeError(f'Initial-state mismatch {label}: {delta}')
            origin = robot.data.root_pos_w.clone()
            foot0 = robot.data.body_pos_w[:, feet].clone()
            # World-fixed reference crosses expose translation in the video.
            for i, pos in enumerate([origin[0], *foot0[0]]):
                path = f'/World/neutral_audit_anchor_{i}'
                curve = UsdGeom.BasisCurves.Define(stage, path)
                x, y = pos[:2].tolist()
                curve.CreatePointsAttr([Gf.Vec3f(x-.10,y,.012), Gf.Vec3f(x+.10,y,.012), Gf.Vec3f(x,y-.10,.012), Gf.Vec3f(x,y+.10,.012)])
                curve.CreateCurveVertexCountsAttr([2, 2])
                curve.CreateTypeAttr('linear')
                curve.CreateWidthsAttr([.008])
                curve.CreateDisplayColorAttr([Gf.Vec3f(1., .3, .1)])
            failed = torch.zeros(args.num_envs, dtype=torch.bool, device=e.device)
            first_failure = [-1] * args.num_envs
            failure_reasons = {}
            drift, speed, tilt = [], [], []
            force_max = torch.zeros(len(sensor.body_names), device=e.device)
            writer_video = None
            if args.video:
                import imageio.v2 as imageio
                writer_video = imageio.get_writer(str(out / f'{label}.mp4'), fps=round(1/e.step_dt))
            try:
                with (out / f'{label}.csv').open('w', newline='') as f, torch.no_grad():
                    w = csv.writer(f)
                    fields = ['step','env','episode_valid','root_dx','root_dy','root_vx','root_vy','tilt_rad','com_x','com_y','left_foot_dx','left_foot_dy','right_foot_dx','right_foot_dy','left_foot_vx','left_foot_vy','right_foot_vx','right_foot_vy']
                    fields += [f'{kind}:{n}' for kind in ('q','ref','target','dq','torque') for n in G1_SOFTMIMIC_29_JOINT_NAMES]
                    fields += [f'force_{axis}:{n}' for n in sensor.body_names for axis in 'xyz']
                    fields += ['single_sensor_fz:'+n for n in audit_bodies]
                    w.writerow(fields)
                    for step in range(args.steps):
                        if policy is None:
                            action = torch.zeros((args.num_envs,29), device=e.device)
                        else:
                            inp = obs['policy']
                            if label.startswith('C'):
                                inp = torch.cat([inp] + [obs[g] for g in ('left_arm_adapter','right_arm_adapter','waist_adapter','support_adapter')], dim=-1)
                            action = policy(inp)
                        # Log pre-step states and the target generated for that state.
                        term = e.action_manager.get_term('joint_pos')
                        target = command.dof_pos[:,1] + action * term._scale
                        q = robot.data.joint_pos[:,ids]
                        dq = robot.data.joint_vel[:,ids]
                        torques = robot.data.applied_torque[:,ids]
                        mass = robot.data.default_mass.to(e.device)
                        com = (robot.data.body_com_pos_w * mass[:,:,None]).sum(1) / mass.sum(1)[:,None]
                        com = com - robot.data.body_pos_w[:,feet].mean(1)
                        rp = robot.data.root_pos_w-origin
                        rv = robot.data.root_lin_vel_w
                        ang = torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1))
                        fp = (robot.data.body_pos_w[:,feet]-foot0)[:,:,:2].flatten(1)
                        fv = robot.data.body_lin_vel_w[:,feet,:2].flatten(1)
                        force = sensor.data.net_forces_w
                        force_max = torch.maximum(force_max, force.norm(dim=-1).amax(0))
                        singles = torch.stack([e.scene['audit_'+body].data.net_forces_w[:,0,2] for body in audit_bodies],dim=1)
                        values = torch.cat([rp[:,:2],rv[:,:2],ang[:,None],com[:,:2],fp,fv,q,command.dof_pos[:,1],target,dq,torques,force.flatten(1),singles],dim=1).cpu().tolist()
                        for k, row in enumerate(values):
                            w.writerow([step,k,int(not failed[k].item()),*row])
                        valid = ~failed
                        if valid.any():
                            drift.extend(rp[valid,:2].norm(dim=-1).cpu().tolist())
                            speed.extend(rv[valid,:2].norm(dim=-1).cpu().tolist())
                            tilt.extend(ang[valid].cpu().tolist())
                        obs, _, terminated, truncated, _ = env.step(action)
                        for k in torch.where((terminated | truncated) & ~failed)[0].tolist():
                            first_failure[k] = step+1
                            failure_reasons[str(k)] = [name for name in e.termination_manager.active_terms if e.termination_manager.get_term(name)[k].item()]
                        failed |= terminated | truncated
                        if writer_video is not None:
                            writer_video.append_data(env.render())
                        if (step+1)%100 == 0:
                            print(label, step+1, 'first_failures', int(failed.sum()), flush=True)
            finally:
                if writer_video is not None:
                    writer_video.close()
            summaries[label] = {'initial_state_max_delta':delta, 'first_failure_step':first_failure, 'failure_reasons':failure_reasons,
                                'pre_failure_root_drift_max_m':max(drift), 'pre_failure_root_speed_mean_mps':float(np.mean(speed)),
                                'pre_failure_tilt_max_deg':float(np.rad2deg(max(tilt))),
                                'max_force_by_sensor_body_n':dict(zip(sensor.body_names,force_max.cpu().tolist()))}
            (out/'summary.json').write_text(json.dumps(summaries,indent=2))
        print(json.dumps(summaries,indent=2),flush=True)
    finally:
        env.close()

try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    app.close()
