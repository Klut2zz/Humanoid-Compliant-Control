#new change
"""Frozen V2 model30 pressure test, derived from the original medium evaluator.

No training mode. Records pre-cap request, capped peak, IK acceptance, actual
physics-substep wrench and control traces. Original evaluator stays unchanged.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'logs/rsl_rl/stand_force_recovery_mix_v2/2026-09-29_08-55-42_stand_force_recovery_mix_v2/model_30.pt'
sys.path.insert(0, str(ROOT/'softmimic_gym'))
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument('--num_envs', type=int, default=512)
p.add_argument('--steps', type=int, default=500)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--checkpoint', default=str(DEFAULT))
p.add_argument('--force_range', type=float, nargs=2, default=[7.,10.], help='4 7 is telemetry equivalence only')
p.add_argument('--output', required=True)
p.add_argument('--video', help='Single-environment MP4 path; visualization only')
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
assert tuple(args.force_range) in ((4.,7.), (7.,10.))
assert args.steps > 0 and not Path(args.output).exists()
assert not Path(args.output).with_suffix('.json').exists()
assert Path(args.checkpoint).resolve() == DEFAULT
if args.video:
    assert args.num_envs == 1 and not Path(args.video).exists()
    args.enable_cameras = True
os.environ['DUAL_ISOLATION'] = 'combined'
os.environ['DUAL_FORCE_MAX'] = '4'
os.environ['DIAGNOSTIC_ARM_SCALE'] = '1.0'
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
import softmimic_gym.tasks
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from stand_force_pressure_7to10_cfg import TASK, StandForcePressureCfg
from stand_force_recovery_mix_v2_cfg import StandForceRecoveryMixV2Cfg
from stand_force_joint_v1_cfg import NAMES, state
from stand_force_joint_v1_model import runner_config
from softmimic_gym.assets.robots.g1_compat import G1_SOFTMIMIC_KEYPOINT_BODY_NAMES


def digest(module):
    h = hashlib.sha256()
    for key, value in module.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def same_tree(a, b):
    if isinstance(a, torch.Tensor): torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: same_tree(a[k],b[k])
    elif isinstance(a, (tuple,list)):
        assert len(a) == len(b)
        for x,y in zip(a,b): same_tree(x,y)
    else: assert a == b


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    checkpoint = Path(args.checkpoint).resolve()
    meta = json.loads((checkpoint.parent/'experiment_contract.json').read_text())
    assert meta['task'] == 'Isaac-G1-StandForceJoint-RecoveryMixV2-v0'
    protected = {**meta['sha256'], str(checkpoint): hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    for path, sha in protected.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha, path
    cfg = StandForcePressureCfg(pressure_force_range_n=tuple(args.force_range))
    original_cfg = StandForceRecoveryMixV2Cfg(recovery_sampling='medium')
    old, new = original_cfg.to_dict(), cfg.to_dict()
    new.pop('pressure_force_range_n')
    new['commands']['upper_body_joints']['class_type'] = old['commands']['upper_body_joints']['class_type']
    assert json.dumps(old, default=repr, sort_keys=True) == json.dumps(new, default=repr, sort_keys=True)
    cfg.scene.num_envs = args.num_envs; cfg.seed = args.seed; cfg.sim.device = args.device
    if args.video:
        cfg.viewer.origin_type = 'world'; cfg.viewer.resolution = (1280,720)
    env = gym.make(TASK, cfg=cfg, render_mode='rgb_array' if args.video else None)
    e = env.unwrapped; writer = None
    try:
        env.reset(seed=args.seed)
        wrapped = RslRlVecEnvWrapper(env)
        config = runner_config(args.seed)
        expected = copy.deepcopy(meta['runner']); expected['seed'] = args.seed
        assert config == expected
        runner = OnPolicyRunner(wrapped, config, log_dir=None, device=e.device)
        runner.logger.writer = None
        runner.load(str(checkpoint),strict=True,map_location=e.device)
        actor, critic = runner.alg.actor, runner.alg.critic
        saved = torch.load(checkpoint,map_location=e.device,weights_only=False)
        same_tree(actor.state_dict(),saved['actor_state_dict'])
        same_tree(critic.state_dict(),saved['critic_state_dict'])
        same_tree(runner.alg.optimizer.state_dict(),saved['optimizer_state_dict'])
        optimizer_before = copy.deepcopy(runner.alg.optimizer.state_dict())
        term = e.action_manager.get_term('joint_pos')
        networks = (actor,critic,term.teacher_lower,term.base.upper)
        for network in networks:
            network.eval()
            for parameter in network.parameters(): parameter.requires_grad_(False)
        hashes = [digest(n) for n in networks]
        def forbidden_update(*unused,**kwargs): raise RuntimeError('Pressure test forbids optimizer updates')
        runner.alg.optimizer.step = forbidden_update
        assert e.obs_buf['lower_policy'].shape[-1] == 109 and e.obs_buf['policy'].shape[-1] == 197
        r,c,ids,fids,feet,fe,com,loads = state(e)
        bids = [r.body_names.index(s+'_wrist_yaw_link') for s in ('left','right')]
        kids = [list(G1_SOFTMIMIC_KEYPOINT_BODY_NAMES).index(s+'_wrist_yaw_link') for s in ('left','right')]
        original_apply = term.apply_actions
        delivered = torch.zeros_like(c.force_world)
        physical_error = torch.zeros(e.num_envs,device=e.device); physical_calls = 0
        def monitored_apply():
            nonlocal physical_calls
            original_apply()
            local = r.permanent_wrench_composer.composed_force_as_torch[c._env_ids,c.selected_body_ids]
            world = quat_apply(r.data.body_quat_w[c._env_ids,c.selected_body_ids],local)
            delivered.copy_(world)
            physical_error.copy_(torch.maximum(physical_error,(world-c.force_world).abs().max(-1).values))
            physical_calls += 1
        term.apply_actions = monitored_apply
        if args.video:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw, ImageFont
            Path(args.video).parent.mkdir(parents=True,exist_ok=True)
            writer = imageio.get_writer(args.video,fps=round(1/e.step_dt),codec='libx264',macro_block_size=1)
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
            c.set_debug_vis(True)
            anchor = r.data.root_pos_w[0].detach().cpu().numpy()
            e.sim.set_camera_view(eye=(float(anchor[0]+2.3),float(anchor[1]+2.3),1.7),
                                  target=(float(anchor[0]),float(anchor[1]),.78))
        alive = torch.ones(e.num_envs,device=e.device,dtype=torch.bool)
        trace = []; initial_feet = feet.clone(); nominal0 = c._dof_pos[:,1].clone()
        print('ALL_NETWORKS_FROZEN; MEAN_ACTION; NO_OPTIMIZER_UPDATE',flush=True)
        with torch.inference_mode():
            for step in range(args.steps):
                action = actor(wrapped.get_observations())
                r,c,ids,fids,feet,fe,com,loads = state(e)
                torch.testing.assert_close(c._dof_pos[:,1],nominal0,atol=1e-6,rtol=0)
                assert c._dof_vel.abs().max() < 1e-6
                assert torch.all(c._recovery_cohort == 2)
                fields = dict(alive=alive,q=r.data.joint_pos[:,ids],ref=c.dof_pos[:,1],nominal_ref=c._dof_pos[:,1],
                    root=r.data.root_pos_w-e.scene.env_origins,root_error=r.data.root_pos_w[:,:2]-c.root_state['root_pos'][:,1,:2],
                    root_vel=r.data.root_link_lin_vel_w,root_com_vel=r.data.root_com_lin_vel_w,
                    foot_error=fe,foot_drift=feet-initial_feet,foot_load=loads[:,:,2],com=com-e.scene.env_origins,
                    tilt=torch.acos((-r.data.projected_gravity_b[:,2]).clamp(-1,1)),
                    wrist=r.data.body_pos_w[:,bids]-e.scene.env_origins[:,None],
                    wrist_ref=c.keypoints[:,1,kids,:3]-e.scene.env_origins[:,None],
                    wrist_nominal=c._keypoints[:,1,kids,:3]-e.scene.env_origins[:,None],action=action,
                    force=c.force_world,event_index=c._independent_event_index,event_time=c._event_time,ramp=c._ramp_time,hold=c._hold_time,
                    sampled_peak=c._sampled_peak_force_world,accepted_peak=c._accepted_peak_force_world,body=c.selected_body_ids,
                    target_displacement=c.target_displacement_world,ik_scale=c.ik_feasible_scale,ik_pending=c._solution_pending,
                    curriculum_cohort=c._recovery_cohort,rear_biased=c._recovery_rear_biased,
                    requested_peak=c.pressure_requested_peak_world,requested_peak_n=c.pressure_requested_peak_n,
                    displacement_cap_scale=c.pressure_displacement_cap_scale,event_stiffness=c._event_stiffness,
                    ik_active_residual_m=c.ik_active_residual_m,ik_foot_motion_m=c.ik_max_foot_motion_m,
                    ik_com_residual_m=c.ik_com_residual_m,ik_direction_cosine=c.ik_direction_cosine,
                    ik_peak_joint_speed=c.ik_peak_joint_speed_rad_s)
                frame = {k:v.detach().cpu().numpy().copy() for k,v in fields.items()}
                obs,reward,done,truncated,_ = env.step(action)
                assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
                for k,v in dict(done=done,truncated=truncated,target=r.data.joint_pos_target[:,ids],
                               correction=e._wb_residual,release_age=e._wb_release_age,physical_force=delivered).items():
                    frame[k] = v.detach().cpu().numpy().copy()
                trace.append(frame); alive &= ~done
                if writer:
                    picture = Image.fromarray(env.render()); draw = ImageDraw.Draw(picture)
                    # HUD force/phase refer to the just-applied control interval,
                    # including release/reset boundaries; picture is post-step.
                    t = float(frame['event_time'][0]); ramp = float(frame['ramp'][0]); hold = float(frame['hold'][0])
                    phase = 'RECOVERY / WAIT' if t<0 else 'RISE' if t<ramp else 'HOLD' if t<ramp+hold else 'RELEASE'
                    side = 'LEFT' if int(frame['body'][0])==bids[0] else 'RIGHT'
                    lines = [f'V2 model30 FROZEN | {args.force_range[0]:g}-{args.force_range[1]:g} N | t={(step+1)*e.step_dt:.2f}s | {phase}',
                        f'{side}: requested peak={float(frame["requested_peak_n"][0]):.2f} N | accepted peak={float(np.linalg.norm(frame["accepted_peak"][0])):.2f} N',
                        f'Applied (last substep)={float(delivered[0].norm()):.2f} N | IK scale={float(frame["ik_scale"][0]):.3f} | resets={int(np.sum([f["done"][0] for f in trace]))}',
                        f'Root offset={float((r.data.root_pos_w[0,:2]-c.root_state["root_pos"][0,1,:2]).norm()*1000):.1f} mm | speed={float(r.data.root_link_lin_vel_w[0,:2].norm()*1000):.1f} mm/s',
                        'Red: nominal | Yellow: online ref | Blue: actual | Orange: force (arrow size capped)']
                    draw.rectangle((0,0,1279,132),fill=(20,20,20))
                    for i,line in enumerate(lines): draw.text((10,5+i*25),line,font=font,fill='white')
                    writer.append_data(np.asarray(picture))
                if (step+1)%100==0: print('PROGRESS',step+1,'first_episode_failures',int((~alive).sum()),flush=True)
        assert physical_calls == args.steps*cfg.decimation
        assert float(physical_error.max()) < 1e-4
        assert hashes == [digest(n) for n in networks]
        same_tree(runner.alg.optimizer.state_dict(),optimizer_before)
        for path,sha in protected.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
        out = Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out,**{k:np.stack([f[k] for f in trace]) for k in trace[0]},
            dt=np.asarray(e.step_dt),seed=np.asarray(args.seed),cohort=np.asarray('force'),sampling=np.asarray('pressure'),
            checkpoint=np.asarray(str(checkpoint)),joint_names=np.asarray(NAMES),wrist_body_ids=np.asarray(bids),
            requested_force_range=np.asarray(args.force_range),max_displacement_m=np.asarray(c.cfg.max_displacement_m))
        result = dict(task=TASK,checkpoint=str(checkpoint),requested_force_range_N=args.force_range,seed=args.seed,
            environments=args.num_envs,steps=args.steps,failed=int((~alive).sum()),optimizer_updates=0,
            physical_substeps_verified=physical_calls,physical_wrench_max_error_N=float(physical_error.max()),
            only_control_change=('force peak quantile 4-7 to 7-10; same direction, timing, IK/physics/gains/controller'
                if tuple(args.force_range)==(7.,10.) else 'none: telemetry-only 4-7 N equivalence'),
            all_network_weights_unchanged=True,optimizer_state_unchanged=True,protected_sha256=protected,video=args.video)
        out.with_suffix('.json').write_text(json.dumps(result,indent=2))
        print('FROZEN_PRESSURE_EVAL_PASS',json.dumps(result),flush=True)
    finally:
        if writer is not None: writer.close(); print('VIDEO_SAVED',args.video,flush=True)
        env.close()


try: main()
except BaseException:
    import traceback
    traceback.print_exc(); sys.stderr.flush(); raise
finally: app.close()
