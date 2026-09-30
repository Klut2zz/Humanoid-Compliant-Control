#new change
"""Evaluation only: frozen model99, 4--7 N, exact original observation/control.

Optional single-environment video uses visual-only reference/force markers.
Records the physical wrench buffer every physics substep; never trains.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'softmimic_gym'))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--num_envs', type=int, default=512)
p.add_argument('--steps', type=int, default=500)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', required=True)
p.add_argument('--video', default=None, help='MP4 path; requires num_envs=1')
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
assert args.steps > 0 and not Path(args.output).exists()
if args.video:
    assert args.num_envs == 1 and not Path(args.video).exists()
    args.enable_cameras = True
os.environ['DUAL_ISOLATION'] = 'combined'
os.environ['DUAL_FORCE_MAX'] = '4'  # Parent constructor, overridden ONLY by independent cfg.
os.environ['DIAGNOSTIC_ARM_SCALE'] = '1.0'
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
import softmimic_gym.tasks
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from stand_force_joint_medium_cfg import TASK, StandForceJointMediumCfg
from stand_force_joint_v1_cfg import TASK as ORIGINAL_TASK, NAMES, state
from stand_force_joint_v1_model import runner_config
from softmimic_gym.assets.robots.g1_compat import G1_SOFTMIMIC_KEYPOINT_BODY_NAMES


def digest(module):
    h = hashlib.sha256()
    for key, value in module.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    checkpoint = Path(args.checkpoint).resolve()
    meta = json.loads((checkpoint.parent / 'experiment_contract.json').read_text())
    assert meta['task'] == ORIGINAL_TASK
    protected = {**meta['sha256'], str(checkpoint): hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    for path, sha in protected.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha, path
    cfg = StandForceJointMediumCfg(zero_force_fraction=0.)
    cfg.scene.num_envs = args.num_envs; cfg.seed = args.seed; cfg.sim.device = args.device
    if args.video:
        cfg.viewer.origin_type = 'world'
        cfg.viewer.resolution = (1280, 720)
    env = gym.make(TASK, cfg=cfg, render_mode='rgb_array' if args.video else None)
    e = env.unwrapped; writer = None
    try:
        env.reset(seed=args.seed)
        wrapped = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(wrapped, runner_config(args.seed), log_dir=None, device=e.device)
        runner.logger.writer = None
        actor, critic = runner.alg.actor, runner.alg.critic
        saved = torch.load(checkpoint, map_location=e.device, weights_only=False)
        actor.load_state_dict(saved['actor_state_dict'], strict=True)
        critic.load_state_dict(saved['critic_state_dict'], strict=True)
        term = e.action_manager.get_term('joint_pos')
        networks = (actor, critic, term.teacher_lower, term.base.upper)
        #new change: ScriptModules require freezing each parameter explicitly.
        for network in networks:
            network.eval()
            for parameter in network.parameters(): parameter.requires_grad_(False)
        assert all(not p.requires_grad for network in networks for p in network.parameters())
        hashes = [digest(network) for network in networks]
        def forbidden_update(*unused, **kwargs):
            raise RuntimeError('This entry forbids optimizer updates')
        runner.alg.optimizer.step = forbidden_update
        assert e.obs_buf['lower_policy'].shape[-1] == 109 and e.obs_buf['policy'].shape[-1] == 197
        r, c, ids, fids, feet, fe, com, forces = state(e)
        bids = [r.body_names.index(s + '_wrist_yaw_link') for s in ('left', 'right')]
        kids = [list(G1_SOFTMIMIC_KEYPOINT_BODY_NAMES).index(s + '_wrist_yaw_link') for s in ('left', 'right')]
        # Read-only observation after the unchanged action has populated the wrench.
        original_apply = term.apply_actions
        delivered = torch.zeros_like(c.force_world)
        physical_error = torch.zeros(e.num_envs, device=e.device)
        physical_calls = 0
        def monitored_apply():
            nonlocal physical_calls
            original_apply()
            local = r.permanent_wrench_composer.composed_force_as_torch[c._env_ids, c.selected_body_ids]
            world = quat_apply(r.data.body_quat_w[c._env_ids, c.selected_body_ids], local)
            delivered.copy_(world)
            physical_error.copy_(torch.maximum(physical_error, (world-c.force_world).abs().max(-1).values))
            physical_calls += 1
        term.apply_actions = monitored_apply
        if args.video:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw, ImageFont
            Path(args.video).parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(args.video, fps=round(1/e.step_dt), codec='libx264', macro_block_size=1)
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
            c.set_debug_vis(True)
            anchor = r.data.root_pos_w[0].detach().cpu().numpy()
            e.sim.set_camera_view(eye=(float(anchor[0]+2.3), float(anchor[1]+2.3), 1.7),
                                  target=(float(anchor[0]), float(anchor[1]), .78))
        alive = torch.ones(e.num_envs, device=e.device, dtype=torch.bool)
        trace = []; initial_feet = feet.clone(); nominal0 = c._dof_pos[:, 1].clone()
        print('ALL_NETWORKS_FROZEN; MEAN_ACTION; NO_OPTIMIZER_UPDATE', flush=True)
        with torch.inference_mode():
            for step in range(args.steps):
                obs = wrapped.get_observations(); action = actor(obs)
                r, c, ids, fids, feet, fe, com, forces = state(e)
                torch.testing.assert_close(c._dof_pos[:, 1], nominal0, atol=1e-6, rtol=0)
                assert c._dof_vel.abs().max() < 1e-6
                fields = dict(alive=alive, q=r.data.joint_pos[:, ids], ref=c.dof_pos[:, 1], nominal_ref=c._dof_pos[:, 1],
                    root=r.data.root_pos_w-e.scene.env_origins, root_error=r.data.root_pos_w[:, :2]-c.root_state['root_pos'][:, 1, :2],
                    root_vel=r.data.root_link_lin_vel_w, root_com_vel=r.data.root_com_lin_vel_w,
                    foot_error=fe, foot_drift=feet-initial_feet, foot_load=forces[:, :, 2], com=com-e.scene.env_origins,
                    tilt=torch.acos((-r.data.projected_gravity_b[:, 2]).clamp(-1, 1)),
                    wrist=r.data.body_pos_w[:, bids]-e.scene.env_origins[:, None],
                    wrist_ref=c.keypoints[:, 1, kids, :3]-e.scene.env_origins[:, None],
                    wrist_nominal=c._keypoints[:, 1, kids, :3]-e.scene.env_origins[:, None], action=action,
                    force=c.force_world, event_index=c._independent_event_index, event_time=c._event_time,
                    ramp=c._ramp_time, hold=c._hold_time, sampled_peak=c._sampled_peak_force_world,
                    accepted_peak=c._accepted_peak_force_world, body=c.selected_body_ids,
                    target_displacement=c.target_displacement_world, ik_scale=c.ik_feasible_scale,
                    ik_pending=c._solution_pending)
                frame = {k: v.detach().cpu().numpy().copy() for k, v in fields.items()}
                obs, reward, done, truncated, _ = env.step(action)
                assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
                for k, v in dict(done=done, truncated=truncated, target=r.data.joint_pos_target[:, ids],
                                correction=e._wb_residual, release_age=e._wb_release_age,
                                physical_force=delivered).items():
                    frame[k] = v.detach().cpu().numpy().copy()
                trace.append(frame); alive &= ~done
                if writer:
                    picture = Image.fromarray(env.render())
                    draw = ImageDraw.Draw(picture)
                    f = float(c.force_world[0].norm()); t = float(c._event_time[0]); ramp = float(c._ramp_time[0])
                    phase = ('RECOVERY / WAIT' if t < 0 else 'RISE' if t < ramp else
                             'HOLD' if t < ramp+float(c._hold_time[0]) else 'RELEASE')
                    side = 'LEFT' if int(c.selected_body_ids[0]) == bids[0] else 'RIGHT'
                    root_mm = float((r.data.root_pos_w[0,:2]-c.root_state['root_pos'][0,1,:2]).norm()*1000)
                    speed = float(r.data.root_link_lin_vel_w[0,:2].norm()*1000)
                    lines = [f'model99 frozen | 4-7 N | t={(step+1)*e.step_dt:.2f}s | {phase}',
                             f'{side} wrist: F={f:.2f} N | IK scale={float(c.ik_feasible_scale[0]):.2f}',
                             f'Root offset={root_mm:.1f} mm | speed={speed:.1f} mm/s',
                             'Red: nominal | Yellow: online ref | Blue: actual | Orange: force']
                    draw.rectangle((0, 0, 1000, 108), fill=(20, 20, 20))
                    for i, line in enumerate(lines): draw.text((10, 5+i*25), line, font=font, fill='white')
                    writer.append_data(np.asarray(picture))
                if (step+1) % 100 == 0:
                    print('PROGRESS', step+1, 'first_episode_failures', int((~alive).sum()), flush=True)
        assert physical_calls == args.steps*cfg.decimation
        assert float(physical_error.max()) < 1e-4, physical_error.max()
        assert hashes == [digest(network) for network in networks]
        for path, sha in protected.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha, path
        out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **{k: np.stack([f[k] for f in trace]) for k in trace[0]},
            dt=np.asarray(e.step_dt), seed=np.asarray(args.seed), cohort=np.asarray('force'),
            checkpoint=np.asarray(str(checkpoint)), joint_names=np.asarray(NAMES), wrist_body_ids=np.asarray(bids),
            force_range=np.asarray([4., 7.]), physical_wrench_max_error_N=physical_error.cpu().numpy())
        result = dict(task=TASK, checkpoint=str(checkpoint), force_range_N=[4,7], seed=args.seed,
            environments=args.num_envs, steps=args.steps, failed=int((~alive).sum()), optimizer_updates=0,
            physical_substeps_verified=physical_calls, physical_wrench_max_error_N=float(physical_error.max()),
            only_control_configuration_change='force_range_n: [2,4] -> [4,7]', all_network_weights_unchanged=True,
            protected_sha256=protected, video=args.video)
        out.with_suffix('.json').write_text(json.dumps(result, indent=2))
        print('FROZEN_MEDIUM_EVAL_PASS', json.dumps(result), flush=True)
    finally:
        if writer is not None: writer.close(); print('VIDEO_SAVED', args.video, flush=True)
        env.close()


try: main()
#new change
except BaseException:
    # Kit shutdown can exit before Python prints an unhandled traceback.
    import traceback
    traceback.print_exc(); sys.stderr.flush()
    raise
finally: app.close()
