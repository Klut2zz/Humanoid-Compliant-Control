#new change
"""Separate entry: baseline/zero/gradient checks do not update any weights."""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'softmimic_gym'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--mode', choices=['baseline','zero','gradient','train'], default='zero')
p.add_argument('--task', choices=['Isaac-G1-Dual540-ForceAware-ArmsResidual-v0'],
               default='Isaac-G1-Dual540-ForceAware-ArmsResidual-v0')
p.add_argument('--num_envs', type=int, default=16)
p.add_argument('--steps', type=int, default=250)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--force_max', type=int, choices=[0,4], default=4)
p.add_argument('--max_iterations', type=int, default=10)
p.add_argument('--run_name', default='dual540_force_arms_v1')
p.add_argument('--output', default=None)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
os.environ['DUAL_ISOLATION']='combined'
os.environ['DUAL_FORCE_MAX']=str(args.force_max)
os.environ['DIAGNOSTIC_ARM_SCALE']='1.0'
app=AppLauncher(args).app
import copy
import hashlib
import json
import numpy as np
import torch
from datetime import datetime
import gymnasium as gym
import softmimic_gym.tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from dual540_force_train_cfg import TASK,Dual540ForceTrainCfg,LOWER
from dual540_paired_events_cfg import Dual540PairedEventsCfg
from dual540_force_model import runner_config
from neutral_stand_geometry_cfg import state

def main():
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    cfg=Dual540PairedEventsCfg() if args.mode=='baseline' else Dual540ForceTrainCfg()
    cfg.scene.num_envs=args.num_envs; cfg.seed=args.seed; cfg.sim.device=args.device
    env=gym.make(TASK,cfg=cfg); e=env.unwrapped
    try:
        obs,_=env.reset(seed=args.seed)
        if args.mode=='baseline':
            actor=torch.jit.load(str(LOWER),map_location=e.device).eval()
            # Match the wrapper's reset in the residual branch exactly.
            wrapped=RslRlVecEnvWrapper(env)
            obs=e.obs_buf
            runner=None
        else:
            wrapped=RslRlVecEnvWrapper(env)
            logdir=None
            if args.mode=='train':
                logdir=str(Path('logs/rsl_rl/dual540_force_arms')/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_'+args.run_name))
                Path(logdir).mkdir(parents=True,exist_ok=False)
            runner=OnPolicyRunner(wrapped,runner_config(args.seed),log_dir=logdir,device=e.device)
            # RSL initializes the writer in learn(); pre-training checkpoint saves
            # must also work in audit modes which deliberately never call learn().
            runner.logger.writer=None
            actor=runner.alg.actor
            term=e.action_manager.get_term('joint_pos')
            frozen=list(term.lower.parameters())+list(term.base.upper.parameters())
            optimized={id(p) for g in runner.alg.optimizer.param_groups for p in g['params']}
            assert all(not x.requires_grad and id(x) not in optimized for x in frozen)
            assert e.obs_buf['lower_policy'].shape[1]==109
            with torch.no_grad(): assert torch.count_nonzero(actor(wrapped.get_observations()))==0
            print('FREEZE_ZERO_PASS',{'lower_obs':109,'residual_dim':14,'policy_obs':e.obs_buf['policy'].shape[1],
                  'limit_rad':.02,'frozen_parameters':sum(x.numel() for x in frozen)},flush=True)
        if args.mode=='train':
            # New residual/critic only. Never resume a 12D/29D legacy PPO optimizer.
            from neutral_dual_adapt_cfg import UPPER
            provenance={'lower':str(LOWER),'upper':str(UPPER),'force_max':args.force_max,
                        'residual_limit_rad':.02,'seed':args.seed,
                        'sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (LOWER,UPPER)}}
            Path(logdir,'base_controllers.json').write_text(json.dumps(provenance,indent=2))
            runner.save(str(Path(logdir,'model_zero.pt')))
            runner.learn(args.max_iterations,init_at_random_ep_len=False)
            return
        if args.mode=='gradient':
            # Real PPO rollout/returns/backprop; intercept optimizer.step, never update.
            before={n:p.detach().clone() for n,p in actor.named_parameters()}
            critic_before={n:p.detach().clone() for n,p in runner.alg.critic.named_parameters()}
            frozen_before=[p.detach().clone() for p in frozen]
            checked=[]
            def check_step(*a,**kw):
                grads={n:float(x.grad.norm()) for n,x in actor.named_parameters() if x.grad is not None}
                assert all(torch.isfinite(x.grad).all() for g in runner.alg.optimizer.param_groups for x in g['params'] if x.grad is not None)
                assert grads['mlp.4.weight']>0 and grads['mlp.4.bias']>0
                assert any(x.grad is not None and x.grad.abs().max()>0 for x in runner.alg.critic.parameters())
                checked.append(grads)
            runner.alg.optimizer.step=check_step
            obs=wrapped.get_observations()
            active_samples=0
            # Warm up to include an active external-force event in the training batch.
            with torch.inference_mode():
                for _ in range(100): obs,_,_,_=wrapped.step(torch.zeros((args.num_envs,14),device=e.device))
                for _ in range(runner.cfg['num_steps_per_env']):
                    action=runner.alg.act(obs)
                    obs,reward,done,extra=wrapped.step(action)
                    assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
                    active_samples+=int((e.command_manager.get_term('upper_body_joints').force_world.norm(dim=-1)>0).sum())
                    # 56 joint features precede the two three-axis wrist forces.
                    torch.testing.assert_close(obs['policy'][:,56:62].norm(dim=-1)*8.,
                        e.command_manager.get_term('upper_body_joints').force_world.norm(dim=-1),
                        atol=2e-5,rtol=1e-5)
                    assert term.processed_actions.abs().max()<=.02
                    runner.alg.process_env_step(obs,reward,done,extra)
                runner.alg.compute_returns(obs)
            losses=runner.alg.update()
            assert checked
            assert all(torch.equal(before[n],x) for n,x in actor.named_parameters())
            assert all(torch.equal(critic_before[n],x) for n,x in runner.alg.critic.named_parameters())
            assert all(torch.equal(a,b) and b.grad is None for a,b in zip(frozen_before,frozen))
            if args.force_max: assert active_samples>0
            # Exercise checkpoint serialization without retaining a trained model.
            import tempfile
            with tempfile.TemporaryDirectory(prefix='dual540_checkpoint_') as directory:
                path=str(Path(directory,'zero.pt'))
                runner.save(path)
                runner.load(path)
                assert all(torch.equal(before[n],x) for n,x in actor.named_parameters())
            print('GRADIENT_PASS',json.dumps({'minibatches':len(checked),'actor_gradients':checked,
                    'losses':losses,'active_force_samples':active_samples,
                    'checkpoint_roundtrip':True,'optimizer_updates':0}),flush=True)
            return
        trace=[]
        with torch.no_grad():
            for step in range(args.steps):
                action=actor(obs['policy']) if args.mode=='baseline' else actor(wrapped.get_observations())
                obs,reward,done,truncated,_=env.step(action)
                r,c,ids,fids,feet,fe,com,forces=state(e)
                assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
                trace.append({k:v.detach().cpu().numpy().copy() for k,v in dict(
                    q=r.data.joint_pos[:,ids],target=r.data.joint_pos_target[:,ids],
                    root=r.data.root_pos_w,ref=c.dof_pos[:,1],force=c.force_world,
                    done=done,foot_error=fe,com=com).items()})
        assert args.output and not Path(args.output).exists()
        np.savez_compressed(args.output,**{k:np.stack([f[k] for f in trace]) for k in trace[0]})
        print('ROLLOUT_PASS',args.mode,args.output,'failed_envs',int(np.stack([f['done'] for f in trace]).any(0).sum()),flush=True)
    finally: env.close()

try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    app.close()
