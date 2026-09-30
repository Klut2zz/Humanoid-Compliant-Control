#new change
"""Whole-body coordination entry; audits never call optimizer.step or learn."""
import argparse,os,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'softmimic_gym'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['baseline','zero','gradient','train'],default='zero')
p.add_argument('--task',default='Isaac-G1-Dual540-WholeBodyForceResidual-v0',choices=['Isaac-G1-Dual540-WholeBodyForceResidual-v0'])
p.add_argument('--num_envs',type=int,default=16)
p.add_argument('--steps',type=int,default=250)
p.add_argument('--seed',type=int,default=0)
p.add_argument('--force_max',choices=[0,4],type=int,default=4)
p.add_argument('--zero_force_fraction',choices=[0.,.25,1.],type=float,default=.25)
p.add_argument('--max_iterations',type=int,default=25)
p.add_argument('--run_name',default='wholebody_force_v1')
p.add_argument('--output',default=None)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
os.environ['DUAL_ISOLATION']='combined';os.environ['DUAL_FORCE_MAX']=str(args.force_max)
os.environ['DIAGNOSTIC_ARM_SCALE']='1.0'
app=AppLauncher(args).app
import numpy as np
import torch
import gymnasium as gym
import softmimic_gym.tasks
from datetime import datetime
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from dual540_wholebody_force_cfg import TASK,WholeBodyForceCfg,MixedFrozenCfg,runner_config,ARM,LEG_IDS,WAIST,LIMITS,LOWER
from neutral_dual_adapt_cfg import UPPER
from neutral_stand_geometry_cfg import state

def digest(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    cls=MixedFrozenCfg if args.mode=='baseline' else WholeBodyForceCfg
    cfg=cls(zero_force_fraction=args.zero_force_fraction)
    cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;cfg.sim.device=args.device
    env=gym.make(TASK,cfg=cfg);e=env.unwrapped
    try:
        env.reset(seed=args.seed);wrapped=RslRlVecEnvWrapper(env)
        if args.mode=='baseline':
            actor=torch.jit.load(str(LOWER),map_location=e.device).eval()
        else:
            logdir=None
            if args.mode=='train':
                logdir=Path('logs/rsl_rl/dual540_wholebody_force')/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_'+args.run_name)
                logdir.mkdir(parents=True,exist_ok=False)
            rcfg=runner_config(args.seed)
            runner=OnPolicyRunner(wrapped,rcfg,log_dir=str(logdir) if logdir else None,device=e.device)
            runner.logger.writer=None
            actor=runner.alg.actor;term=e.action_manager.get_term('joint_pos')
            frozen=(term.lower,term.base.upper);before=[digest(m) for m in frozen]
            optids={id(x) for g in runner.alg.optimizer.param_groups for x in g['params']}
            assert all(not x.requires_grad and id(x) not in optids for m in frozen for x in m.parameters())
            assert e.obs_buf['lower_policy'].shape[-1]==109
            with torch.no_grad():assert torch.count_nonzero(actor(wrapped.get_observations()))==0
            print('WHOLEBODY_ENTRY_PASS',json.dumps({'observations':e.obs_buf['policy'].shape[-1],
                'outputs':29,'group_limits_rad':{'arms':.02,'waist':.01,'legs':.01},
                'zero_force_fraction':args.zero_force_fraction,'base_controllers_frozen':True}),flush=True)
        if args.mode=='train':
            manifest=dict(lower=str(LOWER),upper=str(UPPER),seed=args.seed,
                force_max=args.force_max,zero_force_fraction=args.zero_force_fraction,
                limits_rad=LIMITS,runner=runner_config(args.seed),
                sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (LOWER,UPPER)})
            (logdir/'base_controllers.json').write_text(json.dumps(manifest,indent=2))
            runner.save(str(logdir/'model_zero.pt'))
            runner.learn(args.max_iterations,init_at_random_ep_len=False)
            assert before==[digest(m) for m in frozen]
            print('TRAIN_FINISHED_BASE_HASHES_UNCHANGED',flush=True)
            return
        if args.mode=='gradient':
            old_actor=digest(actor);old_critic=digest(runner.alg.critic);checks=[]
            def check_step(*a,**kw):
                assert all(torch.isfinite(x.grad).all() for g in runner.alg.optimizer.param_groups for x in g['params'] if x.grad is not None)
                g=actor.mlp[-1].weight.grad
                norms={k:float(g[ids].norm()) for k,ids in [('arms',ARM),('waist',WAIST),('legs',LEG_IDS)]}
                assert all(v>0 for v in norms.values())
                assert any(x.grad is not None and x.grad.norm()>0 for x in runner.alg.critic.parameters())
                checks.append(norms)
            runner.alg.optimizer.step=check_step
            active=0;zero_ids=torch.arange(e.num_envs,device=e.device)%4<round(args.zero_force_fraction*4)
            with torch.inference_mode():
                obs=wrapped.get_observations()
                for _ in range(100):obs,_,_,_=wrapped.step(torch.zeros((e.num_envs,29),device=e.device))
                for _ in range(runner.cfg['num_steps_per_env']):
                    obs,reward,done,extra=wrapped.step(runner.alg.act(obs))
                    c=e.command_manager.get_term('upper_body_joints')
                    assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
                    assert (term.processed_actions.abs()<=e._wb_limits).all()
                    assert torch.count_nonzero(c.force_world[zero_ids])==0
                    torch.testing.assert_close(obs['policy'][:,116:122].norm(dim=-1)*8,c.force_world.norm(dim=-1),atol=2e-5,rtol=1e-5)
                    active+=int((c.force_world.norm(dim=-1)>0).sum())
                    runner.alg.process_env_step(obs,reward,done,extra)
                runner.alg.compute_returns(obs)
            losses=runner.alg.update()
            assert checks and active>0
            assert old_actor==digest(actor) and old_critic==digest(runner.alg.critic)
            assert before==[digest(m) for m in frozen]
            assert all(x.grad is None for m in frozen for x in m.parameters())
            import tempfile
            with tempfile.TemporaryDirectory(prefix='wholebody_force_') as d:
                path=str(Path(d)/'zero.pt');runner.save(path);runner.load(path)
                assert old_actor==digest(actor) and old_critic==digest(runner.alg.critic)
            print('WHOLEBODY_GRADIENT_PASS',json.dumps(dict(group_gradient_norms=checks,
                active_force_samples=active,losses=losses,optimizer_updates=0,checkpoint_roundtrip=True)),flush=True)
            return
        assert args.output and not Path(args.output).exists()
        trace=[]
        with torch.inference_mode():
            for step in range(args.steps):
                obs=wrapped.get_observations()
                action=actor(obs['policy']) if args.mode=='baseline' else actor(obs)
                obs,reward,done,truncated,_=env.step(action)
                r,c,ids,fids,feet,fe,com,forces=state(e)
                assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
                trace.append({k:v.detach().cpu().numpy().copy() for k,v in dict(q=r.data.joint_pos[:,ids],
                    target=r.data.joint_pos_target[:,ids],root=r.data.root_pos_w,ref=c.dof_pos[:,1],
                    force=c.force_world,done=done,foot_error=fe,com=com,event_index=c._independent_event_index).items()})
        if args.mode=='zero':
            # Check +/- mapping and no substep accumulation, after recorded rollout.
            with torch.inference_mode():
                term.process_actions(torch.zeros_like(term.raw_actions));term.apply_actions()
                base=term._asset.data.joint_pos_target[:,term.joint_ids].clone()
                for sign in (-1,1):
                    term._processed.copy_(sign*e._wb_limits*.5);term.apply_actions()
                    expected=base+term._processed
                    torch.testing.assert_close(term._asset.data.joint_pos_target[:,term.joint_ids],expected,atol=0,rtol=0)
                    term.apply_actions()
                    torch.testing.assert_close(term._asset.data.joint_pos_target[:,term.joint_ids],expected,atol=0,rtol=0)
                term.reset()
                assert torch.count_nonzero(term.processed_actions)==0
                assert (e._wb_release_age==3).all()
                # Isolated reward gating unit test, after all rollout samples.
                # Restore the command buffer; never advance physics with synthetic force.
                from dual540_wholebody_force_cfg import coordination_reward
                c=e.command_manager.get_term('upper_body_joints')
                force_saved=c.force_world.clone()
                assert torch.count_nonzero(coordination_reward(e,'recovery'))==0
                e._wb_release_age.fill_(.5);c.force_world.zero_()
                value=coordination_reward(e,'recovery')
                assert torch.isfinite(value).all() and value.max()>0
                c.force_world[:,0]=1.
                assert torch.count_nonzero(coordination_reward(e,'recovery'))==0
                c.force_world.copy_(force_saved);term.reset()
            assert before==[digest(m) for m in frozen]
            print('MAPPING_LIMIT_RESET_RECOVERY_GATE_PASS',flush=True)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output,**{k:np.stack([f[k] for f in trace]) for k in trace[0]})
        print('WHOLEBODY_ROLLOUT_PASS',args.mode,args.output,'failures',int(np.stack([f['done'] for f in trace]).any(0).sum()),flush=True)
    finally:env.close()

try:main()
except BaseException:
    import traceback
    traceback.print_exc();sys.stderr.flush();raise
finally:app.close()
