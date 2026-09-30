#new change
"""Independent entry. Audit/eval never optimize; only --mode train trains."""
import argparse,os,sys,json,hashlib,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'softmimic_gym'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['audit','baseline','eval','train'],default='audit')
p.add_argument('--num_envs',type=int,default=32);p.add_argument('--steps',type=int,default=300)
p.add_argument('--seed',type=int,default=0);p.add_argument('--cohort',choices=['mixed','zero','force'],default='mixed')
p.add_argument('--max_iterations',type=int,default=100)
p.add_argument('--run_name',default='stand_force_joint_v1')
p.add_argument('--checkpoint',default=None);p.add_argument('--output',default=None)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
assert 1<=args.max_iterations<=100
if args.mode=='train':assert args.cohort=='mixed' and args.checkpoint is None
if args.mode=='audit':assert args.cohort=='mixed' and args.checkpoint is None
if args.mode in ('baseline','eval'):assert args.output and not Path(args.output).exists()
os.environ['DUAL_ISOLATION']='combined';os.environ['DUAL_FORCE_MAX']='4';os.environ['DIAGNOSTIC_ARM_SCALE']='1.0'
app=AppLauncher(args).app
import numpy as np
import torch
import gymnasium as gym
import softmimic_gym.tasks
from datetime import datetime
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from stand_force_joint_v1_cfg import TASK,StandForceJointCfg,LOWER,UPPER,CHECKPOINT540,NEUTRAL,NAMES,ARM,WAIST,LEG_IDS,UPPER_IDS,state
from stand_force_joint_v1_model import runner_config
from softmimic_gym.assets.robots.g1_compat import G1_SOFTMIMIC_KEYPOINT_BODY_NAMES

def digest(module):
    h=hashlib.sha256()
    for key,v in module.state_dict().items():h.update(key.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def contract(seed):
    files=[LOWER,UPPER,CHECKPOINT540,NEUTRAL,Path(__file__),Path(__file__).with_name('stand_force_joint_v1_cfg.py'),Path(__file__).with_name('stand_force_joint_v1_model.py')]
    return dict(task=TASK,seed=seed,reference='fixed neutral; online IK remains active',force_range_N=[2,4],wait_s=[3,5],
      zero_force_fraction=.25,trainable='entire 540 lower trunk/head plus context and bilateral/waist outputs; critic',
      frozen='old upper JIT; teacher lower JIT; all source checkpoints',leg_action_scale=.30,arms_limit_rad=.02,waist_limit_rad=.01,
      exploration_raw_std={'legs':.02,'upper':.05},budget_iterations=100,save_interval=10,
      evaluation=dict(seeds=[0,7,123],environments=512,steps=1500,cohorts=['zero','force'],stochastic=False,
        select_checkpoint='model_50 (51 updates, RSL zero-based numbering), then model_99 (100 updates); no automatic extra budget',
        zero_gate='each seed >=99% survival and >=99% support-qualified: root peak<50mm, ankle-link XY error peak<20mm, tilt peak<5deg',
        force_gate='each seed >=95% survival, >=95% support-qualified and >=95% qualified recovery; mean wrist error <= initial same-seed +2mm; positive wrist force-response projection >=80% of initial',
        recovery='quiet-window survival AND root<50mm, root-link XY speed<30mm/s, feet<20mm, tilt<5deg, held for final0.5s',
        no_progress='stop this configuration if final multi-seed gains do not meet gates; do not extend automatically'),
      sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},runner=runner_config(seed))

def main():
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    cfg=StandForceJointCfg(zero_force_fraction={'mixed':.25,'zero':1.,'force':0.}[args.cohort])
    cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;cfg.sim.device=args.device
    env=gym.make(TASK,cfg=cfg);e=env.unwrapped
    try:
        env.reset(seed=args.seed);wrapped=RslRlVecEnvWrapper(env)
        logdir=None
        if args.mode=='train':
            logdir=Path('logs/rsl_rl/stand_force_joint_v1')/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_'+args.run_name)
            logdir.mkdir(parents=True,exist_ok=False)
        runner=OnPolicyRunner(wrapped,runner_config(args.seed),log_dir=str(logdir) if logdir else None,device=e.device)
        runner.logger.writer=None;actor=runner.alg.actor;critic=runner.alg.critic
        term=e.action_manager.get_term('joint_pos');teacher=term.teacher_lower
        runner.alg.teacher=teacher
        frozen=(teacher,term.base.upper);frozen_before=[digest(m) for m in frozen]
        optids={id(p) for g in runner.alg.optimizer.param_groups for p in g['params']}
        assert all(not p.requires_grad and id(p) not in optids for m in frozen for p in m.parameters())
        assert all(p.requires_grad and id(p) in optids for m in (actor.mlp,critic.mlp) for p in m.parameters())
        assert e.obs_buf['lower_policy'].shape[-1]==109 and e.obs_buf['policy'].shape[-1]==197
        assert 'anchor_mask' not in actor.obs_groups and 'anchor_mask' not in critic.obs_groups
        if args.checkpoint:
            ckpt=Path(args.checkpoint);meta=json.loads((ckpt.parent/'experiment_contract.json').read_text())
            assert meta['task']==TASK
            for f in (LOWER,UPPER,CHECKPOINT540,NEUTRAL):assert hashlib.sha256(f.read_bytes()).hexdigest()==meta['sha256'][str(f)]
            saved=torch.load(ckpt,map_location=e.device,weights_only=False)
            actor.load_state_dict(saved['actor_state_dict'],strict=True);critic.load_state_dict(saved['critic_state_dict'],strict=True)
        else:
            with torch.no_grad():
                obs=wrapped.get_observations();mean=actor(obs);old=teacher(obs['lower_policy'])
                torch.testing.assert_close(mean[:,LEG_IDS],old,atol=1e-6,rtol=0)
                assert torch.count_nonzero(mean[:,UPPER_IDS])==0
                assert float(runner.alg.anchor_loss(obs,mean))<1e-10
            print('INITIAL_ACTION_EQUIVALENCE_PASS',float((mean[:,LEG_IDS]-old).abs().max()),flush=True)
        print('FULL_LOWER_TRAINABLE_UPPER_BASE_FROZEN_PASS',flush=True)
        if args.mode=='train':
            (logdir/'experiment_contract.json').write_text(json.dumps(contract(args.seed),indent=2))
            runner.save(str(logdir/'model_initial.pt'))
            runner.learn(args.max_iterations,init_at_random_ep_len=False)
            assert frozen_before==[digest(m) for m in frozen]
            print('TRAIN_FINISHED_FROZEN_BASES_UNCHANGED',str(logdir),flush=True);return
        actor_before=digest(actor);critic_before=digest(critic)
        if args.mode=='audit':
            checks=[]
            def no_step(*unused,**unused_kw):
                layers={**{f'lower_{i}':m for i,m in enumerate(actor.mlp.lower) if isinstance(m,torch.nn.Linear)},
                        'force_context':actor.mlp.context,'upper':actor.mlp.upper}
                values={k:float(m.weight.grad.norm()) for k,m in layers.items()}
                for group,indices in [('left_arm',[i for i in ARM if NAMES[i].startswith('left_')]),('right_arm',[i for i in ARM if NAMES[i].startswith('right_')]),('waist',WAIST)]:
                    values[group]=float(actor.mlp.upper.weight.grad[[UPPER_IDS.index(i) for i in indices]].norm())
                assert all(v>0 and np.isfinite(v) for v in values.values()),values
                assert all(torch.isfinite(p.grad).all() for g in runner.alg.optimizer.param_groups for p in g['params'] if p.grad is not None)
                assert any(p.grad is not None and p.grad.norm()>0 for p in critic.parameters())
                checks.append(values)
            runner.alg.optimizer.step=no_step
            active=0;adapted=0
            with torch.inference_mode():
                obs=wrapped.get_observations()
                for _ in range(100):obs,_,_,_=wrapped.step(actor(obs))
                for _ in range(runner.cfg['num_steps_per_env']):
                    obs,reward,done,extras=wrapped.step(runner.alg.act(obs))
                    c=e.command_manager.get_term('upper_body_joints')
                    zero=obs['anchor_mask'][:,0].bool()
                    assert torch.count_nonzero(c.force_world[zero])==0
                    active+=int((c.force_world.norm(dim=-1)>1e-4).sum())
                    adapted+=int(((c.dof_pos[:,1]-c._dof_pos[:,1]).norm(dim=-1)>1e-6).sum())
                    runner.alg.process_env_step(obs,reward,done,extras)
                runner.alg.compute_returns(obs)
            losses=runner.alg.update()
            assert checks and active>0 and adapted>0
            assert digest(actor)==actor_before and digest(critic)==critic_before
            # Verify old-policy constraint has a restoring gradient, independent
            # of the initially zero penalty (no optimizer or checkpoint edits).
            with torch.no_grad():actor.mlp.lower[-1].bias[0]+=.01
            loss=runner.alg.anchor_loss(wrapped.get_observations(),actor(wrapped.get_observations()))
            gradient=torch.autograd.grad(loss,actor.mlp.lower[-1].bias)[0]
            assert float(gradient[0])>0
            with torch.no_grad():actor.mlp.lower[-1].bias.copy_(torch.load(CHECKPOINT540,map_location=e.device,weights_only=False)['actor_state_dict']['mlp.6.bias'])
            assert digest(actor)==actor_before
            # Check delivered targets in articulation order; do not advance
            # physics or change any gains. Repeated apply must not accumulate.
            with torch.no_grad():
                action=actor(wrapped.get_observations()).detach()
                term.process_actions(action);term.apply_actions()
                before=term._asset.data.joint_pos_target[:,term.joint_ids].clone()
                changed=action.clone();changed[:,LEG_IDS]+=.001;changed[:,UPPER_IDS]+=.1
                term.process_actions(changed);term.apply_actions()
                after=term._asset.data.joint_pos_target[:,term.joint_ids].clone()
                expected=(changed.tanh()-action.tanh())*e._wb_limits
                expected[:,LEG_IDS]=.30*(changed[:,LEG_IDS]-action[:,LEG_IDS])
                torch.testing.assert_close(after-before,expected,atol=2e-6,rtol=0)
                term.apply_actions()
                torch.testing.assert_close(term._asset.data.joint_pos_target[:,term.joint_ids],after,atol=1e-7,rtol=0)
            print('29DOF_TARGET_MAPPING_NO_ACCUMULATION_PASS',flush=True)
            if args.output:
                out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
                (out/'experiment_contract.json').write_text(json.dumps(contract(args.seed),indent=2))
                runner.save(str(out/'model_initial.pt'));runner.load(str(out/'model_initial.pt'))
                assert digest(actor)==actor_before and digest(critic)==critic_before
                (out/'audit.json').write_text(json.dumps(dict(gradients=checks,losses=losses,active=active,adapted=adapted,optimizer_updates=0),indent=2))
            print('NO_UPDATE_GRADIENT_ANCHOR_CHECKPOINT_PASS',json.dumps({'checks':checks,'active':active,'adapted':adapted,'optimizer_updates':0}),flush=True)
        else:
            actor.eval().requires_grad_(False);critic.eval().requires_grad_(False)
            r,c,ids,fids,feet,fe,com,forces=state(e)
            bids=[r.body_names.index(s+'_wrist_yaw_link') for s in ('left','right')]
            kids=[list(G1_SOFTMIMIC_KEYPOINT_BODY_NAMES).index(s+'_wrist_yaw_link') for s in ('left','right')]
            alive=torch.ones(e.num_envs,device=e.device,dtype=torch.bool);trace=[];initial_feet=feet.clone()
            nominal0=c._dof_pos[:,1].clone();active=0;adapted=0
            with torch.inference_mode():
                for step in range(args.steps):
                    obs=wrapped.get_observations();action=actor(obs)
                    if args.mode=='baseline':
                        action.zero_();action[:,LEG_IDS]=teacher(obs['lower_policy'])
                    r,c,ids,fids,feet,fe,com,forces=state(e)
                    torch.testing.assert_close(c._dof_pos[:,1],nominal0,atol=1e-6,rtol=0)
                    assert c._dof_vel.abs().max()<1e-6
                    fields=dict(alive=alive,q=r.data.joint_pos[:,ids],ref=c.dof_pos[:,1],nominal_ref=c._dof_pos[:,1],
                        root=r.data.root_pos_w-e.scene.env_origins,root_error=r.data.root_pos_w[:,:2]-c.root_state['root_pos'][:,1,:2],
                        root_vel=r.data.root_link_lin_vel_w,root_com_vel=r.data.root_com_lin_vel_w,
                        foot_error=fe,foot_drift=feet-initial_feet,foot_load=forces[:,:,2],com=com-e.scene.env_origins,
                        tilt=torch.acos((-r.data.projected_gravity_b[:,2]).clamp(-1,1)),
                        wrist=r.data.body_pos_w[:,bids]-e.scene.env_origins[:,None],wrist_ref=c.keypoints[:,1,kids,:3]-e.scene.env_origins[:,None],
                        wrist_nominal=c._keypoints[:,1,kids,:3]-e.scene.env_origins[:,None],action=action,
                        force=c.force_world,event_index=c._independent_event_index,event_time=c._event_time,ramp=c._ramp_time,hold=c._hold_time,
                        sampled_peak=c._sampled_peak_force_world,accepted_peak=c._accepted_peak_force_world,body=c.selected_body_ids,
                        target_displacement=c.target_displacement_world,ik_scale=c.ik_feasible_scale)
                    frame={k:v.detach().cpu().numpy().copy() for k,v in fields.items()}
                    obs,reward,done,truncated,_=env.step(action)
                    assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
                    for k,v in dict(done=done,truncated=truncated,target=r.data.joint_pos_target[:,ids],
                                    correction=e._wb_residual,release_age=e._wb_release_age).items():frame[k]=v.detach().cpu().numpy().copy()
                    trace.append(frame);alive &= ~done
                    active+=int((c.force_world.norm(dim=-1)>1e-4).sum());adapted+=int(((c.dof_pos[:,1]-c._dof_pos[:,1]).norm(dim=-1)>1e-6).sum())
                    if (step+1)%100==0:print('PROGRESS',step+1,'first_episode_failures',int((~alive).sum()),flush=True)
            Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(args.output,**{k:np.stack([f[k] for f in trace]) for k in trace[0]},dt=np.asarray(e.step_dt),
                seed=np.asarray(args.seed),cohort=np.asarray(args.cohort),checkpoint=np.asarray(args.checkpoint or 'initial540'),joint_names=np.asarray(NAMES),wrist_body_ids=np.asarray(bids))
            Path(args.output).with_suffix('.json').write_text(json.dumps(dict(contract=contract(args.seed),mode=args.mode,cohort=args.cohort,
                num_envs=args.num_envs,steps=args.steps,checkpoint=args.checkpoint,optimizer_updates=0),indent=2))
            print('FROZEN_EVAL_PASS',json.dumps({'failed':int((~alive).sum()),'active':active,'adapted':adapted,'steps':args.steps,'num_envs':args.num_envs}),flush=True)
        assert digest(actor)==actor_before and digest(critic)==critic_before
        assert frozen_before==[digest(m) for m in frozen]
        print('ALL_NETWORK_WEIGHTS_UNCHANGED',flush=True)
    finally:env.close()

try:main()
except BaseException:
    import traceback
    traceback.print_exc();sys.stderr.flush();raise
finally:app.close()
