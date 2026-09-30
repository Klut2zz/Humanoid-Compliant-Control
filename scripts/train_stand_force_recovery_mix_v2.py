#new change
"""Independent V2 recovery branch from V1 model50; only cohort weights change. Default audit NEVER updates weights.

Training must be requested explicitly with --mode train. It restores the V1 model50
actor, critic AND Adam state, runs exactly 50 updates, and saves every 10.
Evaluation modes zero/low/medium use unbiased held-out event sampling.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'logs/rsl_rl/stand_force_recovery_mix_v1/2026-09-28_10-57-44_stand_force_recovery_mix_v1/model_50.pt'
PARENT_TASK='Isaac-G1-StandForceJoint-RecoveryMixV1-v0'
BASE_TASK='Isaac-G1-StandForceJointV1-v0'
TASK_NAME='Isaac-G1-StandForceJoint-RecoveryMixV2-v0'
sys.path.insert(0,str(ROOT/'softmimic_gym'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--mode',choices=('audit','eval','train'),default='audit')
p.add_argument('--task',default=TASK_NAME)
p.add_argument('--sampling',choices=('mixed','zero','low','medium'),default='mixed')
p.add_argument('--checkpoint',default=str(SOURCE))
p.add_argument('--num_envs',type=int,default=128)
p.add_argument('--steps',type=int,default=500)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--max_iterations',type=int,default=50)
p.add_argument('--run_name',default='stand_force_recovery_mix_v2')
p.add_argument('--output')
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
assert args.task==TASK_NAME
if args.mode=='train':
    assert args.sampling=='mixed' and args.max_iterations==50
    assert args.seed not in (0,7,123), 'Reserve these event seeds for evaluation.'
    assert Path(args.checkpoint).resolve()==SOURCE
if args.sampling=='mixed':assert args.num_envs%4==0
if args.mode=='audit':assert args.steps>=250, 'Audit needs a 48-step gradient sample and natural force release.'
if args.mode!='train':assert args.output and not Path(args.output).exists()
os.environ['DUAL_ISOLATION']='combined';os.environ['DUAL_FORCE_MAX']='4';os.environ['DIAGNOSTIC_ARM_SCALE']='1.0'
app=AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
import softmimic_gym.tasks
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from stand_force_recovery_mix_v2_cfg import TASK, StandForceRecoveryMixV2Cfg
from stand_force_recovery_mix_cfg import StandForceRecoveryMixCfg as V1Cfg
from stand_force_recovery_sampling_v2 import curriculum_cohort
from stand_force_joint_v1_cfg import NAMES, ARM, WAIST, LEG_IDS, UPPER_IDS, state
from stand_force_joint_v1_model import runner_config
from softmimic_gym.assets.robots.g1_compat import G1_SOFTMIMIC_KEYPOINT_BODY_NAMES


def digest(module):
    h=hashlib.sha256()
    for k,v in module.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def same_tree(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same_tree(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same_tree(x,y)
    else:assert a==b,(a,b)


def verify_files(protected):
    for file,sha in protected.items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==sha,file


def make_contract(protected):
    additional=[Path(__file__),Path(__file__).with_name('stand_force_recovery_mix_v2_cfg.py'),
                Path(__file__).with_name('stand_force_recovery_sampling_v2.py')]
    return dict(task=TASK,source_checkpoint=str(SOURCE),source_iteration=50,
        actor_critic_optimizer_restored=True,branch_updates=50,save_completed_updates=[10,20,30,40,50],
        curriculum={'zero':.25,'low_2to4N':.25,'hard_4to7N':.5,'hard_rear_event_probability':.5,
                    'rear_azimuth':'world -X +/-30 degrees; fixed nominal Stand heading',
                    'heldout_evaluation_direction_bias':False},
        unchanged=['architecture','rewards','actor/critic observation','action mapping','gains','friction','collision',
                   'half-cosine envelope','3-5 second post-release wait','learning rate','exploration std','anchor'],
        runner=runner_config(args.seed),trainable='entire existing lower trunk/head; force context; bilateral/waist output; critic',
        frozen='teacher540 JIT; original upper1375 JIT and supervised arms; fixed exploration std; source files/checkpoints',
        evaluation={'seeds':[0,7,123],'environments':512,'steps':1500,'sampling':['zero','low','medium'],
            'baseline':'V1 model50 is immediate comparison; original model99 remains rollback; deterministic, unbiased evaluation',
            'zero':'per-seed survival and support-qualified >=99%; no failure increase versus model99',
            'low':'per-seed survival/support/recovery >=95%; no failure increase versus model99',
            'medium':'per-seed survival/support/recovery >=95%; fewer pooled recovery failures than V1 model50 (7/1536 on recorded seeds)',
            'support':'root peak<50mm; foot-link XY error peak<20mm; tilt peak<5deg',
            'recovery':'complete quiet window; final0.5s root<50mm, root XY speed<30mm/s, feet<20mm, tilt<5deg',
            'wrist':'100mm application tolerance; report MAE/P95/peaks, not only mean',
            'no_progress':'no automatic additional iterations; stop if no population recovery improvement'},
        sha256={**protected,**{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in additional}})


class RecoveryRunner(OnPolicyRunner):
    def save(self,path,infos=None):
        # Disable the base runner's zero-based 0/10/.../49 schedule only.
        # save_completed below is invoked AFTER updates 10/20/30/40/50.
        return None

    def save_completed(self,folder,completed):
        path=folder/('model_initial.pt' if completed==0 else f'model_{completed}.pt')
        assert not path.exists()
        value=self.alg.save();value['iter']=completed
        value['infos']={'source_checkpoint':str(SOURCE),'source_iteration':50,'completed_branch_updates':completed}
        torch.save(value,path)
        print('CHECKPOINT_SAVED_COMPLETED_UPDATES',completed,str(path),flush=True)


def main():
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    ckpt=Path(args.checkpoint).resolve();meta=json.loads((ckpt.parent/'experiment_contract.json').read_text())
    assert meta['task'] in (PARENT_TASK,TASK)
    protected={**meta['sha256'],str(ckpt):hashlib.sha256(ckpt.read_bytes()).hexdigest()}
    protected[str(SOURCE)]=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    verify_files(protected)
    #new change: compare the entire V2 config to V1, excluding only command implementation.
    cfg=StandForceRecoveryMixV2Cfg(recovery_sampling=args.sampling)
    previous_cfg=V1Cfg(recovery_sampling=args.sampling)
    old_cfg,new_cfg=previous_cfg.to_dict(),cfg.to_dict()
    old_cfg['commands']['upper_body_joints']['class_type']=new_cfg['commands']['upper_body_joints']['class_type']
    assert json.dumps(old_cfg,default=repr,sort_keys=True)==json.dumps(new_cfg,default=repr,sort_keys=True)
    print('V2_CONFIG_ONLY_SAMPLER_CHANGED_PASS',flush=True)
    cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;cfg.sim.device=args.device
    env=gym.make(TASK,cfg=cfg);e=env.unwrapped
    try:
        env.reset(seed=args.seed);wrapped=RslRlVecEnvWrapper(env)
        folder=None
        if args.mode=='train':
            folder=ROOT/'logs/rsl_rl/stand_force_recovery_mix_v2'/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_'+args.run_name)
            folder.mkdir(parents=True,exist_ok=False)
        config=runner_config(args.seed)
        expected=copy.deepcopy(meta['runner']);expected['seed']=args.seed
        assert config==expected,'Do not silently change PPO, std or network configuration.'
        runner=RecoveryRunner(wrapped,config,log_dir=str(folder) if folder else None,device=e.device)
        runner.logger.writer=None
        runner.load(str(ckpt),strict=True,map_location=e.device)
        actor,critic=runner.alg.actor,runner.alg.critic
        term=e.action_manager.get_term('joint_pos');runner.alg.teacher=term.teacher_lower
        bases=(term.teacher_lower,term.base.upper);base_hashes=[digest(m) for m in bases]
        saved=torch.load(ckpt,map_location=e.device,weights_only=False)
        same_tree(actor.state_dict(),saved['actor_state_dict']);same_tree(critic.state_dict(),saved['critic_state_dict'])
        same_tree(runner.alg.optimizer.state_dict(),saved['optimizer_state_dict'])
        assert runner.alg.learning_rate==1e-5 and all(g['lr']==1e-5 for g in runner.alg.optimizer.param_groups)
        optids={id(v) for g in runner.alg.optimizer.param_groups for v in g['params']}
        assert all(p.requires_grad and id(p) in optids for m in (actor.mlp,critic.mlp) for p in m.parameters())
        assert all(not p.requires_grad and id(p) not in optids for m in bases for p in m.parameters())
        assert all(not p.requires_grad and id(p) not in optids for p in actor.distribution.parameters())
        assert e.obs_buf['lower_policy'].shape[-1]==109 and e.obs_buf['policy'].shape[-1]==197
        c=e.command_manager.get_term('upper_body_joints')
        expected_cohorts=curriculum_cohort(torch.arange(e.num_envs,device=e.device),args.sampling)
        assert torch.equal(c._recovery_cohort,expected_cohorts)
        assert torch.equal(e.obs_buf['anchor_mask'][:,0].bool(),expected_cohorts==0)
        counts=torch.bincount(expected_cohorts,minlength=3).tolist()
        if args.sampling=='mixed':
            assert counts==[e.num_envs//4,e.num_envs//4,e.num_envs//2]
        print('V2_COHORT_AND_ZERO_ANCHOR_PASS',counts,flush=True)
        torch.testing.assert_close(actor.distribution.log_std_param.exp()[LEG_IDS],
                                  torch.full((len(LEG_IDS),),.02,device=e.device))
        torch.testing.assert_close(actor.distribution.log_std_param.exp()[UPPER_IDS],
                                  torch.full((len(UPPER_IDS),),.05,device=e.device))
        print('SOURCE_ACTOR_CRITIC_ADAM_EXACT_LOAD_PASS',str(ckpt),flush=True)
        print('TRAINABLE_FULL_LOWER_CONTEXT_BILATERAL_WAIST_CRITIC; BASES_AND_STD_FROZEN',flush=True)
        runner.current_learning_iteration=0
        actor_hash,critic_hash=digest(actor),digest(critic)
        if args.mode=='train':
            (folder/'experiment_contract.json').write_text(json.dumps(make_contract(protected),indent=2))
            runner.save_completed(folder,0)
            original_update=runner.alg.update;completed=0
            def update_and_checkpoint():
                nonlocal completed
                value=original_update();completed+=1
                if completed%10==0:
                    verify_files(protected);assert base_hashes==[digest(m) for m in bases]
                    runner.save_completed(folder,completed)
                    r,c,ids,fids,feet,fe,com,loads=state(e)
                    metrics={}
                    for g,name in enumerate(('zero','low','hard')):
                        mask=c._recovery_cohort==g
                        metrics[name]={'environments':int(mask.sum()),'active_force':int((c.force_world[mask].norm(dim=-1)>1e-4).sum()),
                            'root_speed_mm_s':float(r.data.root_link_lin_vel_w[mask,:2].norm(dim=-1).mean()*1000),
                            'root_error_mm':float((r.data.root_pos_w[mask,:2]-c.root_state['root_pos'][mask,1,:2]).norm(dim=-1).mean()*1000)}
                    print('RECOVERY_COHORT_SNAPSHOT',completed,json.dumps(metrics),flush=True)
                return value
            runner.alg.update=update_and_checkpoint
            runner.learn(50,init_at_random_ep_len=False)
            assert completed==50 and base_hashes==[digest(m) for m in bases]
            verify_files(protected)
            print('TRAIN_FINISHED_50_UPDATES_FROZEN_BASES_UNCHANGED',str(folder),flush=True)
            return

        # No policy/value optimizer update is possible in audit or eval.
        optimizer_before=copy.deepcopy(runner.alg.optimizer.state_dict())
        checks=[]
        def no_step(*unused,**kwargs):
            layers={**{f'lower_{i}':m for i,m in enumerate(actor.mlp.lower) if isinstance(m,torch.nn.Linear)},
                    'context':actor.mlp.context,'upper':actor.mlp.upper}
            values={k:float(m.weight.grad.norm()) for k,m in layers.items()}
            for name,indices in [('left_arm',[i for i in ARM if NAMES[i].startswith('left_')]),
                                 ('right_arm',[i for i in ARM if NAMES[i].startswith('right_')]),('waist',WAIST)]:
                values[name]=float(actor.mlp.upper.weight.grad[[UPPER_IDS.index(i) for i in indices]].norm())
            assert all(np.isfinite(v) and v>0 for v in values.values()),values
            assert all(torch.isfinite(v.grad).all() for m in (actor,critic) for v in m.parameters() if v.grad is not None)
            assert any(v.grad is not None and v.grad.norm()>0 for v in critic.parameters())
            assert all(v.grad is None for m in bases for v in m.parameters())
            assert all(v.grad is None for v in actor.distribution.parameters())
            values['critic']=float(torch.sqrt(sum(p.grad.square().sum() for p in critic.parameters() if p.grad is not None)))
            checks.append(values)
        runner.alg.optimizer.step=no_step if args.mode=='audit' else lambda *a,**k: (_ for _ in ()).throw(RuntimeError('No eval updates'))
        reference=copy.deepcopy(actor)
        reference.load_state_dict(saved['actor_state_dict'],strict=True)
        reference.eval()
        for parameter in reference.parameters():parameter.requires_grad_(False)
        if args.mode=='eval':
            for m in (actor,critic):
                m.eval()
                for parameter in m.parameters():parameter.requires_grad_(False)
        r,c,ids,fids,feet,fe,com,loads=state(e)
        bids=[r.body_names.index(s+'_wrist_yaw_link') for s in ('left','right')]
        kids=[list(G1_SOFTMIMIC_KEYPOINT_BODY_NAMES).index(s+'_wrist_yaw_link') for s in ('left','right')]
        physical=torch.zeros_like(c.force_world);physical_max=torch.zeros(e.num_envs,device=e.device)
        original_apply=term.apply_actions
        def monitored_apply():
            original_apply()
            local=r.permanent_wrench_composer.composed_force_as_torch[c._env_ids,c.selected_body_ids]
            physical.copy_(quat_apply(r.data.body_quat_w[c._env_ids,c.selected_body_ids],local))
            physical_max.copy_(torch.maximum(physical_max,(physical-c.force_world).abs().max(-1).values))
        term.apply_actions=monitored_apply
        alive=torch.ones(e.num_envs,device=e.device,dtype=torch.bool);initial_feet=feet.clone()
        trace=[];max_mean_diff=0.;natural_release_count=0
        for step in range(args.steps):
            with torch.inference_mode():
                obs=wrapped.get_observations();mean=actor(obs)
                diff=float((mean-reference(obs)).abs().max());max_mean_diff=max(max_mean_diff,diff)
                assert diff==0.,diff
                stochastic=args.mode=='audit' and 100<=step<148
                action=runner.alg.act(obs) if stochastic else mean
                r,c,ids,fids,feet,fe,com,loads=state(e)
                zero=c._recovery_cohort==0
                assert torch.count_nonzero(c.force_world[zero])==0
                torch.testing.assert_close(c.dof_pos[zero],c._dof_pos[zero],atol=0,rtol=0)
                fields=dict(alive=alive,q=r.data.joint_pos[:,ids],ref=c.dof_pos[:,1],nominal_ref=c._dof_pos[:,1],
                    root=r.data.root_pos_w-e.scene.env_origins,root_error=r.data.root_pos_w[:,:2]-c.root_state['root_pos'][:,1,:2],
                    root_vel=r.data.root_link_lin_vel_w,root_com_vel=r.data.root_com_lin_vel_w,
                    foot_error=fe,foot_drift=feet-initial_feet,foot_load=loads[:,:,2],com=com-e.scene.env_origins,
                    tilt=torch.acos((-r.data.projected_gravity_b[:,2]).clamp(-1,1)),
                    wrist=r.data.body_pos_w[:,bids]-e.scene.env_origins[:,None],wrist_ref=c.keypoints[:,1,kids,:3]-e.scene.env_origins[:,None],
                    wrist_nominal=c._keypoints[:,1,kids,:3]-e.scene.env_origins[:,None],action=action,
                    force=c.force_world,event_index=c._independent_event_index,event_time=c._event_time,ramp=c._ramp_time,hold=c._hold_time,
                    sampled_peak=c._sampled_peak_force_world,accepted_peak=c._accepted_peak_force_world,body=c.selected_body_ids,
                    target_displacement=c.target_displacement_world,ik_scale=c.ik_feasible_scale,ik_pending=c._solution_pending,
                    curriculum_cohort=c._recovery_cohort,rear_biased=c._recovery_rear_biased)
                frame={k:v.detach().cpu().numpy().copy() for k,v in fields.items()}
                previous_time=c._event_time.clone()
                obs,reward,done,truncated,extras=env.step(action)
                if stochastic:runner.alg.process_env_step(obs,reward,done|truncated,extras)
                assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
                release=(previous_time>=0)&(c._event_time<0)&~done&~truncated
                if release.any():
                    wait=-c._event_time[release]
                    assert (wait>=3.-1e-5).all() and (wait<=5.+1e-5).all()
                    natural_release_count+=int(release.sum())
                for k,v in dict(done=done,truncated=truncated,target=r.data.joint_pos_target[:,ids],correction=e._wb_residual,
                                release_age=e._wb_release_age,physical_force=physical).items():frame[k]=v.detach().cpu().numpy().copy()
                trace.append(frame);alive &= ~done
                if args.mode=='audit' and step==147:runner.alg.compute_returns(obs)
            if args.mode=='audit' and step==147:
                losses=runner.alg.update()
                print('NO_UPDATE_GRADIENT_PASS',json.dumps({'optimizer_updates':0,'intercepted_calls':len(checks),'losses':losses}),flush=True)
            if (step+1)%100==0:print('PROGRESS',step+1,'failures',int((~alive).sum()),flush=True)
        assert physical_max.max()<1e-4
        assert digest(actor)==actor_hash and digest(critic)==critic_hash
        same_tree(runner.alg.optimizer.state_dict(),optimizer_before)
        assert base_hashes==[digest(m) for m in bases];verify_files(protected)
        if args.mode=='audit':assert checks and natural_release_count>0
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out,**{k:np.stack([f[k] for f in trace]) for k in trace[0]},dt=np.asarray(e.step_dt),
            seed=np.asarray(args.seed),cohort=np.asarray('zero' if args.sampling=='zero' else 'force'),sampling=np.asarray(args.sampling),
            checkpoint=np.asarray(str(ckpt)),joint_names=np.asarray(NAMES),wrist_body_ids=np.asarray(bids))
        result=dict(mode=args.mode,sampling=args.sampling,checkpoint=str(ckpt),steps=args.steps,num_envs=args.num_envs,
            max_identical_observation_mean_action_difference=max_mean_diff,optimizer_updates=0,
            actor_critic_optimizer_bases_unchanged=True,gradient_checks=checks,natural_release_windows_verified=natural_release_count,
            cohort_counts=counts,configuration_change_only_sampler=True,
            physical_force_max_error_N=float(physical_max.max()),failures=int((~alive).sum()))
        out.with_suffix('.json').write_text(json.dumps(result,indent=2))
        print('RECOVERY_MIX_AUDIT_EVAL_PASS',json.dumps(result),flush=True)
    finally:env.close()


try:main()
except BaseException:
    import traceback
    traceback.print_exc();sys.stderr.flush();raise
finally:app.close()
