#new change
"""Warm-start independent recovery training. The original trainer is unchanged."""
from pathlib import Path
path=Path(__file__).with_name('train_dual540_wholebody_force.py')
source=path.read_text()
old_task='Isaac-G1-Dual540-WholeBodyForceResidual-v0'
new_task='Isaac-G1-Dual540-WholeBodyForce-RecoveryWindow-Train-v0'
assert source.count(old_task)==2
source=source.replace(old_task,new_task)
initial=Path(__file__).resolve().parents[1]/'logs/rsl_rl/dual540_wholebody_force/2026-09-23_00-38-39_wholebody_force_v1/model_24.pt'
changes={
 "p.add_argument('--mode',choices=['baseline','zero','gradient','train'],default='zero')":
 "p.add_argument('--mode',choices=['audit','gradient','train'],default='audit')",
 "p.add_argument('--output',default=None)":
 #new change
 "p.add_argument('--output',default=None)\np.add_argument('--resume_optimizer',action='store_true')\np.add_argument('--init_checkpoint',default="+repr(str(initial))+")",
 "p.add_argument('--run_name',default='wholebody_force_v1')":
 "p.add_argument('--run_name',default='recovery_window_v1')",
 "from neutral_dual_adapt_cfg import UPPER":
 "from neutral_dual_adapt_cfg import UPPER\nfrom dual540_recovery_train_cfg import TASK,RecoveryWindowTrainCfg",
 "cls=MixedFrozenCfg if args.mode=='baseline' else WholeBodyForceCfg":
 "cls=RecoveryWindowTrainCfg",
 "Path('logs/rsl_rl/dual540_wholebody_force')":
 "Path('logs/rsl_rl/dual540_recovery_window_force')",
 "with torch.no_grad():assert torch.count_nonzero(actor(wrapped.get_observations()))==0":
 """with torch.no_grad():assert torch.count_nonzero(actor(wrapped.get_observations()))==0
            init_path=Path(args.init_checkpoint).resolve()
            provenance=json.loads((init_path.parent/'base_controllers.json').read_text())
            for base_path in (LOWER,UPPER):
                assert hashlib.sha256(base_path.read_bytes()).hexdigest()==provenance['sha256'][str(base_path)]
            assert provenance['limits_rad']==LIMITS
            checkpoint=torch.load(init_path,map_location=e.device,weights_only=False)
            actor.load_state_dict(checkpoint['actor_state_dict'],strict=True)
            runner.alg.critic.load_state_dict(checkpoint['critic_state_dict'],strict=True)
            assert len(runner.alg.optimizer.state)==0
            assert runner.current_learning_iteration==0
            #new change
            if args.resume_optimizer:
                assert provenance['task']==TASK, 'Resume requires the same recovery training task'
                assert provenance['zero_force_fraction']==args.zero_force_fraction
                assert provenance['force_max']==args.force_max
                runner.load(str(init_path),map_location=e.device)
                assert len(runner.alg.optimizer.state)>0
                runner.current_learning_iteration=int(checkpoint['iter'])+1
                print('RECOVERY_RESUME_PASS',json.dumps({'checkpoint':str(init_path),
                    'optimizer_restored':True,'next_iteration':runner.current_learning_iteration}),flush=True)
            else:
                print('RECOVERY_WARM_START_PASS',json.dumps({'checkpoint':str(init_path),
                'source_iteration':checkpoint['iter'],'fresh_optimizer':True,
                'new_run_iteration':0,'recovery_wait_s':[3,5]}),flush=True)""",
 "(logdir/'base_controllers.json').write_text(json.dumps(manifest,indent=2))":
 """manifest.update(task=TASK,init_checkpoint=str(init_path),
                init_checkpoint_sha256=hashlib.sha256(init_path.read_bytes()).hexdigest(),
                #new change
                source_iteration=checkpoint['iter'],fresh_optimizer=not args.resume_optimizer,
                next_iteration=runner.current_learning_iteration,
                recovery_wait_range_s=[3.,5.],initial_wait_range_s=[.5,1.5],
                dedicated_recovery_reward_seconds=2.)
            (logdir/'base_controllers.json').write_text(json.dumps(manifest,indent=2))""",
 "runner.save(str(logdir/'model_zero.pt'))":
 "runner.save(str(logdir/'model_start.pt'))",
 "for _ in range(100):obs,_,_,_=wrapped.step(torch.zeros((e.num_envs,29),device=e.device))":
 "for _ in range(100):obs,_,_,_=wrapped.step(actor(obs,stochastic_output=False))",
 "event_index=c._independent_event_index).items()":
 "event_index=c._independent_event_index,event_time=c._event_time,ramp=c._ramp_time,hold=c._hold_time).items()",
 "assert args.output and not Path(args.output).exists()":
 "assert args.output and not Path(args.output).exists()\n        actor_before=digest(actor)\n        critic_before=digest(runner.alg.critic)",
 "Path(args.output).parent.mkdir(parents=True,exist_ok=True)":
 "assert actor_before==digest(actor) and critic_before==digest(runner.alg.critic)\n        assert before==[digest(m) for m in frozen]\n        print('AUDIT_ALL_NETWORKS_UNCHANGED',flush=True)\n        Path(args.output).parent.mkdir(parents=True,exist_ok=True)",
}
for old,new in changes.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
