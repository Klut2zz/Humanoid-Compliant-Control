#new change
"""Independent <=10-iteration PPO entry: model74 actor, calibrated V2 critic."""
from pathlib import Path


def build_source():
    root=Path(__file__).resolve().parents[1]
    path=Path(__file__).with_name('train_dual540_recovery_window.py')
    wrapper=path.read_text();ns={'__file__':str(path)}
    exec(compile(wrapper[:wrapper.rindex('exec(compile(')],str(path),'exec'),ns)
    source=ns['source']
    actor=root/'logs/rsl_rl/dual540_recovery_window_force/2026-09-23_10-06-33_recovery_window_continue50_v1/model_74.pt'
    critic=root/'logs/stand_coordination_v1/dual540_recovery_window_train_v1/critic_calibration_v2_terminal020/critic_only.pt'
    changes={
        'args=p.parse_args()':
        "p.add_argument('--critic_checkpoint',default="+repr(str(critic))+")\np.set_defaults(init_checkpoint="+repr(str(actor))+",resume_optimizer=True,max_iterations=10,run_name='critic_v2_short10_v1')\nargs=p.parse_args()\nassert args.resume_optimizer\nassert 1<=args.max_iterations<=10, 'This controlled entry allows at most 10 iterations'",
        "Path('logs/rsl_rl/dual540_recovery_window_force')":
        "Path('logs/rsl_rl/dual540_critic_v2_short')",
        'runner=OnPolicyRunner(wrapped,rcfg,':
        "rcfg['save_interval']=1\n            runner=OnPolicyRunner(wrapped,rcfg,",
        "            print('WHOLEBODY_ENTRY_PASS'":
        """            #new change: retain actor Adam moments, discard stale critic moments.
            critic_path=Path(args.critic_checkpoint).resolve()
            calibrated=torch.load(critic_path,map_location=e.device,weights_only=False)
            assert calibrated['artifact_type']=='critic_only_not_ppo_resume'
            assert calibrated.get('validation_pass') is True
            assert Path(calibrated['source_checkpoint']).resolve()==init_path
            actor_before_critic_load=digest(actor)
            critic_net=runner.alg.critic
            critic_net.load_state_dict(calibrated['critic_state_dict'],strict=True)
            cleared=0
            for parameter in critic_net.parameters():
                if parameter in runner.alg.optimizer.state:
                    runner.alg.optimizer.state.pop(parameter);cleared+=1
            assert cleared==len(list(critic_net.parameters()))
            assert all(parameter in runner.alg.optimizer.state for parameter in actor.parameters())
            assert actor_before_critic_load==digest(actor)
            assert all(torch.equal(value,checkpoint['actor_state_dict'][name].to(value.device)) for name,value in actor.state_dict().items())
            assert runner.cfg['save_interval']==1
            print('V2_SHORT_ENTRY_PASS',json.dumps(dict(actor_exact=True,critic_exact=True,
                actor_optimizer_restored=True,critic_optimizer_reset=True,save_interval=1,
                next_iteration=runner.current_learning_iteration,critic_checkpoint=str(critic_path))),flush=True)
            print('WHOLEBODY_ENTRY_PASS'""",
        "            (logdir/'base_controllers.json').write_text(json.dumps(manifest,indent=2))":
        """            #new change
            manifest['runner']['save_interval']=1
            manifest.update(critic_checkpoint=str(critic_path),
                critic_checkpoint_sha256=hashlib.sha256(critic_path.read_bytes()).hexdigest(),
                actor_optimizer_restored=True,critic_optimizer_reset=True,controlled_iterations=args.max_iterations)
            (logdir/'base_controllers.json').write_text(json.dumps(manifest,indent=2))""",
    }
    for old,new in changes.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    return source,path


if __name__=='__main__':
    source,path=build_source()
    exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
