#new change
"""Resume model84 unchanged, including critic and Adam; no training on import."""
from pathlib import Path


def build_source():
    path = Path(__file__).with_name('train_dual540_recovery_window.py')
    wrapper = path.read_text()
    ns = {'__file__': str(path)}
    exec(compile(wrapper[:wrapper.rindex('exec(compile(')], str(path), 'exec'), ns)
    source = ns['source']
    checkpoint = path.parents[1] / 'logs/rsl_rl/dual540_critic_v2_short/2026-09-24_01-30-29_critic_v2_short10_v1/model_84.pt'
    changes = {
        'args=p.parse_args()':
        "p.set_defaults(init_checkpoint=" + repr(str(checkpoint)) + ",resume_optimizer=True,max_iterations=50,run_name='model84_continue50_v1')\nargs=p.parse_args()\nassert args.resume_optimizer\nassert 1 <= args.max_iterations <= 50",
        "Path('logs/rsl_rl/dual540_recovery_window_force')":
        "Path('logs/rsl_rl/dual540_model84_continue')",
        'runner=OnPolicyRunner(wrapped,rcfg,':
        #new change: preserve the declarative config before RSL-RL mutates it.
        "rcfg['save_interval']=10\n            import copy\n            requested_rcfg=copy.deepcopy(rcfg)\n            runner=OnPolicyRunner(wrapped,rcfg,",
        "            print('WHOLEBODY_ENTRY_PASS'":
        """            #new change: verify resume preserves both networks and PPO configuration.
            for network, key in ((actor,'actor_state_dict'),(runner.alg.critic,'critic_state_dict')):
                assert all(torch.equal(v,checkpoint[key][k].to(v.device)) for k,v in network.state_dict().items())
            old_cfg=dict(provenance['runner']);old_cfg['save_interval']=10;old_cfg['seed']=args.seed
            # Original check retained: assert rcfg==old_cfg
            #new change: compare pre-construction configs, not runtime metadata.
            assert requested_rcfg==old_cfg, 'Unexpected PPO or model configuration change'
            assert all(p in runner.alg.optimizer.state for net in (actor,runner.alg.critic) for p in net.parameters())
            print('MODEL84_CONTINUE_ENTRY_PASS',flush=True)
            print('WHOLEBODY_ENTRY_PASS'""",
        "            (logdir/'base_controllers.json').write_text(json.dumps(manifest,indent=2))":
        "            manifest['runner']['save_interval']=10\n            manifest['controlled_iterations']=args.max_iterations\n            (logdir/'base_controllers.json').write_text(json.dumps(manifest,indent=2))",
    }
    for old, new in changes.items():
        assert source.count(old)==1, old
        source=source.replace(old,new)
    return source,path


if __name__=='__main__':
    source,path=build_source()
    exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
