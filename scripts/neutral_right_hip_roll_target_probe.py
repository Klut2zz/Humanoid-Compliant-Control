# new change
"""Frozen target-only probe using audited deterministic prefix reconstruction.

Does not claim to restore native PhysX solver caches. Run separate fresh
processes and require exact prefix/snapshot and zero-treatment replay equality.
The original diagnostic source is preserved and checked at each injection site.
"""
from pathlib import Path

base = Path('/home/noah/mingkai/general_softmimic/scripts/neutral_right_ankle_kp_ab.py')
source = base.read_text()

def replace_once(old, new):
    global source
    if source.count(old) != 1:
        raise RuntimeError('Diagnostic source changed: ' + old[:80])
    source = source.replace(old, new, 1)

replace_once('AppLauncher.add_app_launcher_args(p)', """# new change
p.add_argument('--target_bias_rad', type=float, choices=[-.01, 0., .01], default=0.)
p.add_argument('--probe_start', type=int, default=60)
AppLauncher.add_app_launcher_args(p)""")
replace_once('app = AppLauncher(args).app', """# new change
if args.right_ankle_kp != 30 or args.num_envs != 1 or args.branch != 'B_model1375' or args.height_mode != 'aligned' or args.support_actor_scale != 1:
    raise ValueError('Probe requires aligned, frozen B, one environment, Kp30, unscaled actor')
app = AppLauncher(args).app""")
replace_once("        initial = None", """        # new change: modify processed target only, never raw actor action/history.
        ankle_column = list(G1_SOFTMIMIC_29_JOINT_NAMES).index('right_hip_roll_joint')
        probe_term = e.action_manager.get_term('joint_pos')
        original_process = probe_term.process_actions
        probe_bias = 0.
        def probe_process(actions):
            original_process(actions)
            probe_term._processed_actions[:, ankle_column] += probe_bias
        probe_term.process_actions = probe_process
        (out/'probe_parameters.json').write_text(json.dumps({
            'target_joint':'right_hip_roll_joint',
            'target_bias_rad':args.target_bias_rad, 'probe_start':args.probe_start,
            'pairing':'fresh-process deterministic prefix; requires external equality audit',
            'application':'processed position target only; actor and raw action unchanged',
            'duration':'held from probe_start until first termination; no bias after reset'
        },indent=2))
        initial = None""")
replace_once('                        if policy is None:', """                        # new change: serialize exposed state/history before intervention.
                        if step == args.probe_start:
                            def tensors(obj):
                                return {n:v.detach().cpu().clone() for n,v in vars(obj).items() if isinstance(v,torch.Tensor)}
                            snapshot = {
                                'root':robot.data.root_state_w.cpu().clone(),
                                'q':robot.data.joint_pos.cpu().clone(),
                                'dq':robot.data.joint_vel.cpu().clone(),
                                'body_state':robot.data.body_state_w.cpu().clone(),
                                'obs':{n:v.cpu().clone() for n,v in obs.items()},
                                'command':tensors(command), 'env':tensors(e),
                                'action_manager':tensors(e.action_manager),
                                'action_term':tensors(probe_term),
                                'history':{},
                                'contact':sensor.data.net_forces_w.cpu().clone(),
                                'torch_rng':torch.get_rng_state(),
                                'cuda_rng':torch.cuda.get_rng_state_all(),
                            }
                            for group,terms in e.observation_manager._group_obs_term_history_buffer.items():
                                for name,buffer in terms.items():
                                    snapshot['history'][group+'/'+name] = tensors(buffer)
                            torch.save(snapshot,out/'branch_snapshot.pt')
                        probe_bias = args.target_bias_rad if step >= args.probe_start and not failed[0].item() else 0.
                        if policy is None:""")
replace_once('                        q = robot.data.joint_pos[:,ids]', """                        # new change: log the actual biased target, not the unmodified actor target.
                        target[:, ankle_column] += probe_bias
                        q = robot.data.joint_pos[:,ids]""")
replace_once('                        for k in torch.where((terminated | truncated) & ~failed)[0].tolist():', """                        # new change: ensure the target written by the action term matches the log.
                        if not (terminated | truncated).any():
                            torch.testing.assert_close(probe_term.processed_actions, target, rtol=0., atol=1.e-6)
                        for k in torch.where((terminated | truncated) & ~failed)[0].tolist():""")
exec(compile(source, str(base), 'exec'), {'__name__':'__main__'})
