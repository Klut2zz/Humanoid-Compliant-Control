#new change
"""Antithetic smooth random residual probes, independent of trained outputs."""
from pathlib import Path
import ast
from test_dual540_residual_release_branch import snapshot

path=Path(__file__).with_name('test_dual540_residual_release_branch.py')
original=path.read_text()
#new change: AST extraction preserves multiline literal indentation.
main_block=ast.parse(original).body[-1]
body=ast.Module(body=main_block.body[:-1],type_ignores=[])
ns={'__file__':str(path),'Path':Path}
exec(compile(ast.fix_missing_locations(body),str(path),'exec'),ns)
source=ns['source']
changes={
    "p.add_argument('--branch',choices=['on','off','repeat'],required=True)":
    "p.add_argument('--branch',choices=['slow_plus','slow_minus','fast'],required=True)",
    '        selected=torch.zeros(e.num_envs,device=e.device,dtype=torch.bool)':
    """        selected=torch.zeros(e.num_envs,device=e.device,dtype=torch.bool)
        # Local CPU RNG does not disturb simulator/event RNG states.
        generator=np.random.default_rng(20260924)
        direction=torch.as_tensor(generator.uniform(-1,1,(e.num_envs,29)),device=e.device,dtype=torch.float32)
        arm=[i for i,n in enumerate(NAMES) if any(k in n for k in ('shoulder','elbow','wrist'))]
        waist=[i for i,n in enumerate(NAMES) if 'waist' in n]
        leg=[i for i in range(29) if i not in arm+waist]
        for ids in (arm,waist,leg):direction[:,ids]/=direction[:,ids].abs().amax(-1,keepdim=True)
        peak=direction*e._wb_limits*.1
        assert torch.all(peak.abs()<=e._wb_limits*.10001)
""",
    "                if step>=args.fork_step and args.branch=='off':\n                    action=action.clone();action[selected]=0":
    """                if step>=args.fork_step:
                    # One second sin^2 pulse: value and slope zero at both ends.
                    k=step-args.fork_step
                    envelope=float(np.sin(np.pi*k/50.)**2) if k<=50 else 0.
                    sign=-1. if args.branch=='slow_minus' else 1.
                    if args.branch=='fast':sign=1. if k%2==0 else -1.
                    desired=peak*envelope*sign
                    action=action.clone()
                    action[selected]=torch.atanh((desired/e._wb_limits).clamp(-.999,.999))[selected]
""",
    'branch=np.asarray(args.branch))':
    'branch=np.asarray(args.branch),probe_peak=peak.cpu().numpy(),pulse_steps=np.asarray(50))',
}
for old,new in changes.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path),'snapshot':snapshot})
