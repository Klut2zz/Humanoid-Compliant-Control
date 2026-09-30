#new change
"""Frozen state-feedback experiment, identical-prefix reconstructed branches.

Adds only bounded hip/ankle pitch targets; retains model84 everywhere else.
The added signal is NOT a scaled policy output or a learned controller.
"""
from pathlib import Path
import ast
from test_dual540_residual_release_branch import snapshot


def build_source():
    path=Path(__file__).with_name('test_dual540_residual_release_branch.py')
    tree=ast.parse(path.read_text())
    ns={'__file__':str(path),'Path':Path}
    exec(compile(ast.fix_missing_locations(ast.Module(body=tree.body[-1].body[:-1],type_ignores=[])),str(path),'exec'),ns)
    source=ns['source']
    start=source.index('                if step==args.fork_step:')
    end=source.index('                assert torch.isfinite(action).all()',start)
    source=source[:start]+'''                feedback.zero_()
                load=forces[:,:,2].clamp(min=0)
                weights=load/load.sum(-1,keepdim=True).clamp(min=1.)
                support=(feet*weights[:,:,None]).sum(1)
                com_support=com-support
                if step==args.fork_step:
                    selected=alive.clone()
                    assert bool(selected.all()), 'Fork must precede all failures'
                    anchor.copy_(com_support)
                    # Horizontal heading frozen at the fork, not a future signal.
                    w,x,y,z=r.data.root_quat_w.unbind(-1)
                    forward[:,0]=1-2*(y*y+z*z);forward[:,1]=2*(x*y+w*z)
                    forward[:,:2]/=forward[:,:2].norm(dim=-1,keepdim=True)
                    snap=snapshot(e,term,wrapped);snap['selected']=selected.cpu().clone()
                    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
                    torch.save(snap,Path(args.output).with_suffix('.snapshot.pt'))
                    # Geometry sanity check, fixed root orientation: hip+ / ankle-
                    # must move feet backwards, allowing a forward root translation.
                    jac=r.root_physx_view.get_jacobians()
                    assert not r.is_fixed_base
                    assert jac.shape[1]==len(r.body_names) and jac.shape[-1]==r.num_joints+6
                    direction=torch.zeros(r.num_joints+6,device=e.device)
                    for hip,ankle in pairs:
                        direction[ids[hip]+6]=1.;direction[ids[ankle]+6]=-1.
                    foot_delta=torch.einsum('efij,j->efi',jac[:,fids,:3,:],direction)
                    slope=-(foot_delta*forward[:,None]).sum(-1)
                    assert bool((slope>.1).all()), 'Unexpected hip/ankle kinematic direction'
                    kinematic_slope=slope.cpu().numpy()
                    print('SAGITTAL_FORK_PASS',step,'count',int(selected.sum()),'forward_root_slope_m_per_rad',float(slope.mean()),flush=True)
                if step>=args.fork_step:
                    root_delta=r.data.root_pos_w-c.root_state['root_pos'][:,1]
                    ex=(root_delta*forward).sum(-1)
                    vx=(r.data.root_lin_vel_w*forward).sum(-1)
                    cx=((com_support-anchor)*forward).sum(-1)
                    demand=-(.08*ex+.025*vx+.04*cx)
                    supported=(load.min(-1).values>10.)&(load.sum(-1)>100.)
                    elapsed=(step-args.fork_step)*e.step_dt
                    blend=.5-.5*np.cos(np.pi*min(elapsed/.25,1.))
                    wanted=(demand[:,None]*2*weights).clamp(-.003,.003)*supported[:,None]*blend
                    smooth.add_((wanted-smooth)*(1.-np.exp(-e.step_dt/.15)))
                    smooth[~alive]=0
                    for side,(hip,ankle) in enumerate(pairs):
                        feedback[:,hip]=smooth[:,side];feedback[:,ankle]=-smooth[:,side]
                    #new change: at the fork the envelope is exactly zero;
                    # avoid even the floating-point tanh/atanh round trip.
                    if args.branch=='feedback' and step>args.fork_step:
                        baseline_residual=torch.tanh(action)*e._wb_limits
                        desired=baseline_residual+feedback
                        assert bool((desired.abs()<e._wb_limits).all())
                        action=action.clone()
                        action[:,controlled]=torch.atanh(desired[:,controlled]/e._wb_limits[controlled])
                    else:feedback.zero_()
''' +source[end:]
    changes={
        "choices=['on','off','repeat']":"choices=['baseline','repeat','feedback']",
        "default=140":"default=100",
        '        trace=[]':'''        trace=[]
        pairs=[(NAMES.index(side+'_hip_pitch_joint'),NAMES.index(side+'_ankle_pitch_joint')) for side in ('left','right')]
        controlled=[i for pair in pairs for i in pair]
        feedback=torch.zeros((e.num_envs,29),device=e.device)
        smooth=torch.zeros((e.num_envs,2),device=e.device)
        anchor=torch.zeros((e.num_envs,3),device=e.device)
        forward=torch.zeros_like(anchor)
        kinematic_slope=None''',
        'fields=dict(alive=alive,':'''fields=dict(feedback=feedback,root_quat=r.data.root_quat_w,
                    support_center=support-e.scene.env_origins,com_support=com_support,
                    feet=feet-e.scene.env_origins[:,None],ref_vel=c.dof_vel[:,1],
                    foot_velocity=r.data.body_lin_vel_w[:,fids],
                    joint_velocity=r.data.joint_vel[:,ids],alive=alive,''',
        'branch=np.asarray(args.branch))':"branch=np.asarray(args.branch),kinematic_slope=kinematic_slope)",
    }
    for old,new in changes.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    return source,path


if __name__=='__main__':
    source,path=build_source()
    exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path),'snapshot':snapshot})
