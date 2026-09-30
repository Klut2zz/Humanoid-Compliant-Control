#new change
"""Deterministic reconstructed-snapshot ablation, with an on/on replay control.

No partial PhysX state restore: each branch replays its complete identical prefix
in a fresh process. Compare snapshots, prefix and repeated on branch before any
causal interpretation. Only selected late-fall environments receive ablation.
"""
from pathlib import Path


def snapshot(env, term, wrapped):
    import torch,random,numpy as np
    def simple(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
        if isinstance(value,torch.Generator):return value.get_state().cpu().clone()
        if value is None or isinstance(value,(str,int,float,bool)):return value
        if isinstance(value,(list,tuple)):
            return [simple(v) for v in value]
        if isinstance(value,dict):return {str(k):simple(v) for k,v in value.items()}
        return type(value).__name__
    def fields(obj):
        return {k:simple(v) for k,v in vars(obj).items() if isinstance(v,(torch.Tensor,torch.Generator,list,tuple,dict,int,float,bool))}
    robot=env.scene['robot'];command=env.command_manager.get_term('upper_body_joints')
    histories={}
    for group,terms in env.observation_manager._group_obs_term_history_buffer.items():
        for name,buffer in terms.items():histories[group+'/'+name]=fields(buffer)
    return dict(root_state=robot.data.root_state_w.cpu().clone(),q=robot.data.joint_pos.cpu().clone(),
        dq=robot.data.joint_vel.cpu().clone(),target=robot.data.joint_pos_target.cpu().clone(),
        velocity_target=robot.data.joint_vel_target.cpu().clone(),effort_target=robot.data.joint_effort_target.cpu().clone(),
        obs={k:v.cpu().clone() for k,v in wrapped.get_observations().items()},env=fields(env),command=fields(command),
        action_manager=fields(env.action_manager),action=fields(term),base_action=fields(term.base),
        actuators={k:fields(v) for k,v in robot.actuators.items()},
        sensors={k:dict(state=fields(v),data=fields(v.data)) for k,v in env.scene.sensors.items()},
        observation_history=histories,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        numpy_rng=simple(np.random.get_state()),python_rng=random.getstate())


if __name__=='__main__':
    path=Path(__file__).with_name('test_dual540_force_checkpoint.py')
    source=path.read_text()
    changes={
        "p.add_argument('--force_max',type=int,choices=[0,4],default=4)":
        "p.add_argument('--force_max',type=int,choices=[0,4],default=4)\np.add_argument('--branch',choices=['on','off','repeat'],required=True)\np.add_argument('--fork_step',type=int,default=140)",
        'from dual540_force_train_cfg import TASK,Dual540ForceTrainCfg':
        'from dual540_recovery_window_cfg import TASK,RecoveryWindowCfg',
        'from dual540_force_model import runner_config':'from dual540_wholebody_force_cfg import runner_config',
        'cfg=Dual540ForceTrainCfg();cfg.scene.num_envs=args.num_envs':
        'assert args.force_max==4\n    cfg=RecoveryWindowCfg(zero_force_fraction=0.);cfg.scene.num_envs=args.num_envs',
        '        trace=[]':'        trace=[]\n        selected=torch.zeros(e.num_envs,device=e.device,dtype=torch.bool)',
        '                action=actor(wrapped.get_observations(),stochastic_output=False)':
        """                action=actor(wrapped.get_observations(),stochastic_output=False)
                if step==args.fork_step:
                    remaining=2*c._ramp_time+c._hold_time-c._event_time
                    selected=alive & (c.force_world.norm(dim=-1)>1e-4) & (c._event_time>=c._ramp_time+c._hold_time) & (remaining>.02) & (remaining<=.4)
                    assert int(selected.sum())>=16, 'Insufficient pre-release snapshots'
                    snap=snapshot(e,term,wrapped)
                    snap['selected']=selected.cpu().clone()
                    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
                    torch.save(snap,Path(args.output).with_suffix('.snapshot.pt'))
                    print('BRANCH_POINT',step,'selected',int(selected.sum()),flush=True)
                if step>=args.fork_step and args.branch=='off':
                    action=action.clone();action[selected]=0
""",
        'residual=.02*torch.tanh(action),force=c.force_world,':
        'residual=e._wb_limits*torch.tanh(action),force=c.force_world,',
        'joint_names=np.asarray(NAMES),force_max=np.asarray(args.force_max))':
        'joint_names=np.asarray(NAMES),force_max=np.asarray(args.force_max),selected=selected.cpu().numpy(),fork_step=np.asarray(args.fork_step),branch=np.asarray(args.branch))',
    }
    for old,new in changes.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path),'snapshot':snapshot})
