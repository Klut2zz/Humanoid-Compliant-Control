#new change
"""Standalone value regression; no actor/PPO optimization or baseline overwrite."""
from pathlib import Path


def calibrate(runner, wrapped, env, term, args, digest, frozen):
    import copy
    import json
    import numpy as np
    import torch
    out=Path(args.output)
    assert not out.exists(),out
    actor,critic=runner.alg.actor,runner.alg.critic
    actor.eval().requires_grad_(False)
    before=[digest(m) for m in (actor,*frozen)]
    old_critic=digest(critic)
    def forbidden(*a,**kw):raise RuntimeError('PPO optimizer forbidden')
    runner.alg.optimizer.step=forbidden
    xs=[];rs=[];ds=[];vs=[];ph=[];health=[]
    last=torch.zeros(env.num_envs,device=env.device)
    age=torch.full_like(last,100.)
    obs=wrapped.get_observations()
    from neutral_stand_geometry_cfg import state
    with torch.no_grad():
        for step in range(1000):
            r,c,ids,fids,feet,fe,com,forces=state(env)
            force=c.force_world.norm(dim=-1)
            age[(last>1e-4)&(force<=1e-4)]=0
            phase=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
            active=force>1e-4;t=c._event_time
            phase[active & (t<c._ramp_time)]=1
            phase[active & (t>=c._ramp_time)&(t<c._ramp_time+c._hold_time)]=2
            phase[active & (t>=c._ramp_time+c._hold_time)]=3
            phase[(~active)&(age<3)]=4
            xs.append(critic.get_latent(obs).cpu().clone())
            vs.append(critic(obs).squeeze(-1).cpu().clone());ph.append(phase.cpu())
            health.append(torch.stack(((r.data.root_pos_w[:,:2]-c.root_state['root_pos'][:,1,:2]).norm(dim=-1),
                r.data.root_lin_vel_w[:,:2].norm(dim=-1),
                torch.acos((-r.data.projected_gravity_b[:,2]).clamp(-1,1)),fe.norm(dim=-1).amax(-1)),dim=-1).cpu())
            last.copy_(force)
            obs,reward,done,extra=wrapped.step(actor(obs,stochastic_output=True))
            assert not extra.get('time_outs',torch.zeros_like(done)).bool().any(), 'Handle timeout bootstrap explicitly before proceeding'
            rs.append(reward.cpu().clone());ds.append(done.cpu().bool().clone())
            age+=env.step_dt;age[active|done.bool()]=100.;last[done.bool()]=0
            if (step+1)%100==0:print('CRITIC_DATA',step+1,flush=True)
        vs.append(critic(obs).squeeze(-1).cpu().clone())
    assert before==[digest(m) for m in (actor,*frozen)]
    x=torch.stack(xs);rew=torch.stack(rs);done=torch.stack(ds);v=torch.stack(vs);phase=torch.stack(ph)
    # Use first 500 starts with a full 500-step future. True terminals stop the
    # sum; right-censored windows retain a small, fixed baseline-value tail.
    n=500;horizon=500;gamma=.99
    target=torch.zeros((n,env.num_envs),dtype=torch.float64)
    live=torch.ones_like(target,dtype=torch.bool)
    for k in range(horizon):
        target+=(gamma**k)*rew[k:k+n].double()*live
        live &= ~done[k:k+n]
    finite=target.clone()
    target+=(gamma**horizon)*v[horizon:horizon+n].double()*live
    train_env=torch.arange(env.num_envs)%5!=0
    val_env=~train_env
    xtrain=x[:n,train_env].reshape(-1,x.shape[-1]).to(env.device)
    xval=x[:n,val_env].reshape(-1,x.shape[-1]).to(env.device)
    ytrain=target[:,train_env].reshape(-1).float().to(env.device)
    yval=target[:,val_env].reshape(-1).float().to(env.device)
    pval=phase[:n,val_env].reshape(-1).numpy()
    def predict(inp):
        with torch.no_grad():return torch.cat([critic.mlp(c).squeeze(-1) for c in inp.split(8192)])
    def metrics(pred,y):
        a=pred.detach().cpu().double().numpy();b=y.detach().cpu().double().numpy()
        return dict(rmse=float(np.sqrt(np.mean((a-b)**2))),bias=float(np.mean(a-b)),
            explained_variance=float(1-np.var(a-b)/np.var(b)),prediction_mean=float(a.mean()),target_mean=float(b.mean()))
    baseline=predict(xval);baseline_metrics=metrics(baseline,yval)
    print('CRITIC_BASELINE_VALIDATION',json.dumps(baseline_metrics),flush=True)
    opt=torch.optim.Adam(critic.parameters(),lr=3e-4)
    frozen_ids={id(p) for m in (actor,*frozen) for p in m.parameters()}
    assert all(id(p) not in frozen_ids for g in opt.param_groups for p in g['params'])
    best=baseline_metrics['rmse'];best_state=copy.deepcopy(critic.state_dict());best_epoch=0;history=[]
    for epoch in range(1,21):
        critic.train()
        for ix in torch.randperm(len(xtrain),device=env.device).split(4096):
            pred=critic.mlp(xtrain[ix]).squeeze(-1)
            loss=(pred-ytrain[ix]).square().mean()
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward()
            assert all(p.grad is None for m in (actor,*frozen) for p in m.parameters())
            torch.nn.utils.clip_grad_norm_(critic.parameters(),10.)
            opt.step()
        critic.eval();m=metrics(predict(xval),yval);m['epoch']=epoch;history.append(m)
        if m['rmse']<best:best=m['rmse'];best_epoch=epoch;best_state=copy.deepcopy(critic.state_dict())
        print('CRITIC_EPOCH',epoch,json.dumps(m),flush=True)
    critic.load_state_dict(best_state)
    after=predict(xval);stage_metrics={}
    for label,name in enumerate(('quiet','rise','hold','fall','recovery')):
        mask=torch.as_tensor(pval==label,device=env.device)
        if mask.any():stage_metrics[name]={'before':metrics(baseline[mask],yval[mask]),'after':metrics(after[mask],yval[mask])}
    assert before==[digest(m) for m in (actor,*frozen)]
    out.mkdir(parents=True,exist_ok=False)
    # A critic-only artifact intentionally omits actor and PPO optimizer keys.
    torch.save(dict(critic_state_dict={k:v.cpu() for k,v in best_state.items()},
        source_checkpoint=args.init_checkpoint,artifact_type='critic_only_not_ppo_resume',
        epoch=best_epoch),out/'critic_only.pt')
    np.savez_compressed(out/'validation.npz',before=baseline.cpu().numpy().reshape(n,-1),
        after=after.cpu().numpy().reshape(n,-1),target=yval.cpu().numpy().reshape(n,-1),
        phase=phase[:,val_env].numpy(),done=done[:,val_env].numpy(),health=torch.stack(health)[:,val_env].numpy(),
        env_ids=torch.arange(env.num_envs)[val_env].numpy(),dt=np.asarray(env.step_dt))
    np.savez_compressed(out/'calibration_samples.npz',observations=x.numpy(),rewards=rew.numpy(),
        dones=done.numpy(),baseline_values=v.numpy(),phase=phase.numpy(),target=target.numpy(),
        truncated_return=finite.numpy(),train_env=train_env.numpy(),health=torch.stack(health).numpy())
    summary=dict(source_checkpoint=args.init_checkpoint,seed=args.seed,environments=env.num_envs,
        sampled_steps=1000,training_environments=int(train_env.sum()),validation_environments=int(val_env.sum()),
        zero_force_fraction=args.zero_force_fraction,stochastic_frozen_actor=True,
        target_horizon_steps=horizon,tail_discount=gamma**horizon,critic_learning_rate=3e-4,
        baseline=baseline_metrics,best=metrics(after,yval),best_epoch=best_epoch,
        stages=stage_metrics,history=history,actor_and_bases_unchanged=True,ppo_updates=0,
        original_critic_hash=old_critic,calibrated_critic_hash=digest(critic))
    (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    print('CRITIC_ONLY_PASS',json.dumps(summary['best']),'frozen_hashes_unchanged=True ppo_updates=0',flush=True)


if __name__=='__main__':
    path=Path(__file__).with_name('train_dual540_recovery_window.py')
    wrapper=path.read_text();ns={'__file__':str(path)}
    exec(compile(wrapper[:wrapper.rindex('exec(compile(')],str(path),'exec'),ns)
    source=ns['source'];start=source.index("        if args.mode=='gradient':")
    end=source.index('        assert args.output and not Path(args.output).exists()',start)
    source=source[:start]+"        if args.mode=='gradient':\n            calibrate(runner,wrapped,e,term,args,digest,frozen)\n            return\n"+source[end:]
    exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path),'calibrate':calibrate})
