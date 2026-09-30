#new change
"""Held-out-seed critic evaluation on identical frozen-policy trajectories."""
from pathlib import Path


def evaluate(data):
    import copy,json,hashlib
    import numpy as np
    import torch
    args=data['args'];env=data['env'];critic=data['critic'];actor=data['actor']
    digest=data['digest'];frozen=data['frozen'];out=data['out']
    ckpt=Path(args.calibrated_critic)
    candidate=copy.deepcopy(critic).eval().requires_grad_(False)
    saved=torch.load(ckpt,map_location=env.device,weights_only=False)
    assert saved['artifact_type']=='critic_only_not_ppo_resume'
    assert Path(saved['source_checkpoint']).resolve()==Path(args.init_checkpoint).resolve()
    candidate.load_state_dict(saved['critic_state_dict'],strict=True)
    candidate_hash=digest(candidate)
    x=data['x'];v=data['v'].double().numpy();r=data['rew'].double().numpy();d=data['done'].numpy()
    p=data['phase'].numpy();target=data['target'].numpy();health=torch.stack(data['health']).numpy()
    with torch.no_grad():
        newv=torch.cat([candidate.mlp(c.to(env.device)).squeeze(-1).cpu() for c in x.flatten(0,1).split(8192)]).reshape(1000,env.num_envs)
        tail=candidate(data['obs']).squeeze(-1).cpu()[None]
        newv=torch.cat((newv,tail)).double().numpy()
    def metrics(a,b):
        return dict(n=int(a.size),rmse=float(np.sqrt(np.mean((a-b)**2))),bias=float(np.mean(a-b)),
            explained_variance=float(1-np.var(a-b)/np.var(b)),prediction_mean=float(a.mean()),target_mean=float(b.mean()))
    def gae(value):
        raw=np.zeros_like(r);normalized=np.zeros_like(r)
        # Match production 48-step GAE, normalizing over each whole rollout.
        for start in range(0,len(r),48):
            end=min(start+48,len(r));adv=np.zeros(env.num_envs)
            for t in range(end-1,start-1,-1):
                live=~d[t]
                delta=r[t]+.99*value[t+1]*live-value[t]
                adv=delta+.99*.95*live*adv
                raw[t]=adv
            b=raw[start:end];normalized[start:end]=(b-b.mean())/(b.std(ddof=1)+1e-8)
        return raw,normalized
    raw0,adv0=gae(v);raw1,adv1=gae(newv)
    stages={}
    for label,name in enumerate(('quiet','rise','hold','fall','recovery')):
        m=p[:500]==label
        stages[name]={'before':metrics(v[:500][m],target[m]),'after':metrics(newv[:500][m],target[m])}
    cohorts={}
    for name,envmask in [('zero_force',np.arange(env.num_envs)%4==0),('low_force',np.arange(env.num_envs)%4!=0)]:
        cohorts[name]={'before':metrics(v[:500,envmask],target[:,envmask]),'after':metrics(newv[:500,envmask],target[:,envmask])}
    events=[]
    for t,e in zip(*np.where((p[1:]==4)&(p[:-1]!=4)&(~d[:-1]))):
        t+=1
        if t>=500:continue
        failure=np.flatnonzero(d[t:t+150,e]);failed=bool(len(failure))
        end=t+int(failure[0])+1 if failed else t+150
        if np.isin(p[t:end,e],[1,2,3]).any():continue
        q=health[t+150,e]
        recovered=not failed and q[0]<.05 and q[1]<.03 and q[2]<np.deg2rad(5) and q[3]<.02
        events.append((int(t),int(e),int(failed),int(recovered)))
    events=np.asarray(events,dtype=np.int64)
    def auc(a,y):
        y=np.asarray(y,bool)
        if not y.any() or y.all():return None
        u=a[y,None];w=a[~y][None,:]
        return float(np.mean((u>w)+.5*(u==w)))
    results={}
    ti,ei,failed,recovered=events.T
    for name,value,raw,adv in [('before',v,raw0,adv0),('after',newv,raw1,adv1)]:
        vv=value[ti,ei];aa=adv[ti,ei]
        results[name]=dict(value_survival_auc=auc(vv,~failed.astype(bool)),value_recovery_auc=auc(vv,recovered),
            advantage_survival_auc=auc(aa,~failed.astype(bool)),advantage_recovery_auc=auc(aa,recovered))
        for group,m in [('failed',failed.astype(bool)),('recovered',recovered.astype(bool))]:
            results[name][group]=dict(n=int(m.sum()),value_mean=float(vv[m].mean()),
                raw_advantage_mean=float(raw[ti[m],ei[m]].mean()),advantage_mean=float(aa[m].mean()),
                positive_advantage_fraction=float(np.mean(aa[m]>0)))
        # Adjust release comparisons for rollout block and offset bin.
        differences=[];weights=[]
        for block in range(11):
            for binid in range(4):
                m=(ti//48==block)&((ti%48)//12==binid)
                good=m&recovered.astype(bool);bad=m&failed.astype(bool)
                if good.any() and bad.any():
                    differences.append(aa[good].mean()-aa[bad].mean());weights.append(min(good.sum(),bad.sum()))
        results[name]['stratified_recovered_minus_failed_advantage']=float(np.average(differences,weights=weights))
    summary=dict(seed=args.seed,environments=env.num_envs,steps=1000,zero_force_fraction=args.zero_force_fraction,
        source_checkpoint=args.init_checkpoint,calibrated_critic=str(ckpt),calibrated_file_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        before=metrics(v[:500],target),after=metrics(newv[:500],target),stages=stages,cohorts=cohorts,
        events=len(events),failed_events=int(failed.sum()),recovered_events=int(recovered.sum()),outcomes=results,
        all_networks_unchanged=True,optimizer_updates=0,fit_on_this_seed=False,
        target='500-step stopped discounted rewards plus identical frozen original-critic tail',
        advantage='production 48-step GAE gamma .99 lambda .95; first 500 release starts only')
    assert data['before']==[digest(m) for m in (actor,*frozen)]
    assert data['old_critic']==digest(critic)
    assert candidate_hash==digest(candidate)
    out.mkdir(parents=True,exist_ok=False)
    (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(out/'evaluation.npz',values_before=v,values_after=newv,rewards=r,done=d,phase=p,
        health=health,target=target,raw_adv_before=raw0,raw_adv_after=raw1,adv_before=adv0,adv_after=adv1,
        release_events=events,dt=np.asarray(env.step_dt))
    print('HELDOUT_CRITIC_PASS',json.dumps(summary),flush=True)


if __name__=='__main__':
    # Reuse the audited collector verbatim; stop before every fitting operation.
    import inspect
    from calibrate_dual540_critic_only import calibrate
    collector=inspect.getsource(calibrate)
    collector=collector[:collector.index('    train_env=')]+"    evaluate(locals())\n"
    scope={'Path':Path,'evaluate':evaluate}
    exec(compile(collector,__file__,'exec'),scope)
    path=Path(__file__).with_name('train_dual540_recovery_window.py')
    wrapper=path.read_text();ns={'__file__':str(path)}
    exec(compile(wrapper[:wrapper.rindex('exec(compile(')],str(path),'exec'),ns)
    source=ns['source']
    old="p.add_argument('--output',default=None)"
    assert source.count(old)==1
    source=source.replace(old,old+"\np.add_argument('--calibrated_critic',required=True)")
    start=source.index("        if args.mode=='gradient':")
    end=source.index('        assert args.output and not Path(args.output).exists()',start)
    source=source[:start]+"        if args.mode=='gradient':\n            calibrate(runner,wrapped,e,term,args,digest,frozen)\n            return\n"+source[end:]
    exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path),'calibrate':scope['calibrate']})
