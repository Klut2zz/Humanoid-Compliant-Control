#new change
"""One frozen six-case acceptance matrix; all artifacts are numeric/log data.

Run initial matrix without --checkpoint. Run a trained checkpoint with
--baseline-dir pointing at that initial matrix. No learning is performed.
"""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
import numpy as np

def norm(x):return np.linalg.norm(x,axis=-1)
def summarize(z):
    valid=z['alive'];failed=(valid&z['done']).any(0);dt=float(z['dt'])
    root=norm(z['root_error']);feet=norm(z['foot_error']).max(-1)
    tilt=z['tilt'];speed=norm(z['root_vel'][...,:2]);root_peak=np.where(valid,root,0).max(0);foot_peak=np.where(valid,feet,0).max(0)
    wrist=norm(z['wrist']-z['wrist_ref'])*1000;mask=valid&(np.arange(len(valid))[:,None]>=50)
    support=(~failed)&(root_peak<.05)&(foot_peak<.02)&(np.where(valid,tilt,0).max(0)<np.deg2rad(5))
    out={'failed':int(failed.sum()),'survival':float((~failed).mean()),'support_qualified':float(support.mean()),
         'root_peak_p95_mm':float(np.percentile(root_peak,95)*1000),'foot_peak_p95_mm':float(np.percentile(foot_peak,95)*1000),
         'wrist_mae_mm':[float(wrist[...,i][mask].mean()) for i in (0,1)]}
    names=z['joint_names'].tolist()
    groups={'arms':[i for i,n in enumerate(names) if any(k in n for k in ('shoulder','elbow','wrist'))],
            'waist':[i for i,n in enumerate(names) if n.startswith('waist')],
            'legs':[i for i,n in enumerate(names) if any(k in n for k in ('hip','knee','ankle'))]}
    out['correction_rms_mrad']={g:float(np.sqrt(np.mean(z['correction'][...,ids][mask]**2))*1000) for g,ids in groups.items()}
    release=np.where((z['event_index'][1:]>z['event_index'][:-1])&(z['event_time'][1:]<0)&valid[1:]);rows=[];window=round(.5/dt)
    for t,e in zip(*release):
        t+=1
        # Actual force before release, not mere event scheduling in zero cohort.
        if norm(z['force'][max(0,t-100):t,e]).max()<.1:continue
        end=t+int(np.floor(-float(z['event_time'][t,e])/dt))-1
        if end>=len(valid):continue
        fail=bool(z['done'][t:end+1,e].any())
        good=(root[t:end+1,e]<.05)&(feet[t:end+1,e]<.02)&(tilt[t:end+1,e]<np.deg2rad(5))&(speed[t:end+1,e]<.03)&valid[t:end+1,e]
        recovered=not fail and len(good)>=window and bool(good[-window:].all())
        first=None
        if recovered:
            for k in range(len(good)-window+1):
                if good[k:k+window].all():first=(k+window)*dt;break
        rows.append({'env':int(e),'release_step':int(t),'end_step':int(end),'failed':fail,'recovered':recovered,'recovery_confirmed_s':first})
    out['recovery']={'eligible_events':len(rows),'failed':sum(x['failed'] for x in rows),'qualified':sum(x['recovered'] for x in rows),
       'qualified_fraction':float(np.mean([r['recovered'] for r in rows])) if rows else None,
       'median_confirmation_s':float(np.median([r['recovery_confirmed_s'] for r in rows if r['recovered']])) if any(r['recovered'] for r in rows) else None}
    return out,rows

def response(force,zero):
    # Wrist displacement relative to root, minus the matched zero-force run;
    # this removes whole-body translation from the compliance proxy.
    common=force['alive']&zero['alive']
    for key in ('event_index','event_time','ramp','hold','body'):
        np.testing.assert_array_equal(force[key][common],zero[key][common],err_msg='unpaired '+key)
    diff=(force['wrist']-force['root'][:,:,None,:])-(zero['wrist']-zero['root'][:,:,None,:])
    magnitude=norm(force['force']);direction=force['force']/np.maximum(magnitude[...,None],1e-8)
    values=[]
    for side,bid in enumerate(force['wrist_body_ids']):
        mask=force['alive']&zero['alive']&(force['body']==bid)&(magnitude>1.)
        projection=(diff[:,:,side]*direction).sum(-1)
        ref=((force['wrist_ref'][:,:,side]-force['wrist_nominal'][:,:,side])*direction).sum(-1)
        values.append({'samples':int(mask.sum()),'actual_mm':float(projection[mask].mean()*1000),
                       'reference_mm':float(ref[mask].mean()*1000)})
    return values

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--checkpoint');p.add_argument('--baseline-dir')
    p.add_argument('--analyze-only',action='store_true');args=p.parse_args();out=Path(args.output_dir).resolve()
    if args.checkpoint:args.checkpoint=str(Path(args.checkpoint).resolve())
    if not args.analyze_only:
        out.mkdir(parents=True,exist_ok=False)
        for seed in (0,7,123):
            for cohort in ('zero','force'):
                label=f'{cohort}_seed{seed}'
                cmd=[sys.executable,str(Path(__file__).with_name('train_stand_force_joint_v1.py')),'--mode','eval','--cohort',cohort,
                     '--num_envs','512','--steps','1500','--seed',str(seed),'--output',str(out/(label+'.npz')),'--headless']
                if args.checkpoint:cmd+=['--checkpoint',args.checkpoint]
                with (out/(label+'.log')).open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=Path(__file__).resolve().parents[1])
    result={'matrix':{},'force_response':{},'gates':{},'baseline_comparison':None,
        'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'promotion_rule':'All absolute safety gates and per-seed no failure increase; recovery qualification >= baseline-1 percentage point; root p95 <= baseline+5mm, feet p95 <= baseline+2mm, each wrist MAE <= baseline+2mm, positive force-response >=80% baseline. Additionally pooled force survival +2 percentage points OR mean seed root p95 reduction >=max(1mm,10%) OR mean wrist MAE reduction >=1mm. No additional budget automatically.',
        'metric_notes':{'feet':'ankle-link XY error versus fixed FK reference; not contact-point sliding',
        'response':'force-minus-zero wrist displacement relative to root translation; proxy, not measured stiffness or palm orientation'}}
    initial=json.loads((Path(args.baseline_dir)/'acceptance.json').read_text()) if args.baseline_dir else None
    if initial:
        assert Path(args.baseline_dir).resolve()!=out,'Keep the initial acceptance directory unchanged.'
        assert initial.get('evaluator_sha256')==result['evaluator_sha256'],'Re-analyze both saved datasets with the same evaluator version.'
    checkpoints=set()
    for seed in (0,7,123):
        loaded={c:dict(np.load(out/f'{c}_seed{seed}.npz')) for c in ('zero','force')}
        for c,z in loaded.items():
            assert z['alive'].shape==(1500,512) and int(z['seed'])==seed and str(z['cohort'])==c
            checkpoints.add(str(z['checkpoint']))
            label=f'{c}_seed{seed}';m,events=summarize(z);result['matrix'][label]=m
            (out/(label+'_events.json')).write_text(json.dumps(events,indent=2))
            if c=='zero':ok=m['survival']>=.99 and m['support_qualified']>=.99
            else:ok=m['survival']>=.95 and m['support_qualified']>=.95 and (m['recovery']['qualified_fraction'] or 0)>=.95
            result['gates'][label]=bool(ok)
        result['force_response'][str(seed)]=response(loaded['force'],loaded['zero'])
    assert len(checkpoints)==1,'Do not mix checkpoints inside one acceptance matrix.'
    result['checkpoint']=checkpoints.pop()
    if initial:
        checks={}
        for label,m in result['matrix'].items():
            old=initial['matrix'][label]
            checks[label+'_wrist']=all(x<=y+2 for x,y in zip(m['wrist_mae_mm'],old['wrist_mae_mm']))
            checks[label+'_failure']=m['failed']<=old['failed']
            checks[label+'_drift']=m['root_peak_p95_mm']<=old['root_peak_p95_mm']+5 and m['foot_peak_p95_mm']<=old['foot_peak_p95_mm']+2
            if label.startswith('force_'):
                checks[label+'_recovery']=(m['recovery']['qualified_fraction'] or 0)>=(old['recovery']['qualified_fraction'] or 0)-.01
        for seed,rows in result['force_response'].items():
            checks['force_response_'+seed]=all(r['actual_mm']>0 and old['actual_mm']>0 and r['actual_mm']>=.8*old['actual_mm'] for r,old in zip(rows,initial['force_response'][seed]))
        result['baseline_comparison']=checks
        result['safety_preservation_pass']=all(result['gates'].values()) and all(checks.values())
        labels=[f'force_seed{s}' for s in (0,7,123)]
        old=[initial['matrix'][k] for k in labels];new=[result['matrix'][k] for k in labels]
        failure_gain=float(np.mean([a['survival']-b['survival'] for a,b in zip(new,old)]))
        root_old=float(np.mean([a['root_peak_p95_mm'] for a in old]));root_new=float(np.mean([a['root_peak_p95_mm'] for a in new]))
        wrist_gain=float(np.mean([a['wrist_mae_mm'] for a in old])-np.mean([a['wrist_mae_mm'] for a in new]))
        result['population_gain']={'survival_fraction':failure_gain,'root_peak_p95_reduction_mm':root_old-root_new,'mean_wrist_error_reduction_mm':wrist_gain}
        result['clear_population_gain']=failure_gain>=.02 or (root_old-root_new>=max(1.,.1*root_old)) or wrist_gain>=1.
        result['accepted']=bool(result['safety_preservation_pass'] and result['clear_population_gain'])
    else:result['accepted']=None
    (out/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
