#new change
"""Require exact reconstruction/replay before interpreting residual ablation."""
import argparse,json
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--directory',required=True);args=p.parse_args();root=Path(args.directory)
z={k:dict(np.load(root/(k+'.npz'))) for k in ('on','repeat','off')}
#new change: this scatter-add accumulator is only consumed by the reporting
# method, never observations, rewards, IK, event scheduling or control.
reporting_exceptions=[]
def equal(a,b,path=''):
    if path=='/command/_actual_active_error_sum_by_slot':
        delta=float((a-b).abs().max())
        assert delta<1e-4,path
        reporting_exceptions.append(dict(path=path,max_difference=delta))
        return
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.shape==b.shape,path
        assert torch.allclose(a,b,rtol=0,atol=0,equal_nan=True),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
    else:assert a==b or (isinstance(a,float) and np.isnan(a) and np.isnan(b)),path
snap=torch.load(root/'on.snapshot.pt',map_location='cpu',weights_only=False)
for case in ('repeat','off'):equal(snap,torch.load(root/(case+'.snapshot.pt'),map_location='cpu',weights_only=False))
b=z['on'];a=z['off'];fork=int(b['fork_step']);dt=float(b['dt']);selected=np.flatnonzero(b['selected'])
def same(x,y):
    if x.dtype.kind in 'fc':return np.array_equal(x,y,equal_nan=True)
    return np.array_equal(x,y)
for k in b:
    if k=='branch':continue
    assert same(b[k],z['repeat'][k]),'repeat/'+k
    if b[k].ndim>=2 and b[k].shape[:2]==b['alive'].shape:
        assert same(b[k][:fork],a[k][:fork]),'prefix/'+k
assert same(a['selected'],b['selected'])
assert np.count_nonzero(a['residual'][fork:,selected])==0
assert np.max(abs((b['target'][fork,selected]-a['target'][fork,selected])-b['residual'][fork,selected]))<3e-7
norm=lambda x:np.linalg.norm(x,axis=-1)
events=[];results={case:[] for case in ('on','off')};common=[]
for e in selected:
    #new change: near-zero half-cosine force may precede actual event completion
    # by one frame. Anchor the window to the natural event transition instead.
    f=norm(b['force'][:,e]);trans=np.flatnonzero((b['event_index'][1:,e]>b['event_index'][:-1,e]) & (b['event_time'][1:,e]<0) & b['alive'][1:,e])+1
    trans=trans[trans>=fork]
    assert len(trans)>0
    release=int(trans[0]);wait=-float(b['event_time'][release,e]);assert 2.98<=wait<=5.01
    # Last recorded state before the next force event begins, not a reset state.
    end=release+int(np.floor(wait/dt))-1
    assert end<len(f)
    assert b['event_index'][release,e]>b['event_index'][release-1,e]
    events.append((int(e),release,end,wait))
    common_alive=a['alive'][fork:end+1,e]&b['alive'][fork:end+1,e]
    for k in ('force','ref','nominal_ref','body','ramp','hold','sampled_peak','stiffness','accepted_peak'):
        assert same(a[k][fork:end+1,e][common_alive],b[k][fork:end+1,e][common_alive]),f'input/{e}/{k}'
    pair={}
    for case,zcase in [('on',b),('off',a)]:
        failures=np.flatnonzero(zcase['done'][fork:end,e])
        failed=bool(len(failures));stop=fork+int(failures[0]) if failed else end
        valid=slice(fork,stop+1);q=zcase['root_error'][:,e];speed=norm(zcase['root_vel'][:,e,:2]);foot=norm(zcase['foot_error'][:,e]).max(-1)
        tilt=zcase['tilt'][:,e];recovered=(not failed and norm(q[end])<.05 and speed[end]<.03 and foot[end]<.02 and tilt[end]<np.deg2rad(5))
        values=dict(failed=failed,recovered=bool(recovered),stop=stop,
            root_peak_mm=float(norm(q[valid]).max()*1000),speed_peak_mps=float(speed[valid].max()),
            foot_peak_mm=float(foot[valid].max()*1000),tilt_peak_deg=float(np.rad2deg(tilt[valid]).max()))
        if not failed:
            tail=slice(max(release,end-24),end+1)
            values.update(root_end_mm=float(norm(q[end])*1000),speed_end_mps=float(speed[end]),
                foot_end_mm=float(foot[end]*1000),tilt_end_deg=float(np.rad2deg(tilt[end])),
                speed_tail_mps=float(speed[tail].mean()))
        results[case].append(values);pair[case]=values
    stop=min(pair['on']['stop'],pair['off']['stop']);s=slice(fork,stop+1)
    common.append(dict(root_max_difference_mm=float(norm(b['root_error'][s,e]-a['root_error'][s,e]).max()*1000),
        root_speed_max_difference_mps=float(norm(b['root_vel'][s,e,:2]-a['root_vel'][s,e,:2]).max()),
        foot_max_difference_mm=float(norm(b['foot_error'][s,e]-a['foot_error'][s,e]).max()*1000)))
summary=dict(pairing_pass=True,control_snapshot_equal=True,reporting_only_exceptions=reporting_exceptions,prefix_equal=True,on_repeat_exact=True,
    external_force_and_reference_equal_until_failure=True,selected_environments=len(selected),fork_step=fork,
    wait_range_s=[min(v[3] for v in events),max(v[3] for v in events)],cases={})
for case,rows in results.items():
    summary['cases'][case]=dict(failed=sum(r['failed'] for r in rows),recovered=sum(r['recovered'] for r in rows),
        root_peak_max_mm=max(r['root_peak_mm'] for r in rows),foot_peak_max_mm=max(r['foot_peak_mm'] for r in rows))
both=[i for i in range(len(selected)) if not results['on'][i]['failed'] and not results['off'][i]['failed']]
summary['common_survivors']=len(both);summary['paired_common_survivors']={}
for k in ('root_end_mm','speed_end_mps','speed_tail_mps','foot_end_mm','tilt_end_deg'):
    x=np.array([results['on'][i][k] for i in both]);y=np.array([results['off'][i][k] for i in both]);delta=x-y
    summary['paired_common_survivors'][k]=dict(on_mean=float(x.mean()),off_mean=float(y.mean()),
        on_minus_off_mean=float(delta.mean()),median=float(np.median(delta)),max_abs=float(abs(delta).max()))
summary['failure_overlap']=dict(both=sum(x['failed'] and y['failed'] for x,y in zip(results['on'],results['off'])),
    on_only=sum(x['failed'] and not y['failed'] for x,y in zip(results['on'],results['off'])),
    off_only=sum(y['failed'] and not x['failed'] for x,y in zip(results['on'],results['off'])))
summary['max_common_path_difference']={k:max(row[k] for row in common) for k in common[0]}
print(json.dumps(summary,indent=2))
(root/'pairing_metrics.json').write_text(json.dumps(summary,indent=2))
