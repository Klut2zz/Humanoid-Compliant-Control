#new change
"""Strict input/prefix audit and descriptive temporal-probe outcomes."""
import argparse,json,ast
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--directory',required=True)
a=p.parse_args();root=Path(a.directory);base=Path(a.baseline)
z={'zero':dict(np.load(base.with_suffix('.npz')))}
for k in ('slow_plus','slow_minus','fast'):z[k]=dict(np.load(root/(k+'.npz')))
scope={'torch':torch,'np':np,'reporting_exceptions':[]}
tree=ast.parse(Path(__file__).with_name('analyze_dual540_release_ablation.py').read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='equal')
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),__file__,'exec'),scope)
equal=scope['equal'];snap=torch.load(base.with_suffix('.snapshot.pt'),map_location='cpu',weights_only=False)
b=z['zero'];fork=int(b['fork_step']);dt=float(b['dt']);selected=np.flatnonzero(b['selected']);norm=lambda v:np.linalg.norm(v,axis=-1)
for case in ('slow_plus','slow_minus','fast'):
    equal(snap,torch.load(root/(case+'.snapshot.pt'),map_location='cpu',weights_only=False))
    c=z[case];assert np.array_equal(c['selected'],b['selected'])
    for k in ('q','target','root','ref','force','root_vel','foot_error','com','foot_load','tilt'):
        assert np.array_equal(c[k][:fork],b[k][:fork],equal_nan=True),(case,k,'prefix')
    assert np.count_nonzero(c['residual'][fork+51:,selected])==0
    assert abs(c['residual'][fork:fork+51,selected]).max()<=.002001
assert np.array_equal(z['slow_plus']['residual'][fork:,selected],-z['slow_minus']['residual'][fork:,selected])
assert np.allclose(z['slow_plus']['residual'][fork:,selected]**2,z['fast']['residual'][fork:,selected]**2,rtol=0,atol=1e-12)
results={k:[] for k in z};rows=[]
for e in selected:
    trans=np.flatnonzero((b['event_index'][1:,e]>b['event_index'][:-1,e])&(b['event_time'][1:,e]<0)&b['alive'][1:,e])+1
    release=int(trans[trans>=fork][0]);wait=-float(b['event_time'][release,e]);end=release+int(np.floor(wait/dt))-1
    for case,c in z.items():
        common=b['alive'][fork:end+1,e]&c['alive'][fork:end+1,e]
        for k in ('force','ref','nominal_ref','body','ramp','hold','sampled_peak','accepted_peak'):
            assert np.array_equal(b[k][fork:end+1,e][common],c[k][fork:end+1,e][common],equal_nan=True),(case,e,k)
        failures=np.flatnonzero(c['done'][fork:end,e]);failed=bool(len(failures));stop=fork+int(failures[0]) if failed else end
        q=norm(c['root_error'][:,e]);speed=norm(c['root_vel'][:,e,:2]);foot=norm(c['foot_error'][:,e]).max(-1);tilt=c['tilt'][:,e]
        recovered=not failed and q[end]<.05 and speed[end]<.03 and foot[end]<.02 and tilt[end]<np.deg2rad(5)
        d=dict(env=int(e),failed=failed,recovered=bool(recovered),stop=stop,release=release,end=end,
            root_peak_mm=float(q[fork:stop+1].max()*1000),foot_peak_mm=float(foot[fork:stop+1].max()*1000),tilt_peak_deg=float(np.rad2deg(tilt[fork:stop+1]).max()))
        if not failed:
            tail=slice(max(release,end-24),end+1)
            d.update(root_end_mm=float(q[end]*1000),speed_tail_mps=float(speed[tail].mean()),foot_end_mm=float(foot[end]*1000),tilt_end_deg=float(np.rad2deg(tilt[end])))
        results[case].append(d)
summary=dict(pairing_pass=True,reporting_only_exceptions=scope['reporting_exceptions'],selected=len(selected),equal_pulse_energy=True,
    peak_limits_rad=dict(arms=.002,waist=.001,legs=.001),duration_s=1.,cases={})
for case,c in z.items():
    rs=results[case];baseline=results['zero'];common=[i for i in range(len(rs)) if not rs[i]['failed'] and not baseline[i]['failed']]
    d=dict(failed=sum(v['failed'] for v in rs),recovered=sum(v['recovered'] for v in rs),
        rescued=sum(x['failed'] and not y['failed'] for x,y in zip(baseline,rs)),
        newly_failed=sum(y['failed'] and not x['failed'] for x,y in zip(baseline,rs)),common_survivors=len(common))
    for key in ('root_end_mm','speed_tail_mps','foot_end_mm','tilt_end_deg'):
        delta=np.array([rs[i][key]-baseline[i][key] for i in common])
        d[key]=dict(mean=float(np.mean([rs[i][key] for i in common])),delta_mean=float(delta.mean()),delta_median=float(np.median(delta)),delta_p95_abs=float(np.percentile(abs(delta),95)))
    diffs=[]
    for i,e in enumerate(selected):
        stop=min(rs[i]['stop'],baseline[i]['stop'])
        diffs.append(float(norm(c['root_error'][fork:stop+1,e]-b['root_error'][fork:stop+1,e]).max()*1000))
    d['root_path_difference_mm']=dict(median=float(np.median(diffs)),p95=float(np.percentile(diffs,95)),max=float(max(diffs)))
    summary['cases'][case]=d
summary['oracle_not_policy_rescues']=sum(results['zero'][i]['failed'] and (not results['slow_plus'][i]['failed'] or not results['slow_minus'][i]['failed']) for i in range(len(selected)))
(root/'metrics.json').write_text(json.dumps(summary,indent=2))
(root/'event_outcomes.json').write_text(json.dumps(results,indent=2))
print(json.dumps(summary,indent=2))
