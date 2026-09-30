#new change
"""Paired withdrawal diagnostics; no model mutation or formal report."""
import argparse,ast,json
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--directory',required=True);args=p.parse_args()
root=Path(args.directory);norm=lambda x:np.linalg.norm(x,axis=-1)
ns={'torch':torch,'np':np,'reporting_exceptions':[]}
tree=ast.parse(Path(__file__).with_name('analyze_dual540_release_ablation.py').read_text())
fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='equal')
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'snapshot_check','exec'),ns)
data={case:dict(np.load(root/(case+'.npz'))) for case in ('continue','fade')}
a=data['continue'];b=data['fade'];fork=int(a['fade_step']);dt=float(a['dt']);shape=a['alive'].shape
ns['equal'](torch.load(root/'continue.withdrawal_snapshot.pt',map_location='cpu',weights_only=False),
            torch.load(root/'fade.withdrawal_snapshot.pt',map_location='cpu',weights_only=False))
for key in a:
    if a[key].ndim>=2 and a[key].shape[:2]==shape:
        assert np.array_equal(a[key][:fork+1],b[key][:fork+1]),'prefix/'+key
previous=dict(np.load(root.parent/'sagittal_feedback_seed0/feedback.npz'))
for key,value in previous.items():
    if key!='branch':assert np.array_equal(value,a[key]),'continued_replay/'+key
assert np.array_equal(a['fade_selected'],b['fade_selected'])
assert abs(b['feedback']).max()<=.003001
phase=np.minimum(np.arange(shape[0]-fork)*dt/float(b['fade_duration']),1.)
envelope=np.where(phase<1,.5*(1+np.cos(np.pi*phase)),0.)
expected=b['feedback'][fork][None]*envelope[:,None,None]*b['alive'][fork:,:,None]
assert np.max(abs(b['feedback'][fork:]-expected))<1e-9
assert np.count_nonzero(b['feedback'][fork+int(np.ceil(float(b['fade_duration'])/dt)):])==0
assert np.max(abs((b['target'][fork+1]-a['target'][fork+1])-(b['feedback'][fork+1]-a['feedback'][fork+1])))<5e-7
out={'snapshot_equal':True,'prefix_through_step150_equal':True,'continue_matches_previous_feedback':True,
     'reporting_only_exceptions':ns['reporting_exceptions'],'steps':shape[0],'num_envs':shape[1],
     'fade_selected':int(a['fade_selected'].sum()),'fade_duration_s':float(b['fade_duration']),
     'added_feedback_zero_from_step':fork+int(np.ceil(float(b['fade_duration'])/dt)),
     'cases':{},'target_envs':{}}
failed={}
for case,z in data.items():
    d=z['done']&z['alive'];failed[case]=d.any(0)
    out['cases'][case]={'failed_total':int(d.any(0).sum()),'failed_before_branch':int(d[:fork].any(0).sum()),
                      'failed_among_branch_survivors':int((d.any(0)&a['fade_selected']).sum())}
fa=failed['continue'];fb=failed['fade']
out['failure_overlap']={'both':int((fa&fb).sum()),'continue_only':int((fa&~fb).sum()),'fade_only':int((fb&~fa).sum())}
inputs=('force','ref','nominal_ref','sampled_peak','accepted_peak','stiffness','body','ramp','hold','event_index','event_time')
input_diff={k:0. for k in inputs};events=[];records={case:[] for case in data};no_release=[]
for e in np.flatnonzero(a['fade_selected']):
    release_ids=np.flatnonzero((a['event_index'][1:,e]>a['event_index'][:-1,e])&(a['event_time'][1:,e]<0)&a['alive'][1:,e])+1
    if not len(release_ids):no_release.append(int(e));continue
    release=int(release_ids[0]);wait=-float(a['event_time'][release,e]);end=release+int(np.floor(wait/dt))-1
    assert end>=fork and end<shape[0]
    common=a['alive'][fork:end+1,e]&b['alive'][fork:end+1,e]
    for key in inputs:
        delta=float(np.max(abs(a[key][fork:end+1,e][common].astype(float)-b[key][fork:end+1,e][common].astype(float))))
        input_diff[key]=max(input_diff[key],delta)
    events.append({'env':int(e),'release_step':release,'end_step':end,'wait_s':wait})
    for case,z in data.items():
        failures=np.flatnonzero(z['done'][fork:end,e]);fail=bool(len(failures));stop=fork+int(failures[0]) if fail else end
        pos=norm(z['root_error'][:,e]);speed=norm(z['root_vel'][:,e,:2]);feet=norm(z['foot_error'][:,e]).max(-1)
        row={'env':int(e),'failed':fail,'recovered':bool(not fail and pos[end]<.05 and speed[end]<.03 and feet[end]<.02 and z['tilt'][end,e]<np.deg2rad(5)),
             'failure_time_s':(stop+1)*dt if fail else None,'root_last_mm':float(pos[stop]*1000),
             'root_peak_mm':float(pos[fork:stop+1].max()*1000),'tilt_peak_deg':float(np.rad2deg(z['tilt'][fork:stop+1,e]).max())}
        if not fail:
            row.update(root_end_mm=float(pos[end]*1000),speed_tail_mps=float(speed[max(release,end-24):end+1].mean()),
                com_velocity_end_mps=((z['com'][end,e]-z['com'][end-10,e])/(10*dt)).tolist(),
                root_position_velocity_end_mps=((z['root'][end,e]-z['root'][end-10,e])/(10*dt)).tolist(),
                foot_end_mm=float(feet[end]*1000),tilt_end_deg=float(np.rad2deg(z['tilt'][end,e])))
        records[case].append(row)
        if e in (212,2,292):
            overall=np.flatnonzero(z['done'][:,e]&z['alive'][:,e])
            out['target_envs'].setdefault(str(e),{'release_step':release,'quiet_end_step':end})[case]={**row,
                'overall_failure_time_s':float((overall[0]+1)*dt) if len(overall) else None}
out['paired_input_maxdiff']=input_diff
assert all(v==0 for v in input_diff.values()),'First recovery window inputs differ'
out['complete_first_recovery_windows']=len(events);out['continue_failed_before_release_selected']=no_release
for case,rows in records.items():out['cases'][case]['quiet_window']={'failed':sum(r['failed'] for r in rows),'recovered':sum(r['recovered'] for r in rows)}
both=[i for i in range(len(events)) if not records['continue'][i]['failed'] and not records['fade'][i]['failed']]
out['common_window_survivors']=len(both);out['paired_recovery_end']={}
for key in ('root_end_mm','speed_tail_mps','foot_end_mm','tilt_end_deg'):
    x=np.array([records['continue'][i][key] for i in both]);y=np.array([records['fade'][i][key] for i in both])
    out['paired_recovery_end'][key]={'continue':float(x.mean()),'fade':float(y.mean()),'delta':float((y-x).mean())}
common=a['alive']&b['alive']&(np.arange(shape[0])[:,None]>=fork)
out['common_alive_wrist_mm']={case:[float((norm(z['wrist']-z['wrist_ref'])*1000)[...,i][common].mean()) for i in (0,1)] for case,z in data.items()}
out['new_failure_ids']=np.flatnonzero(fb&~fa).tolist();out['rescued_ids']=np.flatnonzero(fa&~fb).tolist()
print(json.dumps(out,indent=2))
(root/'metrics.json').write_text(json.dumps(out,indent=2))
(root/'event_outcomes.json').write_text(json.dumps({'events':events,'outcomes':records},indent=2))
