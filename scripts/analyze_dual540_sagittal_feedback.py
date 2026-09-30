#new change
"""Paired frozen feedback acceptance; saves metrics only, no formal report."""
import argparse,ast,json
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--directory',required=True);args=p.parse_args()
root=Path(args.directory)
norm=lambda x:np.linalg.norm(x,axis=-1)
ns={'torch':torch,'np':np,'reporting_exceptions':[]}
tree=ast.parse(Path(__file__).with_name('analyze_dual540_release_ablation.py').read_text())
fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='equal')
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'snapshot_check','exec'),ns)
data={k:dict(np.load(root/(k+'.npz'))) for k in ('baseline','repeat','feedback')}
a=data['baseline'];b=data['feedback'];fork=int(a['fork_step']);dt=float(a['dt']);shape=a['alive'].shape
snap=torch.load(root/'baseline.snapshot.pt',map_location='cpu',weights_only=False)
for case in ('repeat','feedback'):
    ns['equal'](snap,torch.load(root/(case+'.snapshot.pt'),map_location='cpu',weights_only=False))
for key in a:
    if key=='branch':continue
    assert np.array_equal(a[key],data['repeat'][key]),'repeat/'+key
    if a[key].ndim>=2 and a[key].shape[:2]==shape:
        assert np.array_equal(a[key][:fork+1],b[key][:fork+1]),'prefix/'+key
names=list(a['joint_names']);controlled=[]
for side in ('left','right'):
    hip=names.index(side+'_hip_pitch_joint');ankle=names.index(side+'_ankle_pitch_joint');controlled.extend([hip,ankle])
    assert np.array_equal(b['feedback'][...,hip],-b['feedback'][...,ankle])
other=[i for i in range(29) if i not in controlled]
assert np.count_nonzero(b['feedback'][...,other])==0
assert abs(b['feedback']).max()<=.003001
assert np.max(abs(b['target'][fork+1]-a['target'][fork+1]-b['feedback'][fork+1]))<5e-7
summary={'snapshot_equal':True,'baseline_repeat_exact':True,'prefix_equal':True,
         'reporting_only_exceptions':ns['reporting_exceptions'],'fork_step':fork,
         'steps':shape[0],'num_envs':shape[1],'dt':dt,'feedback_limit_rad':.003,'cases':{}}
outcomes={}
for case,z in [('baseline',a),('feedback',b)]:
    fail=z['done']&z['alive'];failed=fail.any(0);outcomes[case]=failed
    valid=z['alive'];metric={'failed':int(failed.sum()),'containment':int((valid&z['containment']).any(0).sum()),
       'failure_time_median_s':float(np.median((fail.argmax(0)[failed]+1)*dt)),
       'failed_at_500':int(fail[:500].any(0).sum())}
    for key,v in [('root_mm',norm(z['root_error'])*1000),('foot_ref_mm',norm(z['foot_error']).max(-1)*1000),
                  ('foot_drift_mm',norm(z['foot_drift'][...,:2]).max(-1)*1000),('tilt_deg',np.rad2deg(z['tilt']))]:
        metric[key]={'mean':float(v[valid].mean()),'max':float(v[valid].max())}
    summary['cases'][case]=metric
fa=outcomes['baseline'];fb=outcomes['feedback']
summary['failure_overlap']={'both':int((fa&fb).sum()),'baseline_only':int((fa&~fb).sum()),'feedback_only':int((fb&~fa).sum())}
# First-event inputs must remain identical up to release or either failure.
inputs=('force','ref','nominal_ref','sampled_peak','accepted_peak','stiffness','body','ramp','hold')
events=[];early=[];maxdiff={k:0. for k in inputs}
paired={case:[] for case in ('baseline','feedback')}
for e in range(shape[1]):
    #new change: the three events already released before the fork are outside
    # this active-event paired window, but remain in overall rollout metrics.
    if a['event_time'][fork,e]<0:continue
    remaining=float(2*a['ramp'][fork,e]+a['hold'][fork,e]-a['event_time'][fork,e])
    expected_release=fork+int(np.ceil(remaining/dt))
    stop=min(expected_release,shape[0]-1)
    common=a['alive'][fork:stop,e]&b['alive'][fork:stop,e]
    for key in inputs:
        delta=float(np.max(abs(a[key][fork:stop,e][common].astype(float)-b[key][fork:stop,e][common].astype(float)))) if common.any() else 0.
        maxdiff[key]=max(maxdiff[key],delta)
    releases=np.flatnonzero((a['event_index'][1:,e]>a['event_index'][:-1,e])&(a['event_time'][1:,e]<0)&a['alive'][1:,e])+1
    releases=releases[(releases>=fork)&(abs(releases-expected_release)<=2)]
    if not len(releases):early.append(e);continue
    release=int(releases[0]);wait=-float(a['event_time'][release,e]);end=release+int(np.floor(wait/dt))-1
    if end>=shape[0]:continue
    events.append({'env':e,'release':release,'end':end,'wait_s':wait})
    common=a['alive'][fork:end+1,e]&b['alive'][fork:end+1,e]
    for key in inputs:
        delta=float(np.max(abs(a[key][fork:end+1,e][common].astype(float)-b[key][fork:end+1,e][common].astype(float))))
        maxdiff[key]=max(maxdiff[key],delta)
    for case,z in [('baseline',a),('feedback',b)]:
        d=np.flatnonzero(z['done'][fork:end,e]);failed=bool(len(d));last=fork+int(d[0]) if failed else end
        speed=norm(z['root_vel'][:,e,:2]);pos=norm(z['root_error'][:,e]);foot=norm(z['foot_error'][:,e]).max(-1)
        row={'env':e,'failed':failed,'stop':last,'recovered':bool(not failed and speed[end]<.03 and pos[end]<.05 and foot[end]<.02 and z['tilt'][end,e]<np.deg2rad(5))}
        if not failed:
            row.update(root_end_mm=float(pos[end]*1000),speed_tail_mps=float(speed[max(release,end-24):end+1].mean()),
                foot_end_mm=float(foot[end]*1000),tilt_end_deg=float(np.rad2deg(z['tilt'][end,e])))
        paired[case].append(row)
summary['first_event_input_maxdiff']=maxdiff
assert all(v==0 for v in maxdiff.values()),'First-event paired input changed'
summary['baseline_failed_before_first_release']=len(early)
summary['already_waiting_at_fork']=int((a['event_time'][fork]<0).sum())
summary['complete_first_recovery_windows']=len(events)
for case,rows in paired.items():summary['cases'][case]['full_quiet_window']={'failed':sum(x['failed'] for x in rows),'recovered':sum(x['recovered'] for x in rows)}
both=[i for i in range(len(events)) if not paired['baseline'][i]['failed'] and not paired['feedback'][i]['failed']]
summary['common_window_survivors']=len(both)
summary['paired_recovery_end']={}
for key in ('root_end_mm','speed_tail_mps','foot_end_mm','tilt_end_deg'):
    x=np.array([paired['baseline'][i][key] for i in both]);y=np.array([paired['feedback'][i][key] for i in both])
    summary['paired_recovery_end'][key]={'baseline':float(x.mean()),'feedback':float(y.mean()),'delta':float((y-x).mean())}
common=a['alive']&b['alive']&(np.arange(shape[0])[:,None]>=fork)
summary['common_alive_wrist_mae_mm']={case:[float((norm(z['wrist']-z['wrist_ref'])*1000)[...,i][common].mean()) for i in range(2)] for case,z in [('baseline',a),('feedback',b)]}
summary['feedback_actual_mrad']={'mean_abs':float(abs(b['feedback'][...,controlled][common]).mean()*1000),'max_abs':float(abs(b['feedback']).max()*1000),
 'max_step_change':float(abs(np.diff(b['feedback'],axis=0)).max()*1000)}
#new change: exact target decomposition; projections describe the coupled
# closed-loop response, NOT a standalone actuator gain or causal intention.
summary['baseline_failure_cohort_early_response']=[]
for lo,hi in ((101,125),(125,150),(150,175)):
    m=a['alive'][lo:hi]&b['alive'][lo:hi]&fa[None]
    f=b['feedback'][lo:hi][...,controlled][m];den=float((f*f).sum())
    target=(b['target'][lo:hi]-a['target'][lo:hi])[...,controlled][m]
    residual=(b['residual'][lo:hi]-a['residual'][lo:hi])[...,controlled][m]
    actual=(b['q'][lo:hi]-a['q'][lo:hi])[...,controlled][m]
    projections={key:float((value*f).sum()/den) for key,value in [('target',target),('base_target',target-residual),('original_residual',residual-f),('actual_q',actual)]}
    summary['baseline_failure_cohort_early_response'].append({'steps':[lo,hi-1],'projections_on_feedback':projections,
      'root_x_delta_mm':float((b['root_error'][lo:hi]-a['root_error'][lo:hi])[...,0][m].mean()*1000),
      'root_vx_delta_mps':float((b['root_vel'][lo:hi]-a['root_vel'][lo:hi])[...,0][m].mean())})
summary['survivor_to_failure_ids']=np.flatnonzero(fb&~fa).tolist()
summary['failure_to_survivor_ids']=np.flatnonzero(fa&~fb).tolist()
print(json.dumps(summary,indent=2))
(root/'metrics.json').write_text(json.dumps(summary,indent=2))
(root/'event_outcomes.json').write_text(json.dumps({'events':events,'outcomes':paired},indent=2))
