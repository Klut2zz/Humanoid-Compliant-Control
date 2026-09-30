#new change
"""Numeric-only pressure diagnostics; no simulation, training, or written report."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_stand_force_joint_medium import analyze, stats
from evaluate_stand_force_joint_v1 import summarize, response


def prefix(path, steps):
    z=dict(np.load(path)); length=len(z['alive'])
    assert length>=steps
    return {k:(v[:steps] if v.ndim and v.shape[0]==length else v) for k,v in z.items()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('trace',type=Path)
    p.add_argument('--baseline',type=Path,required=True,help='Same model, seed, population at 4--7 N')
    p.add_argument('--zero',type=Path,required=True,help='Same model, seed, population at zero force')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assert not args.output.exists()
    z=dict(np.load(args.trace));steps,envs=z['alive'].shape;valid=z['alive']
    meta=json.loads(args.trace.with_suffix('.json').read_text())
    assert meta['all_network_weights_unchanged'] and meta['optimizer_state_unchanged'] and meta['optimizer_updates']==0
    old=prefix(args.baseline,steps);zero=prefix(args.zero,steps)
    assert str(old['checkpoint'])==str(z['checkpoint'])==str(zero['checkpoint'])
    assert int(old['seed'])==int(z['seed'])==int(zero['seed'])
    assert old['alive'].shape==z['alive'].shape==zero['alive'].shape
    common=valid&old['alive']
    for k in ('event_index','event_time','ramp','hold','body'):
        np.testing.assert_array_equal(z[k][common],old[k][common],err_msg=k)
    # Existing 4--7 N data has no pre-cap telemetry. Its minimum k*dmax is
    # 11.2 N in this configuration, so it was not displacement-capped.
    old_mag=np.linalg.norm(old['sampled_peak'],axis=-1)
    assert np.min(z['event_stiffness']*float(z['max_displacement_m']))>7.
    np.testing.assert_allclose(z['requested_peak_n'][common],old_mag[common]+3.,atol=5e-5,rtol=1e-5)
    np.testing.assert_allclose(z['requested_peak'][common]/z['requested_peak_n'][common,None],
        old['sampled_peak'][common]/old_mag[common,None],atol=3e-5,rtol=1e-5)
    r=analyze(args.trace);r['runtime_integrity']=meta
    r['baseline_4to7_same_500_steps']=summarize(old)[0]
    r['paired_nominal_events']='same timing/hands/direction/quantile; requested magnitude +3 N'
    requested=z['requested_peak_n'];capped=np.linalg.norm(z['sampled_peak'],axis=-1)
    accepted=np.linalg.norm(z['accepted_peak'],axis=-1)
    actual=np.linalg.norm(z['physical_force'],axis=-1)
    np.testing.assert_allclose(capped[valid],np.minimum(requested,z['event_stiffness'][...,0]*float(z['max_displacement_m']))[valid],atol=5e-5,rtol=1e-5)
    observed=valid&(z['event_time']>=0)&~z['ik_pending']
    st,en=np.where(observed);keys=np.stack((en,z['event_index'][st,en]),axis=1)
    _,first=np.unique(keys,axis=0,return_index=True);st,en=st[first],en[first]
    rows=[]
    for s,e in zip(st,en):
        idx=int(z['event_index'][s,e]);mask=valid[:,e]&(z['event_index'][:,e]==idx)&(z['event_time'][:,e]>=0)
        body=int(z['body'][s,e]);side='left' if body==int(z['wrist_body_ids'][0]) else 'right'
        rows.append(dict(env=int(e),event=idx,side=side,first_observed_step=int(s),
            requested_peak_N=float(requested[s,e]),post_displacement_cap_N=float(capped[s,e]),
            ik_accepted_peak_N=float(accepted[s,e]),ik_scale=float(z['ik_scale'][s,e]),
            recorded_physical_peak_N=float(actual[mask,e].max()),
            hold_observed=bool(np.any(mask&(z['event_time'][:,e]>=z['ramp'][:,e])&(z['event_time'][:,e]<z['ramp'][:,e]+z['hold'][:,e]))),
            ik_residual_mm=float(z['ik_active_residual_m'][s,e]*1000),
            ik_direction_cosine=float(z['ik_direction_cosine'][s,e])))
    r['force_chain']={
        'events':len(rows),'requested_peak_N':stats(requested[st,en]),'post_displacement_cap_N':stats(capped[st,en]),
        'ik_accepted_peak_N':stats(accepted[st,en]),
        'displacement_capped_events':int(np.sum(capped[st,en]<requested[st,en]-1e-4)),
        'ik_rejected_events':int(np.sum(accepted[st,en]<=1e-6)),
        'accepted_peak_at_least7N_events':int(np.sum(accepted[st,en]>=7.-1e-4)),
        'recorded_physical_peak_N':stats([x['recorded_physical_peak_N'] for x in rows]),
        'by_wrist':{side:{'events':sum(x['side']==side for x in rows),
                         'rejected':sum(x['side']==side and x['ik_scale']<=0 for x in rows)} for side in ('left','right')},
        'note':'An event cut short by failure or recording end may not reach its accepted peak.'}
    r['force_events']=rows
    r['wrist_response_vs_matched_zero']=response(z,zero)
    failed=(valid&z['done']).any(0);old_failed=(old['alive']&old['done']).any(0)
    r['paired_failure_changes']={'new_failed':np.where(failed&~old_failed)[0].tolist(),
                                 'saved':np.where(~failed&old_failed)[0].tolist(),
                                 'failed_in_both':np.where(failed&old_failed)[0].tolist()}
    mask=valid&(np.arange(steps)[:,None]>=50)
    wrist=np.linalg.norm(z['wrist']-z['wrist_ref'],axis=-1)*1000
    r['survivor_wrist_max_mm']=wrist[mask&~failed[None,:]].max(0).tolist() if (~failed).any() else None
    root_speed=np.linalg.norm(z['root_vel'][...,:2],axis=-1)*1000
    r['root_speed_mm_s']=stats(root_speed[valid])
    r['failure_event_context']=[]
    for event in r['failures']:
        s,e=event['step'],event['env'];active=np.where((np.linalg.norm(z['force'][:s+1,e],axis=-1)>.1))[0]
        prior=int(active[-1]) if len(active) else None
        h=max(0,s-round(.5/float(z['dt'])))
        r['failure_event_context'].append({**event,
            'last_accepted_force_vector_N':z['accepted_peak'][prior,e].tolist() if prior is not None else None,
            'last_requested_peak_N':float(requested[prior,e]) if prior is not None else None,
            'last_ik_scale':float(z['ik_scale'][prior,e]) if prior is not None else None,
            'half_second_before_root_offset_mm':(z['root_error'][h,e]*1000).tolist(),
            'half_second_before_root_velocity_mm_s':(z['root_vel'][h,e,:2]*1000).tolist(),
            'half_second_before_tilt_deg':float(np.rad2deg(z['tilt'][h,e]))})
    r['notes']=['Pressure test only: 500 steps is not 30 s multiseed acceptance.',
        'IK-rejected events have no delivered physical force; not proof of robot tolerance.',
        'Recovery qualification only counts complete 3--5 s quiet windows; censored tails excluded.',
        'Foot metric is ankle-link XY reference error, not contact-point slip.',
        'Wrist response removes root translation, not root rotation; it is a compliance proxy.']
    with args.output.open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps({k:r[k] for k in ('fixed_metrics','baseline_4to7_same_500_steps','failure_phase_counts',
        'force_chain','wrist_response_vs_matched_zero','recovery_final_half_second','safety_gate')},indent=2))


if __name__=='__main__':main()
