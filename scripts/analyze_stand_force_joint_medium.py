#new change
"""Numeric-only analysis of medium-force traces; does not launch simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_stand_force_joint_v1 import summarize, response


def stats(x):
    x = np.asarray(x)
    return {'n': int(x.size), **({k: float(v) for k, v in zip(
        ('min', 'mean', 'p50', 'p95', 'max'),
        (x.min(), x.mean(), np.median(x), np.percentile(x, 95), x.max()))} if x.size else {})}


def analyze(path, previous_dir=None):
    z = dict(np.load(path)); n, envs = z['alive'].shape
    valid = z['alive']; dt = float(z['dt'])
    fixed, recovery = summarize(z)
    force = np.linalg.norm(z['force'], axis=-1)
    delivered_error = np.max(abs(z['physical_force']-z['force'])[valid])
    assert delivered_error < 1e-4
    t, ramp, hold = (z[k] for k in ('event_time', 'ramp', 'hold'))
    phase = np.full((n, envs), 'wait', dtype='<U12')
    phase[(t>=0)&(t<ramp)]='rise'
    phase[(t>=ramp)&(t<ramp+hold)]='hold'
    phase[(t>=ramp+hold)&(t<2*ramp+hold)]='release'
    #new change: release_age is recorded AFTER env.step and is reset on failure.
    # Classify failures from pre-step force history/time, never post-reset age.
    has_seen_force=np.maximum.accumulate(force>1e-4,axis=0)
    phase[(t<0)&has_seen_force]='recovery'
    failed = valid & z['done']
    failure_rows=[{'env':int(e), 'step':int(s), 'seconds':float((s+1)*dt),
                   'phase':str(phase[s,e]), 'force_N':float(force[s,e])} for s,e in zip(*np.where(failed))]
    # One row per sampled event after its IK decision; never weight long holds more.
    observed = valid & (t>=0) & ~z['ik_pending']
    st, en = np.where(observed)
    keys = np.stack((en, z['event_index'][st,en]),axis=1)
    _, first = np.unique(keys,axis=0,return_index=True)
    st, en = st[first], en[first]
    requested = np.linalg.norm(z['sampled_peak'][st,en],axis=-1)
    accepted = np.linalg.norm(z['accepted_peak'][st,en],axis=-1)
    scale = z['ik_scale'][st,en]
    np.testing.assert_allclose(accepted, requested*scale, atol=2e-5, rtol=1e-5)
    events = {'events_observed':int(len(st)), 'sampled_peak_N':stats(requested), 'accepted_peak_N':stats(accepted),
              'accepted':int((scale>0).sum()), 'rejected':int((scale<=0).sum()),
              'unscaled':int((scale>=.999).sum()), 'scaled_nonzero':int(((scale>0)&(scale<.999)).sum()),
              'accepted_peak_at_least4N_fraction':float((accepted>=4.-1e-4).mean()),
              'ik_scale':stats(scale), 'actual_hold_force_N':stats(force[valid&(phase=='hold')]),
              'physical_buffer_vs_context_max_error_N':float(delivered_error)}
    tail=[]; release=[]; drift=np.linalg.norm(z['foot_drift'][...,:2],axis=-1).max(-1)
    root=np.linalg.norm(z['root_error'],axis=-1); speed=np.linalg.norm(z['root_vel'][...,:2],axis=-1)
    feet=np.linalg.norm(z['foot_error'],axis=-1).max(-1)
    for event in recovery:
        end,e,start=event['end_step'],event['env'],event['release_step']
        if event['failed']: continue
        sl=slice(max(start,end-24),end+1)
        tail.append([speed[sl,e].mean()*1000,root[sl,e].mean()*1000,feet[sl,e].mean()*1000,
                     np.rad2deg(z['tilt'][sl,e]).mean(),drift[sl,e].mean()*1000])
        release.append([speed[start,e]*1000,root[start,e]*1000,drift[start,e]*1000])
    tail_keys=('root_speed_mm_s','root_offset_mm','foot_reference_error_mm','tilt_deg','foot_drift_from_start_mm')
    result={'fixed_metrics':fixed,'force_integrity':events,'failure_phase_counts':{p:int((failed&(phase==p)).sum()) for p in np.unique(phase)},
            'failures':failure_rows,'recovery_final_half_second':{k:stats(np.asarray(tail)[:,i]) for i,k in enumerate(tail_keys)} if tail else {},
            'release_start':{k:stats(np.asarray(release)[:,i]) for i,k in enumerate(('root_speed_mm_s','root_offset_mm','foot_drift_from_start_mm'))} if release else {},
            'foot_drift_from_start_peak_p95_mm':float(np.percentile(np.where(valid,drift,0).max(0),95)*1000),
            'seconds':n*dt,'environments':envs,'seed':int(z['seed']), 'checkpoint':str(z['checkpoint'])}
    #new change: disclose peaks as well as the mean, with the same 1 s warmup.
    mask=valid&(np.arange(n)[:,None]>=50)
    wrist_error=np.linalg.norm(z['wrist']-z['wrist_ref'],axis=-1)*1000
    result['wrist_error_mm']={s:stats(wrist_error[...,i][mask]) for i,s in enumerate(('left','right'))}
    result['force_events_by_wrist']={s:{'events':int((z['body'][st,en]==bid).sum()),
        'rejected':int(((z['body'][st,en]==bid)&(scale<=0)).sum())}
        for s,bid in zip(('left','right'),z['wrist_body_ids'])}
    # Keep the prior support/recovery gates. Wrist tolerance is explicitly the
    # user's 100 mm application criterion, NOT a 100 mm root/foot allowance.
    result['safety_gate'] = bool(fixed['survival']>=.95 and fixed['support_qualified']>=.95
        and (fixed['recovery']['qualified_fraction'] or 0)>=.95 and max(fixed['wrist_mae_mm'])<100)
    if previous_dir:
        old_path=Path(previous_dir)/f'force_seed{int(z["seed"])}.npz'
        old=np.load(old_path)
        common=valid&old['alive'][:n]
        for k in ('event_index','event_time','ramp','hold','body'):
            np.testing.assert_array_equal(z[k][common],old[k][:n][common],err_msg=k)
        a=old['sampled_peak'][:n]; magnitude=np.linalg.norm(a,axis=-1)
        expected=a/np.maximum(magnitude[...,None],1e-8)*(1.5*magnitude[...,None]+1.)
        np.testing.assert_allclose(z['sampled_peak'][common],expected[common],atol=3e-5,rtol=1e-5)
        result['event_pairing_to_2to4'] = 'exact times/hands; same directions and normalized force quantiles'
        zero=np.load(Path(previous_dir)/f'zero_seed{int(z["seed"])}.npz')
        sliced={k:(zero[k][:n] if zero[k].ndim and zero[k].shape[0]==len(zero['alive']) else zero[k]) for k in zero.files}
        assert str(sliced['checkpoint'])==str(z['checkpoint'])
        result['wrist_response_relative_to_matched_zero']=response(z,sliced)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('--previous-dir');p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    assert not out.exists(), 'Preserve previous results; choose another path.'
    result=analyze(args.trace,args.previous_dir)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
