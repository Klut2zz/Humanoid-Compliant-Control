# new change
"""Print pairing checks and frozen hip-roll probe metrics; creates no report."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data_root', type=Path, default=Path('/tmp'))
root = parser.parse_args().data_root
names = ['zero', 'repeat', 'positive', 'negative']
paths = {n:root/f'hip_roll_target_{n}_20260909' for n in names}
def equal_tree(a,b,path=''):
    if isinstance(a,torch.Tensor):
        if not torch.allclose(a,b,rtol=0,atol=0,equal_nan=True):
            raise AssertionError('Snapshot mismatch: '+path)
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a: equal_tree(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal_tree(x,y,path+'/'+str(i))
    else: assert a==b,path

base=torch.load(paths['zero']/'branch_snapshot.pt',weights_only=False)
for name in names[1:]:
    equal_tree(base,torch.load(paths[name]/'branch_snapshot.pt',weights_only=False))
    for file in ['B_model1375.csv','balance_timing.csv','ground_contacts.csv','raw_ground_contacts.csv']:
        a=pd.read_csv(paths['zero']/file);b=pd.read_csv(paths[name]/file)
        pd.testing.assert_frame_equal(a[a.step<60].reset_index(drop=True),b[b.step<60].reset_index(drop=True),check_exact=True)
        if name=='repeat':pd.testing.assert_frame_equal(a,b,check_exact=True)
    for file in ['initial_state.npy','initial_actor_observation.npy','initial_reference_keypoints.npy']:
        np.testing.assert_array_equal(np.load(paths['zero']/file),np.load(paths[name]/file))
    a=json.load(open(paths['zero']/'metadata.json'));b=json.load(open(paths[name]/'metadata.json'))
    for key in ['resolved_joint_stiffness','resolved_joint_damping','material_properties_env0','physics_shape_audit']:
        assert a[key]==b[key],key
print('PASS: exact exposed branch snapshot/history/RNG, all prefix data, dynamics, full zero-bias replay.')
ds={n:pd.read_csv(p/'B_model1375.csv') for n,p in paths.items()}
# Only one target component can differ at the initial intervention.
cols=[c for c in ds['zero'] if c.startswith('target:')]
for n in ['positive','negative']:
    delta=ds[n].loc[60,cols]-ds['zero'].loc[60,cols]
    assert (delta.drop('target:right_hip_roll_joint').abs()<1e-9).all()
    assert abs(delta['target:right_hip_roll_joint']-({'positive':.01,'negative':-.01}[n]))<1e-6

for name in ['zero','positive','negative']:
    p=paths[name];d=ds[name];t=pd.read_csv(p/'balance_timing.csv');g=pd.read_csv(p/'ground_contacts.csv')
    summary=json.load(open(p/'summary.json'))['B_model1375']; print('\nBRANCH',name,'failure',summary['first_failure_step'])
    for col in ['com_contact_hull_margin_m','capture_contact_hull_margin_m']:
        mask=(t[col]<0)&t.episode_valid.astype(bool)
        starts=[i for i in range(60,len(t)-4) if mask.iloc[i:i+5].all()]
        print(col,'sustained_first',starts[:1])
    for k in [60,65,70,75,80,85,90,95,100,105,110]:
        r=d.iloc[k];s=t.iloc[k]
        if not r.episode_valid:continue
        j='right_hip_roll_joint';forces=g[(g.step==k)&g.body.str.contains('ankle')].set_index('body').ground_fz
        fleft=forces['left_ankle_roll_link'];fright=forces['right_ankle_roll_link']
        print(k, 'q_deg',round(np.rad2deg(r['q:'+j]),3),'oldtarget_deg',round(np.rad2deg(d.iloc[k-1]['target:'+j]),3),
              'q_delta_deg',round(np.rad2deg(r['q:'+j]-ds['zero'].iloc[k]['q:'+j]),3),
              'fz_LR',round(fleft,2),round(fright,2),'right_fraction',round(fright/(fleft+fright),3),
              'margin_mm',round(s.com_contact_hull_margin_m*1000,2),'com_vy',round(s.com_vy,4),
              'tilt_deg',round(np.rad2deg(r.tilt_rad),2))
    for lo,hi in [(60,70),(70,80),(60,95)]:
        j='right_hip_roll_joint';err=(d['q:'+j]-d['target:'+j].shift()).iloc[lo:hi]
        print('MAE',lo,hi,round(np.rad2deg(err.abs().mean()),3))
