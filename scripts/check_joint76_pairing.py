# new change
"""Audit joint76 paired inference and print metrics; no report generation."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

p=argparse.ArgumentParser()
p.add_argument('--data_root',type=Path,default=Path('/tmp'))
root=p.parse_args().data_root
names=['zero','repeat','hip','ankle','both']
paths={n:root/f'joint76_{n}_20260909' for n in names}
def equal(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.allclose(a,b,rtol=0,atol=0,equal_nan=True)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

snapshot=torch.load(paths['zero']/'branch_snapshot.pt',weights_only=False)
for n in names[1:]:
    equal(snapshot,torch.load(paths[n]/'branch_snapshot.pt',weights_only=False))
    for file in ['B_model1375.csv','balance_timing.csv','ground_contacts.csv','raw_ground_contacts.csv']:
        a=pd.read_csv(paths['zero']/file);b=pd.read_csv(paths[n]/file)
        pd.testing.assert_frame_equal(a[a.step<76].reset_index(drop=True),b[b.step<76].reset_index(drop=True),check_exact=True)
        if n=='repeat':pd.testing.assert_frame_equal(a,b,check_exact=True)
    a=json.load(open(paths['zero']/'metadata.json'));b=json.load(open(paths[n]/'metadata.json'))
    for k in ['resolved_joint_stiffness','resolved_joint_damping','material_properties_env0','physics_shape_audit']:assert a[k]==b[k]
    d0=pd.read_csv(paths['zero']/'B_model1375.csv');d=pd.read_csv(paths[n]/'B_model1375.csv')
    for c in [c for c in d if c.startswith('target:')]:
        expected=.01 if ((n in ['hip','both'] and c=='target:right_hip_roll_joint') or (n in ['ankle','both'] and c=='target:right_ankle_pitch_joint')) else 0.
        assert abs(d.loc[76,c]-d0.loc[76,c]-expected)<1e-6
print('PASS exact branch state/history/RNG, prefix, dynamics, zero repeat; target-only intervention checked')

baseline=pd.read_csv(paths['zero']/'balance_timing.csv')
for n in ['zero','hip','ankle','both']:
    d=pd.read_csv(paths[n]/'B_model1375.csv');t=pd.read_csv(paths[n]/'balance_timing.csv');g=pd.read_csv(paths[n]/'ground_contacts.csv')
    print('\nCASE',n,'failure',json.load(open(paths[n]/'summary.json'))['B_model1375']['first_failure_step'])
    for c in ['com_contact_hull_margin_m','capture_contact_hull_margin_m']:
        mask=(t[c]<0)&(t.episode_valid==1)
        print(c,[k for k in range(76,len(t)-4) if mask.iloc[k:k+5].all()][:1])
    print('side_velocity_cross',[k for k in range(77,110) if t.iloc[k-1].com_vy<0<=t.iloc[k].com_vy][:1])
    for k in [80,85,90,95,100,110]:
        r=t.iloc[k];f=g[(g.step==k)&g.body.str.contains('ankle')].set_index('body').ground_fz
        print(k,'cop_forward_delta_mm',round((r.cop_x-baseline.iloc[k].cop_x)*1000,3),
              'vy',round(r.com_vy,5),'vx',round(r.com_vx,5),
              'y_since_branch_mm',round((r.com_wy-t.iloc[76].com_wy)*1000,3),
              'left_load',round(f['left_ankle_roll_link'],2),'margin_mm',round(r.com_contact_hull_margin_m*1000,3))
