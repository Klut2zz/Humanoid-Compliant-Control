#new change
"""Frozen paired-by-initial-condition timing capture; no policy or physics edits.

Select 15/25 via SAGITTAL_SCALE; TRACE_PATH is a new NPZ output path.
Snapshots are PRE-step, previous_target is explicitly the previous command.
issued_target is captured after the step and invalid on reset rows.
"""
import os
from pathlib import Path

root = Path(__file__).resolve().parent
scale = os.environ['SAGITTAL_SCALE']
assert scale in ('15', '25')
source = (root/'test_neutral_dual_adapt.py').read_text()
hook = r'''
s=s.replace('from neutral_dual_adapt_cfg import NeutralDualAdaptCfg, ARM',
            'from neutral_dual_adapt_cfg import ARM\nfrom dual_sagittal_gate_cfg import SagittalSCALECfg as NeutralDualAdaptCfg')
s=s.replace('        with torch.no_grad():', ''' + "'''" + r'''
        import os
        trace_path=os.environ['TRACE_PATH']
        assert not __import__('pathlib').Path(trace_path).exists(), trace_path
        traces=[]
        with torch.no_grad():''' + "'''" + r''')
needle='                reference_before=c.dof_pos[:,1].clone()'
assert s.count(needle)==1
s=s.replace(needle, ''' + "'''" + r'''
                r,c,ids,fids,feet,fe,com,forces=state(e)
                mass=r.data.default_mass.to(e.device)
                com_vel=(r.data.body_com_vel_w[:,:,:3]*mass[:,:,None]).sum(1)/mass.sum(1)[:,None]
                frame={'step':np.asarray(step),
                    'alive':alive.cpu().numpy().copy(),
                    'q':r.data.joint_pos[:,ids].cpu().numpy().copy(),
                    'dq':r.data.joint_vel[:,ids].cpu().numpy().copy(),
                    'ref':c.dof_pos[:,1].cpu().numpy().copy(),
                    'previous_target':r.data.joint_pos_target[:,ids].cpu().numpy().copy(),
                    'com':(com-e.scene.env_origins).cpu().numpy().copy(),
                    'com_vel':com_vel.cpu().numpy().copy(),
                    'feet':(feet-e.scene.env_origins[:,None]).cpu().numpy().copy(),
                    'foot_error':fe.cpu().numpy().copy(),
                    'force':forces.cpu().numpy().copy(),
                    'root':(r.data.root_pos_w-e.scene.env_origins).cpu().numpy().copy(),
                    'root_vel':r.data.root_lin_vel_w.cpu().numpy().copy(),
                    'gravity':r.data.projected_gravity_b.cpu().numpy().copy(),
                    'lower_action':action.cpu().numpy().copy()}
                reference_before=c.dof_pos[:,1].clone()''' + "'''" + r''')
needle='                assert torch.isfinite(reward).all()'
assert s.count(needle)==1
s=s.replace(needle, ''' + "'''" + r'''
                frame['issued_target']=r.data.joint_pos_target[:,ids].cpu().numpy().copy()
                frame['terminated']=terminated.cpu().numpy().copy()
                frame['truncated']=truncated.cpu().numpy().copy()
                traces.append(frame)
                assert torch.isfinite(reward).all()''' + "'''" + r''')
s=s.replace('        steady_mean=', ''' + "'''" + r'''
        np.savez_compressed(trace_path,**{k:np.stack([f[k] for f in traces]) for k in traces[0]},joint_names=np.asarray(NAMES),dt=np.asarray(e.step_dt))
        print('TRACE_SAVED',trace_path,flush=True)
        steady_mean=''' + "'''" + r''')
'''
hook = hook.replace('SagittalSCALECfg', f'Sagittal{scale}Cfg')
needle = "exec(compile(s,str(path),'exec'),"
assert source.count(needle) == 1
source = source.replace(needle, hook + '\n' + needle)
exec(compile(source, str(root/'test_neutral_dual_adapt.py'), 'exec'),
     {'__name__': '__main__', '__file__': str(root/'test_neutral_dual_adapt.py')})
