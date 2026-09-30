#new change
from pathlib import Path
path=Path(__file__).resolve().with_name('test_sagittal_csv_v2_timing.py')
source=path.read_text().replace('SagittalCsvV2_25Cfg','SagittalCsvV2_35Cfg')
needle="exec(compile(source,str(path),'exec'),"
assert source.count(needle)==1
old="                    'q':r.data.joint_pos[:,ids].cpu().numpy().copy(),"
new=old+"\n                    'audit':np.asarray(__import__('wrist_actuator_audit').metadata(r)),\n                    'estimated_torque_raw':r.data.computed_torque[:,ids].cpu().numpy().copy(),\n                    'estimated_torque_clipped':r.data.applied_torque[:,ids].cpu().numpy().copy(),\n                    'velocity_target':r.data.joint_vel_target[:,ids].cpu().numpy().copy(),"
injection='source=source.replace('+repr(old)+','+repr(new)+')\n'
source=source.replace(needle,injection+needle)
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
