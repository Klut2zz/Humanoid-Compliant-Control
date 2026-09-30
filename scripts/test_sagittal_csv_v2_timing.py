#new change
"""Corrected 25% CSV trace; pre-step q/ref and separately labelled targets."""
import os
from pathlib import Path
os.environ['SAGITTAL_SCALE']='25'
path=Path(__file__).resolve().with_name('test_dual_sagittal_timing.py')
source=path.read_text()
old='from dual_sagittal_gate_cfg import SagittalSCALECfg'
assert source.count(old)==1
source=source.replace(old,'from dual_sagittal_csv_v2_cfg import SagittalCsvV2_25Cfg')
old="                    'ref':c.dof_pos[:,1].cpu().numpy().copy(),"
assert source.count(old)==1
source=source.replace(old,old+"\n                    'ref_vel':c.dof_vel[:,1].cpu().numpy().copy(),\n                    'phase_time':c._relative_timestamp.cpu().numpy().copy(),\n                    'source_time':(c.motion_count*c.motion_dt).cpu().numpy().copy(),")
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
