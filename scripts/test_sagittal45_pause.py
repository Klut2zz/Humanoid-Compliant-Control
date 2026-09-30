#new change
import os
from pathlib import Path
os.environ['DIAGNOSTIC_ARM_SCALE']='0.45'
path=Path(__file__).resolve().with_name('test_sagittal_pause.py')
source=path.read_text().replace('from sagittal_pause_reference import branch_step','from sagittal45_pause_reference import branch_step')
needle='source=path.read_text()'
assert source.count(needle)==1
original="source = (root/'test_neutral_dual_adapt.py').read_text()"
replacement="expected[:,ARM]+=e._dual_arm_raw*.15\\n                    expected[:,NAMES.index('right_wrist_pitch_joint')]+=e._pitch_probe_delta"
patched=original+'\nsource=source.replace('+repr('expected[:,ARM]+=e._dual_arm_raw*.15')+','+repr(replacement)+')'
extra='\nsource=source.replace('+repr('from dual_sagittal_gate_cfg import SagittalSCALECfg')+','+repr('from dual_amplitude_50_100_cfg import AmplitudeDiagnosticCfg')+')'
extra+='\nsource=source.replace('+repr(original)+','+repr(patched)+')'
source=source.replace(needle,needle+extra)
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
