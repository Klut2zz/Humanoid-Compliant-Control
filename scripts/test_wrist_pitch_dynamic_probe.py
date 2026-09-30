#new change
from pathlib import Path
path=Path(__file__).resolve().with_name('test_sagittal_csv_v2_timing.py')
source=path.read_text().replace('from dual_sagittal_csv_v2_cfg import SagittalCsvV2_25Cfg','from wrist_pitch_dynamic_probe_cfg import WristPitchDynamicProbeCfg')
# Adjust the existing target identity check to include exactly the added delta.
needle="source=path.read_text()"
assert source.count(needle)==1
old="source = (root/'test_neutral_dual_adapt.py').read_text()"
#new change
replacement="expected[:,ARM]+=e._dual_arm_raw*.15\\n                    expected[:,NAMES.index('right_wrist_pitch_joint')]+=e._pitch_probe_delta"
new=old+"\nsource=source.replace("+repr('expected[:,ARM]+=e._dual_arm_raw*.15')+","+repr(replacement)+")"
source=source.replace(needle,needle+'\nsource=source.replace('+repr(old)+','+repr(new)+')')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
