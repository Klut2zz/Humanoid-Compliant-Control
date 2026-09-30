#new change
"""Reuse identical actuator audit, replacing only the independent config."""
from pathlib import Path
path=Path(__file__).resolve().with_name('test_wrist_actuator_audit.py')
source=path.read_text()
needle="needle=\"exec(compile(source,str(path),'exec'),\""
assert source.count(needle)==1
injection="source=source.replace('from dual_sagittal_csv_v2_cfg import SagittalCsvV2_35Cfg','from wrist_pitch_kp10_cfg import WristPitchKp10Cfg')\n"
source=source.replace(needle,injection+needle)
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
