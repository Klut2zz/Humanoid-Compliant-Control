#new change
"""Frozen smoke/acceptance entry; same upper target identity checks as probes."""
from pathlib import Path
path=Path(__file__).resolve().with_name('test_wrist_pitch_dynamic_probe.py')
source=path.read_text()
old='from wrist_pitch_dynamic_probe_cfg import WristPitchDynamicProbeCfg'
assert source.count(old)==1
source=source.replace(old,'from multipose45_lower_balance_cfg import MultiPose45LowerBalanceCfg')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
