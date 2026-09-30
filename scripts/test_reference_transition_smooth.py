#new change
from pathlib import Path
path=Path(__file__).resolve().with_name('test_wrist_pitch_dynamic_probe.py')
source=path.read_text()
old='from wrist_pitch_dynamic_probe_cfg import WristPitchDynamicProbeCfg'
assert source.count(old)==1
source=source.replace(old,'from reference_transition_smooth_cfg import ReferenceTransitionSmoothCfg')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
