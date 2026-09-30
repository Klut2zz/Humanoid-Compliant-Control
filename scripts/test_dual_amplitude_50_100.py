#new change
from pathlib import Path
path=Path(__file__).resolve().with_name('test_reference_transition_smooth.py')
source=path.read_text()
old='from reference_transition_smooth_cfg import ReferenceTransitionSmoothCfg'
assert source.count(old)==1
source=source.replace(old,'from dual_amplitude_50_100_cfg import AmplitudeDiagnosticCfg')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
