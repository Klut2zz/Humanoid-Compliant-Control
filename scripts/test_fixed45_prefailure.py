#new change
import os
from pathlib import Path
os.environ['DIAGNOSTIC_ARM_SCALE']='0.45'
path=Path(__file__).resolve().with_name('test_dual_amplitude_50_100.py')
source=path.read_text()
old='from dual_amplitude_50_100_cfg import AmplitudeDiagnosticCfg'
assert source.count(old)==1
source=source.replace(old,'from fixed45_prefailure_cfg import Fixed45PrefailureCfg')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
