#new change
from pathlib import Path
path=Path(__file__).resolve().with_name('test_dual_fixed_transition.py')
source=path.read_text()
old='from dual_fixed_transition_cfg import FixedTransitionCfg'
assert source.count(old)==1
source=source.replace(old,'from fixed_sagittal35_cfg import FixedSagittal35Cfg')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
