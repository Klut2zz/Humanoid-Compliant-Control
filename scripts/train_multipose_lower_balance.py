#new change
from pathlib import Path
path=Path(__file__).resolve().with_name('train_dual_fixed_transition.py')
source=path.read_text()
needle='import dual_fixed_transition_train_cfg'
assert source.count(needle)==1
source=source.replace(needle,'import multipose_lower_balance_cfg')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
