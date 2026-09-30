#new change
from pathlib import Path
path=Path(__file__).resolve().with_name('train_dual_fixed_transition.py')
source=path.read_text()
assert source.count('import dual_fixed_transition_train_cfg')==1
source=source.replace('import dual_fixed_transition_train_cfg','import multipose45_lower_balance_cfg')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
