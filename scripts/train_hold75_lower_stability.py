#new change
from pathlib import Path
path=Path(__file__).with_name('train_dual_fixed_transition.py')
source=path.read_text().replace('import dual_fixed_transition_train_cfg','import hold75_lower_stability_cfg')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
