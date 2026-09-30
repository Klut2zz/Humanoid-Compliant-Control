#new change
"""Independent entry; original trainer, actor and accepted references preserved."""
from pathlib import Path
import sys
root=Path('/home/noah/mingkai/general_softmimic')
sys.path.insert(0,str(root/'scripts'))
sys.path.insert(0,str(root/'scripts/rsl_rl'))
path=root/'scripts/rsl_rl/train.py'
source=path.read_text()
needle='import softmimic_gym.tasks  # noqa: F401'
assert source.count(needle)==1
source=source.replace(needle,needle+'\nimport dual_fixed_transition_train_cfg')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
