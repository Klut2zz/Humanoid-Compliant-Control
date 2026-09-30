#new change
"""Reuse audited time-warp and snapshot capture at the 45% pre-failure boundary."""
from pathlib import Path
path=Path(__file__).resolve().with_name('sagittal_pause_reference.py')
source=path.read_text()
assert source.count('if step != 100:')==1
source=source.replace('if step != 100:','if step != 250:')
source=source.replace('PAUSE_INSTALLED step=100','PAUSE_INSTALLED step=250')
exec(compile(source,str(path),'exec'),globals())
