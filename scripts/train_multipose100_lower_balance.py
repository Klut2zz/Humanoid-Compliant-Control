#new change
from pathlib import Path
path=Path(__file__).resolve().with_name('train_multipose45_lower_balance.py')
source=path.read_text().replace('import multipose45_lower_balance_cfg','import multipose100_lower_balance_cfg')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
