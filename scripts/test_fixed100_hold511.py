#new change
from pathlib import Path
path=Path(__file__).with_name('test_fixed100_prefailure.py')
source=path.read_text().replace('from fixed100_prefailure_cfg import Fixed100PrefailureCfg','from fixed100_hold511_cfg import Fixed100Hold511Cfg')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
