#new change
from pathlib import Path
path=Path(__file__).resolve().with_name('test_fixed45_prefailure.py')
source=path.read_text()
old='from fixed45_prefailure_cfg import Fixed45PrefailureCfg'
assert source.count(old)==1
source=source.replace(old,'from fixed45_paused_median_cfg import Fixed45PausedMedianCfg')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
