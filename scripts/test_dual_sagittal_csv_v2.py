#new change
import os
from pathlib import Path
scale=os.environ['SAGITTAL_SCALE']
assert scale in ('15','25')
cfg='SagittalCsvV2Cfg' if scale=='15' else 'SagittalCsvV2_25Cfg'
path=Path(__file__).resolve().with_name('test_neutral_dual_adapt.py')
source=path.read_text()
needle="exec(compile(s,str(path),'exec'),"
assert source.count(needle)==1
hook="s=s.replace('from neutral_dual_adapt_cfg import NeutralDualAdaptCfg, ARM','from neutral_dual_adapt_cfg import ARM\\nfrom dual_sagittal_csv_v2_cfg import "+cfg+" as NeutralDualAdaptCfg')\n"
source=source.replace(needle,hook+needle)
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
