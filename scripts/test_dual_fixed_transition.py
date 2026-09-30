#new change
from pathlib import Path
path=Path(__file__).resolve().with_name('test_neutral_dual_adapt.py')
source=path.read_text()
needle="exec(compile(s,str(path),'exec'),"
assert source.count(needle)==1
hook="s=s.replace('from neutral_dual_adapt_cfg import NeutralDualAdaptCfg, ARM','from neutral_dual_adapt_cfg import ARM\\nfrom dual_fixed_transition_cfg import FixedTransitionCfg as NeutralDualAdaptCfg')\n"
source=source.replace(needle,hook+needle)
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
