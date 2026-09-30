#new change
from pathlib import Path
import sys
root=Path('/home/noah/mingkai/general_softmimic');sys.path.insert(0,str(root/'scripts'))
path=root/'scripts/test_neutral_dual_adapt.py';source=path.read_text()
needle="s=s.replace('cfg=NeutralStandGeometryCfg();','cfg=NeutralDualAdaptCfg();')"
assert source.count(needle)==1
source=source.replace(needle,"s=s.replace('cfg=NeutralStandGeometryCfg();','cfg=NeutralDualAdaptCfg();')\ns=s.replace('from neutral_dual_adapt_cfg import NeutralDualAdaptCfg, ARM','from neutral_dual_adapt_cfg import ARM\\nfrom dual_sagittal_gate_cfg import Sagittal15Cfg as NeutralDualAdaptCfg')")
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
