#new change
from pathlib import Path
path=Path(__file__).resolve().with_name('test_multipose45_lower_balance.py')
source=path.read_text()
old='from multipose45_lower_balance_cfg import MultiPose45LowerBalanceCfg'
assert source.count(old)==1
source=source.replace(old,'from multipose45_single_pose_eval_cfg import SinglePose45EvalCfg')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
