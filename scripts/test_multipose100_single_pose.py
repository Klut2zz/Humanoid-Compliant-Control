#new change
from pathlib import Path
path=Path(__file__).resolve().with_name('test_multipose100_lower_balance.py')
source=path.read_text()
old='from multipose100_lower_balance_cfg import MultiPose100LowerBalanceCfg'
assert source.count(old)==1
source=source.replace(old,'from multipose100_single_pose_eval_cfg import SinglePose100EvalCfg')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
