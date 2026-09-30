#new change
from pathlib import Path
path=Path(__file__).with_name('test_multipose45_lower_balance.py')
source=path.read_text().replace('from multipose45_lower_balance_cfg import MultiPose45LowerBalanceCfg','from hard_pose_stability_cfg import HardPoseStabilityCfg')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
