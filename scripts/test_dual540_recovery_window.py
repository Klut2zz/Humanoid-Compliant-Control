#new change
"""Independent long-recovery recorder. Original acceptance scripts preserved."""
from pathlib import Path
path=Path(__file__).with_name('test_dual540_force_checkpoint.py')
source=path.read_text()
changes={
 'from dual540_force_train_cfg import TASK,Dual540ForceTrainCfg':
 'from dual540_recovery_window_cfg import TASK,RecoveryWindowCfg',
 'from dual540_force_model import runner_config':
 'from dual540_wholebody_force_cfg import runner_config',
 'cfg=Dual540ForceTrainCfg();cfg.scene.num_envs=args.num_envs':
 "assert args.force_max==4\n    cfg=RecoveryWindowCfg(zero_force_fraction=0.);cfg.scene.num_envs=args.num_envs",
 'residual=.02*torch.tanh(action),force=c.force_world,':
 'residual=e._wb_limits*torch.tanh(action),force=c.force_world,',
}
for old,new in changes.items():
 assert source.count(old)==1,old
 source=source.replace(old,new)
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
