#new change
from pathlib import Path
path=Path(__file__).with_name('test_dual540_online_force.py')
source=path.read_text().replace('from dual540_online_force_cfg import Dual540OnlineForceCfg','from dual540_basic_audit_cfg import Dual540BasicAuditCfg').replace('from dual540_online_force_cfg import capture_online','from dual540_basic_audit_cfg import capture_audit as capture_online')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
