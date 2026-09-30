#new change
from pathlib import Path
import os
original=Path.read_text
def read(self,*args,**kwargs):
    s=original(self,*args,**kwargs)
    if self.name=='test_dual_amplitude_50_100.py':
        s=s.replace('from dual_amplitude_50_100_cfg import AmplitudeDiagnosticCfg','from dual540_online_force_cfg import Dual540OnlineForceCfg')
    if self.name=='test_dual_sagittal_timing.py':
        needle="                reference_before=c.dof_pos[:,1].clone()'''"
        assert s.count(needle)==1
        s=s.replace(needle,"                from dual540_online_force_cfg import capture_online\n                frame.update(capture_online(e))\n"+needle)
    return s
Path.read_text=read
path=Path(__file__).with_name('record_frozen_dual_reference.py' if os.environ.get('DUAL_VIDEO')=='1' else 'test_dual_amplitude_50_100.py')
exec(compile(path.read_text(),str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
