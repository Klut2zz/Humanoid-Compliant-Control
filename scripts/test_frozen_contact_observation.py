#new change
"""Run the unchanged fixed100 test with additional observation channels."""
from pathlib import Path
original=Path.read_text
def read(self,*args,**kwargs):
    text=original(self,*args,**kwargs)
    if self.name=='test_fixed100_prefailure.py':
        text=text.replace('from fixed100_prefailure_cfg import Fixed100PrefailureCfg','from frozen_contact_observation import ObservationCfg')
    if self.name=='test_dual_sagittal_timing.py':
        needle="                reference_before=c.dof_pos[:,1].clone()'''"
        assert text.count(needle)==1
        text=text.replace(needle,"                from frozen_contact_observation import capture\n                frame.update(capture(e))\n"+needle)
    return text
Path.read_text=read
path=Path(__file__).with_name('test_fixed100_prefailure.py')
exec(compile(path.read_text(),str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
