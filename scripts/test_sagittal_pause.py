#new change
"""Run with PAUSE_MODE=baseline/replay/pause, distinct TRACE_PATH; no training."""
import os
from pathlib import Path
os.environ['SAGITTAL_SCALE']='25'
assert os.environ['PAUSE_MODE'] in ('baseline','replay','pause')
path=Path(__file__).resolve().with_name('test_dual_sagittal_timing.py')
source=path.read_text()
needle='                r,c,ids,fids,feet,fe,com,forces=state(e)'
assert source.count(needle)==1
source=source.replace(needle,'                from sagittal_pause_reference import branch_step\n                branch_step(e,c,obs,step)\n'+needle)
needle="                    'ref':c.dof_pos[:,1].cpu().numpy().copy(),"
assert source.count(needle)==1
source=source.replace(needle,needle+"\n                    'ref_vel':c.dof_vel[:,1].cpu().numpy().copy(),\n                    'phase_time':c._relative_timestamp.cpu().numpy().copy(),\n                    'keypoint_ref':c.keypoints[:,1].cpu().numpy().copy(),")
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
