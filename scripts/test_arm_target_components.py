#new change
import os
from pathlib import Path
os.environ['SAGITTAL_SCALE']='25'
path=Path(__file__).resolve().with_name('test_dual_sagittal_csv_v2.py')
source=path.read_text()
needle='source=source.replace(needle,hook+needle)'
assert source.count(needle)==1
extra="s=s.replace('        alive=torch.ones','        from arm_target_components import install\\n        install(e)\\n        alive=torch.ones')\n"
extra+="s=s.replace('        steady_mean=','        e._report_arm_components()\\n        steady_mean=')\n"
source=source.replace(needle,'hook+='+repr(extra)+'\n'+needle)
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
