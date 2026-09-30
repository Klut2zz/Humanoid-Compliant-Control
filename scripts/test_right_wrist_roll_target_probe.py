#new change
import os
from pathlib import Path
os.environ['SAGITTAL_SCALE']='25'
bias=float(os.environ['ROLL_TARGET_BIAS'])
assert bias in (0.,.01,.02)
path=Path(__file__).resolve().with_name('test_dual_sagittal_csv_v2.py')
source=path.read_text()
needle='source=source.replace(needle,hook+needle)'
assert source.count(needle)==1
extra="s=s.replace('        alive=torch.ones','        from right_wrist_roll_target_probe import install\\n        install(e,"+repr(bias)+")\\n        alive=torch.ones')\n"
extra+="s=s.replace('                    expected[:,ARM]+=e._dual_arm_raw*.15','                    expected[:,ARM]+=e._dual_arm_raw*.15\\n                    expected[:,NAMES.index(\"right_wrist_roll_joint\")]+="+repr(bias)+"')\n"
source=source.replace(needle,'hook+='+repr(extra)+'\n'+needle)
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
