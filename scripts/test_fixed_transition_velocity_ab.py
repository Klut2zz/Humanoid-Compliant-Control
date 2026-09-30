#new change
"""Only select reset-velocity axes; fixed controllers/physics unchanged."""
import os
from pathlib import Path
mode=os.environ['VELOCITY_MODE']
axes={'none':(), 'linear':('x','y'), 'angular':('roll','pitch')}
#new change
axes.update({axis:(axis,) for axis in ('x','y','roll','pitch')})
assert mode in axes
path=Path(__file__).resolve().with_name('test_dual_fixed_transition.py')
source=path.read_text()
needle='source=source.replace(needle,hook+needle)'
assert source.count(needle)==1
extra="s=s.replace(\"for key in ('x','y','roll','pitch'):\",\"for key in "+repr(axes[mode])+":\")\n"
extra+="s=s.replace(\"        assert e.action_manager.total_action_dim==12\",\"        print('RESET_VELOCITY_AXES',repr(cfg.events.reset_robot.params['velocity_range']),flush=True)\\n        assert e.action_manager.total_action_dim==12\")\n"
source=source.replace(needle,'hook+='+repr(extra)+'\n'+needle)
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
