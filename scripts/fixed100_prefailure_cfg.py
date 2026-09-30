#new change
"""Independent held full-amplitude pre-failure posture; original configs retained."""
from pathlib import Path
path=Path(__file__).resolve().with_name('fixed45_prefailure_cfg.py')
source=path.read_text()
changes={
 'Fixed45PrefailureCfg':'Fixed100PrefailureCfg',
 'assert self.arm_motion_scale==.45':'assert self.arm_motion_scale==1.0',
 'model406_scale045_seed0_500.npz':'model425_scale100_seed0_500.npz',
 "assert not d['terminated'][:251,18].any()":"assert not d['terminated'][:101,11].any()",
 "d['ref'][250,18]":"d['ref'][100,11]",
 "d['phase_time'][250,18]":"d['phase_time'][100,11]",
 'fixed45_env18_step250':'fixed100_env11_step100',
 "'env':18,'step':250":"'env':11,'step':100",
 'FIXED45_PASS':'FIXED100_PASS',
}
for old,new in changes.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)
exec(compile(source,str(path),'exec'),globals())
