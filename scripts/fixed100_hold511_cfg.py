#new change
"""Independent 100% plateau pose, from model511 env104 step175; no physics edits."""
from pathlib import Path
path=Path(__file__).with_name('fixed45_prefailure_cfg.py')
source=path.read_text()
changes={
 'Fixed45PrefailureCfg':'Fixed100Hold511Cfg',
 'assert self.arm_motion_scale==.45':'assert self.arm_motion_scale==1.0',
 'model406_scale045_seed0_500.npz':'model511_scale100_seed0_500.npz',
 "assert not d['terminated'][:251,18].any()":"assert not d['terminated'][:176,104].any()\n            assert np.max(np.abs(d['ref_vel'][175,104]))<1e-6",
 "d['ref'][250,18]":"d['ref'][175,104]",
 "d['phase_time'][250,18]":"d['phase_time'][175,104]",
 'fixed45_env18_step250':'fixed100_model511_env104_step175',
 "'env':18,'step':250":"'env':104,'step':175",
 'FIXED45_PASS':'FIXED100_HOLD511_PASS',
}
for old,new in changes.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)
exec(compile(source,str(path),'exec'),globals())
