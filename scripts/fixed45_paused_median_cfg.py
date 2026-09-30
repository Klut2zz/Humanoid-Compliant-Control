#new change
"""Held actual pause posture nearest the 142 newly-failed environments' median."""
from pathlib import Path
path=Path(__file__).resolve().with_name('fixed45_prefailure_cfg.py')
source=path.read_text()
changes={
 'Fixed45PrefailureCfg':'Fixed45PausedMedianCfg',
 'model406_scale045_seed0_500.npz':'model406_scale045_branch250_pause_500.npz',
 "assert not d['terminated'][:251,18].any()":"assert not d['terminated'][:281,1].any()",
 "d['ref'][250,18]":"d['ref'][280,1]",
 "d['phase_time'][250,18]":"d['phase_time'][280,1]",
 'fixed45_env18_step250':'fixed45_paused_median_env1_step280',
 "'env':18,'step':250":"'env':1,'step':280",
}
for old,new in changes.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)
exec(compile(source,str(path),'exec'),globals())
