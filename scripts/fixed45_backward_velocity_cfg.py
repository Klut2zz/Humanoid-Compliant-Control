#new change
"""Frozen recovery probe: fixed world-X initial velocity, no angular noise."""
from isaaclab.utils import configclass
from fixed45_prefailure_cfg import Fixed45PrefailureCfg

@configclass
class Fixed45BackwardVelocityCfg(Fixed45PrefailureCfg):
    def __post_init__(self):
        super().__post_init__()
        velocity=self.events.reset_robot.params['velocity_range']
        assert all(tuple(v)==(0.,0.) for v in velocity.values())
        velocity['x']=(-.02,-.02)
        print('BACKWARD_VELOCITY_PROBE',velocity,flush=True)
