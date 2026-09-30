#new change
"""Inference-only single-joint stiffness probe; production config unchanged."""
import copy
import re
from isaaclab.utils import configclass
from dual_sagittal_csv_v2_cfg import SagittalCsvV2_35Cfg
from neutral_dual_adapt_cfg import NAMES

@configclass
class WristPitchKp10Cfg(SagittalCsvV2_35Cfg):
    def __post_init__(self):
        super().__post_init__()
        arms=copy.deepcopy(self.scene.robot.actuators['arms'])
        original=dict(arms.stiffness)
        assert original['.*_wrist_.*']==5.0
        updated=dict(original)
        del updated['.*_wrist_.*']
        updated['left_wrist_.*|right_wrist_(roll|yaw)_joint']=5.0
        updated['right_wrist_pitch_joint']=10.0
        for name in NAMES:
            before=[v for k,v in original.items() if re.fullmatch(k,name)]
            after=[v for k,v in updated.items() if re.fullmatch(k,name)]
            assert after==([10.0] if name=='right_wrist_pitch_joint' else before), name
        assert arms.damping['.*_wrist_.*']==1.0
        arms.stiffness=updated
        self.scene.robot.actuators['arms']=arms
        print('WRIST_KP10_PROBE: only right_wrist_pitch_joint stiffness 5 -> 10; damping unchanged',flush=True)
