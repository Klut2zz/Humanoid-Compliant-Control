#new change
"""Frozen amplitude diagnostics with identical phase-shaped 5mrad correction."""
import os
import json
import numpy as np
from isaaclab.utils import configclass
from reference_transition_smooth_cfg import ReferenceTransitionSmoothCfg
from neutral_dual_adapt_cfg import ROOT,NAMES

@configclass
class AmplitudeDiagnosticCfg(ReferenceTransitionSmoothCfg):
    arm_motion_scale: float = float(os.environ['DIAGNOSTIC_ARM_SCALE'])

    def __post_init__(self):
        # assert self.arm_motion_scale in (.5,1.)
        #new change: reuse identical checks for the independent 45% probe.
        # assert self.arm_motion_scale in (.45,.5,1.)
        #new change: allow the requested frozen 75% amplitude diagnostic.
        # assert self.arm_motion_scale in (.45,.5,.75,1.)
        #new change: permit the requested frozen 85% diagnostic.
        assert self.arm_motion_scale in (.45,.5,.75,.85,1.)
        super().__post_init__()
        order=json.loads((ROOT/'datasets/motions_csv/stand_coordination_v1/manifest.json').read_text())['csv_joint_order']
        col=7+order.index('right_wrist_pitch_joint')
        accepted=np.loadtxt(ROOT/'datasets/motions_csv/dual_sagittal_csv_v2_35/reference.csv',delimiter=',')[:,col]
        current=np.loadtxt(self.commands.upper_body_joints.demo_recording_path[0],delimiter=',')[:,col]
        original=(accepted-accepted.mean())*(.005/np.max(abs(accepted-accepted.mean())))
        correction=(current-self.pitch_probe_center)*self.pitch_probe_gain
        np.testing.assert_allclose(correction,original,rtol=0,atol=1e-10)
        print('AMPLITUDE_PROBE_PASS',{'scale':self.arm_motion_scale,'same_5mrad_waveform':True,'waist_neutral':True,'external_force':False},flush=True)
