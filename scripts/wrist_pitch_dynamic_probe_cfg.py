#new change
"""Small zero-reference-mean target probe; no actor/history/gain changes."""
import json
import numpy as np
from isaaclab.utils import configclass
from dual_sagittal_csv_v2_cfg import SagittalCsvV2_35Cfg
from neutral_dual_adapt_cfg import FrozenArmsAction, ROOT, NAMES

J=NAMES.index('right_wrist_pitch_joint')

class DynamicPitchAction(FrozenArmsAction):
    def apply_actions(self):
        super().apply_actions()
        c=self._env.command_manager.get_term('upper_body_joints')
        delta=(c.dof_pos[:,1,J]-self._env.cfg.pitch_probe_center)*self._env.cfg.pitch_probe_gain
        assert delta.abs().max()<0.00501
        self._env._pitch_probe_delta=delta.detach().clone()
        target=c.dof_pos[:,1,J]+self._env._dual_arm_raw[:,self.arm_ids.index(self._asset.joint_names.index(NAMES[J]))]*.15+delta
        self._asset.set_joint_position_target(target[:,None],joint_ids=[self._asset.joint_names.index(NAMES[J])])

@configclass
class WristPitchDynamicProbeCfg(SagittalCsvV2_35Cfg):
    def __post_init__(self):
        super().__post_init__()
        manifest=json.loads((ROOT/'datasets/motions_csv/stand_coordination_v1/manifest.json').read_text())
        data=np.loadtxt(self.commands.upper_body_joints.demo_recording_path[0],delimiter=',')
        ref=data[:,7+manifest['csv_joint_order'].index(NAMES[J])]
        # Full prerecorded reference, uniformly sampled: no rollout/future-state input.
        self.pitch_probe_center=float(ref.mean())
        self.pitch_probe_gain=float(0.005/np.max(np.abs(ref-ref.mean())))
        correction=(ref-self.pitch_probe_center)*self.pitch_probe_gain
        assert abs(correction.mean())<1e-12
        assert self.scene.robot.actuators['arms'].stiffness['.*_wrist_.*']==5
        assert self.scene.robot.actuators['arms'].damping['.*_wrist_.*']==1
        self.actions.joint_pos.class_type=DynamicPitchAction
        print('DYNAMIC_PITCH_PROBE',json.dumps({'center_rad':self.pitch_probe_center,'gain':self.pitch_probe_gain,'full_reference_mean_rad':float(correction.mean()),'range_rad':[float(correction.min()),float(correction.max())]}),flush=True)
