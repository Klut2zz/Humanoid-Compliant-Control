#new change
"""Frozen reference-only / physical-only branches; keep latent event generation."""
import os
import torch
from isaaclab.utils import configclass
from dual540_online_force_cfg import Dual540OnlineForceCfg,OnlineForceAction,capture_online

MODE=os.environ['DUAL_ISOLATION']
assert MODE in ('reference_only','physical_only')

class IsolatedAction(OnlineForceAction):
    def apply_actions(self):
        super().apply_actions()
        c=self._env.command_manager.get_term('upper_body_joints')
        if MODE=='reference_only':
            # Clear buffered physical wrench before scene.write_data_to_sim; retain force context.
            ids=c.sorted_keypoint_ids.tolist()
            zero=torch.zeros((self.num_envs,len(ids),3),device=self.device)
            self._asset.set_external_force_and_torque(zero,zero,body_ids=ids)

@configclass
class Dual540IsolationCfg(Dual540OnlineForceCfg):
    def __post_init__(self):
        super().__post_init__()
        base=self.commands.upper_body_joints.class_type
        if MODE=='physical_only':
            class NominalDeliveredReference(base):
                @property
                def dof_pos(self):return self._dof_pos
                @property
                def dof_vel(self):return self._dof_vel
                @property
                def keypoints(self):return self._keypoints
                @property
                def keypoint_rotations(self):return self._keypoint_rotations
                @property
                def adapted_dof_pos(self):return self._dof_pos
                @property
                def adapted_dof_vel(self):return self._dof_vel
                @property
                def adapted_keypoints(self):return self._keypoints
                @property
                def adapted_keypoint_rotations(self):return self._keypoint_rotations
            self.commands.upper_body_joints.class_type=NominalDeliveredReference
        self.actions.joint_pos.class_type=IsolatedAction
        print('ISOLATION_MODE',MODE,'latent IK/events and force context retained',flush=True)

def capture_isolation(e):
    out=capture_online(e)
    c=e.command_manager.get_term('upper_body_joints')
    out['latent_online_delta']=(c._online_dof_pos[:,1]-c._dof_pos[:,1]).detach().cpu().numpy().copy()
    out['physical_force_world']=(torch.zeros_like(c.force_world) if MODE=='reference_only' else c.force_world).detach().cpu().numpy().copy()
    return out
