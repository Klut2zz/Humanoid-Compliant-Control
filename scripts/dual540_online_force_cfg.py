#new change
"""Frozen dual-controller online IK bridge; no kinematic state writes."""
import copy
import os
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate_inverse
from dual_amplitude_50_100_cfg import AmplitudeDiagnosticCfg
from neutral_dual_adapt_cfg import FrozenArmsAction,NAMES
from softmimic_gym.tasks.locomotion.tracking.config.g1.general_tracking_v1 import G1StandCoordinationForceEventABSmooth2to4NEnvCfg
from softmimic_gym.tasks.locomotion.tracking.mdp.commands.online_compliance_reference_command import OnlineComplianceReferenceCommand
from softmimic_gym.tasks.locomotion.tracking.mdp.events import apply_forcefield_force_torque

class OnlineForceAction(FrozenArmsAction):
    def apply_actions(self):
        super().apply_actions()
        c=self._env.command_manager.get_term('upper_body_joints')
        j=NAMES.index('right_wrist_pitch_joint')
        # Preserve nominal 5 mrad correction; online IK must not amplify it.
        delta=(c._dof_pos[:,1,j]-self._env.cfg.pitch_probe_center)*self._env.cfg.pitch_probe_gain
        assert delta.abs().max()<.00501
        self._env._pitch_probe_delta=delta.detach().clone()
        index=self._asset.joint_names.index(NAMES[j])
        target=c.dof_pos[:,1,j]+self._env._dual_arm_raw[:,self.arm_ids.index(index)]*.15+delta
        self._asset.set_joint_position_target(target[:,None],joint_ids=[index])
        # World wrench direction held fixed; recompute body coordinates each physics substep.
        c.forcefield_forces_b=quat_rotate_inverse(self._asset.data.body_quat_w[c._env_ids,c.selected_body_ids],c.force_world)
        apply_forcefield_force_torque(self._env,None,'upper_body_joints')

@configclass
class Dual540OnlineForceCfg(AmplitudeDiagnosticCfg):
    def __post_init__(self):
        super().__post_init__()
        nominal=self.commands.upper_body_joints
        old=G1StandCoordinationForceEventABSmooth2to4NEnvCfg()
        command=copy.deepcopy(old.commands.upper_body_joints)
        for key,value in vars(nominal).items():
            if key!='class_type':setattr(command,key,copy.deepcopy(value))
        force=float(os.environ.get('DUAL_FORCE_MAX','0'))
        assert force in (0.,4.)
        command.force_range_n=(0.,0.) if force==0 else (2.,4.)
        command.force_envelope_profile='half_cosine'
        command.physical_force_enabled=force>0
        command.debug_vis=False
        base=nominal.class_type
        class Bridge(OnlineComplianceReferenceCommand,base):
            def _build_online_reference(self):
                super()._build_online_reference()
                if not torch.any(self._peak_joint_delta!=0):return
                #new change: FK each adapted horizon pose, not a stale Cartesian delta from event onset.
                from isaaclab.utils.math import quat_rotate,quat_from_matrix,quat_mul,quat_conjugate
                n,h,_=self._dof_pos.shape
                root=self._root_rot[:,:,[3,0,1,2]].reshape(-1,4)
                for side in ('left','right'):
                    chain=self._reference_serial_chains[side]
                    ids=self._reference_chain_action_ids[side]
                    nominal=chain.forward_kinematics(self._dof_pos[:,:,ids].reshape(n*h,-1),end_only=False)
                    adapted=chain.forward_kinematics(self._online_dof_pos[:,:,ids].reshape(n*h,-1),end_only=False)
                    for name,k in zip((side+'_shoulder_yaw_link',side+'_elbow_link',side+'_wrist_yaw_link'),self._reference_chain_keypoint_ids[side]):
                        x=nominal[name].get_matrix();y=adapted[name].get_matrix()
                        delta=quat_rotate(root,y[:,:3,3]-x[:,:3,3]).reshape(n,h,3)
                        self._online_keypoints[:,:,k,:3]=self._keypoints[:,:,k,:3]+delta
                        dr=quat_from_matrix(y[:,:3,:3]@x[:,:3,:3].transpose(-1,-2))
                        dw=quat_mul(quat_mul(root,dr),quat_conjugate(root))
                        self._online_keypoint_rotations[:,:,k]=quat_mul(dw,self._keypoint_rotations[:,:,k].reshape(-1,4)).reshape(n,h,4)
            def _sample_events(self,env_ids):
                if not hasattr(self,'_force_rng'):
                    self._force_rng=torch.Generator(device=self.device).manual_seed(int(self._env.cfg.seed)+91871)
                original=self.torch_gen
                self.torch_gen=self._force_rng
                try:super()._sample_events(env_ids)
                finally:self.torch_gen=original
        command.class_type=Bridge
        self.commands.upper_body_joints=command
        self.actions.joint_pos.class_type=OnlineForceAction
        # Same flag on F=0 and F>0 branches, explicitly audited by equivalence test.
        # self.sim.physx.enable_external_forces_every_iteration=True
        #new change: preserve the accepted model540 solver settings; apply wrench each simulation substep.
        assert not getattr(self.events,'write_online_reference_kinematic_state',None)
        print('DUAL540_ONLINE_CONFIG',{'force':command.force_range_n,'profile':command.force_envelope_profile,'upper':'existing frozen1375+supervised arms','physics_substep_wrench':True},flush=True)

def capture_online(e):
    c=e.command_manager.get_term('upper_body_joints')
    def arr(x):return x.detach().cpu().numpy().copy()
    return {k:arr(v) for k,v in dict(force_world=c.force_world,force_body_id=c.selected_body_ids,
        force_active=c.active_force_mask,event_time=c._event_time,ramp_time=c._ramp_time,hold_time=c._hold_time,
        ik_scale=c.ik_feasible_scale,nominal_ref=c._dof_pos[:,1],online_delta=c.dof_pos[:,1]-c._dof_pos[:,1],
        wrist_body_pos=e.scene['robot'].data.body_pos_w[:,c._candidate_body_ids],
        keypoint_ref=c.keypoints[:,1],target_displacement=c.target_displacement_world).items()}
