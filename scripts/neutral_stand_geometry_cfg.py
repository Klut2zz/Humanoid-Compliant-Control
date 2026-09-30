#new change
"""Independent scratch 12DOF neutral balance task. No legacy actor/adapters."""
from pathlib import Path
import copy
import gymnasium as gym
import torch
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from softmimic_gym.tasks.locomotion.tracking.config.g1.general_tracking_v1 import G1GeneralTrackingV1EnvCfg
from softmimic_gym.tasks.locomotion.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1GeneralTrackingV1RunnerCfg
from softmimic_gym.tasks.locomotion.tracking.mdp.actions import UnifiedReferenceResidualJointPositionAction, UnifiedReferenceResidualJointPositionActionCfg
from softmimic_gym.assets.robots.g1_compat import G1_SOFTMIMIC_29_JOINT_NAMES, G1_SOFTMIMIC_KEYPOINT_BODY_NAMES
import softmimic_gym.tasks.locomotion.tracking.mdp as mdp

ROOT=Path('/home/noah/mingkai/general_softmimic')
TASK='Isaac-G1-NeutralStand-CorrectGeometry-12DOF-v0'
NAMES=list(G1_SOFTMIMIC_29_JOINT_NAMES)
LEG_IDS=[i for i,n in enumerate(NAMES) if any(s in n for s in ('hip_','knee','ankle_'))]
UPPER_IDS=[i for i in range(29) if i not in LEG_IDS]
FEET=['left_ankle_roll_link','right_ankle_roll_link']

def state(env):
    r=env.scene['robot'];c=env.command_manager.get_term('upper_body_joints')
    ids=c.control_joint_ids
    fids=[r.body_names.index(n) for n in FEET]
    k=[list(G1_SOFTMIMIC_KEYPOINT_BODY_NAMES).index(n) for n in FEET]
    feet=r.data.body_pos_w[:,fids]
    foot_error=feet[:,:,:2]-c.keypoints[:,1,k,:2]
    mass=r.data.default_mass.to(env.device)
    com=(r.data.body_com_pos_w*mass[:,:,None]).sum(1)/mass.sum(1)[:,None]
    sensor=env.scene['contact_forces']
    # Original implementation queried PhysX prim_paths on every reward call:
    # forces=sensor.data.net_forces_w[:,[sensor.body_names.index(n) for n in FEET]]
    #new change
    # Body ordering is immutable after initialization; cache only integer IDs,
    # never dynamic contacts or state (which must refresh after every reset).
    if not hasattr(env,'_neutral_foot_sensor_ids'):
        sensor_names=sensor.body_names
        env._neutral_foot_sensor_ids=[sensor_names.index(n) for n in FEET]
    forces=sensor.data.net_forces_w[:,env._neutral_foot_sensor_ids]
    return r,c,ids,fids,feet,foot_error,com,forces

def balance_obs(env):
    r,c,ids,fids,feet,fe,com,forces=state(env)
    torso=r.body_names.index('torso_link')
    from isaaclab.utils.math import quat_apply_inverse
    gravity=torch.zeros_like(r.data.root_lin_vel_w);gravity[:,2]=-1
    return torch.cat([
        r.data.projected_gravity_b,r.data.root_ang_vel_b*.25,r.data.root_lin_vel_b,
        quat_apply_inverse(r.data.body_quat_w[:,torso],gravity),r.data.body_ang_vel_w[:,torso]*.25,
        r.data.root_pos_w[:,:2]-c.root_state['root_pos'][:,1,:2],
        r.data.root_pos_w[:,2:3],(com-feet.mean(1)),fe.flatten(1),
        r.data.body_lin_vel_w[:,fids].flatten(1),forces.flatten(1)/400.,
        (forces[:,:,2]>5).float(),r.data.joint_pos[:,ids]-c.dof_pos[:,1],
        r.data.joint_vel[:,ids]*.05,env.action_manager.action],dim=1)

def balance_reward(env,kind):
    r,c,ids,fids,feet,fe,com,forces=state(env)
    if kind=='upright':return torch.exp(-r.data.projected_gravity_b[:,:2].square().sum(1)/.04)
    if kind=='root':return torch.exp(-(r.data.root_pos_w[:,:2]-c.root_state['root_pos'][:,1,:2]).square().sum(1)/.0025)
    if kind=='speed':return torch.exp(-r.data.root_lin_vel_w.square().sum(1)/.0225)
    if kind=='feet':return torch.exp(-fe.square().sum((1,2))/.0018)
    if kind=='com':return torch.exp(-(com[:,:2]-feet.mean(1)[:,:2]).square().sum(1)/.0064)
    if kind=='contact':return (forces[:,:,2]>10).float().mean(1)
    if kind=='pose':return (r.data.joint_pos[:,ids][:,LEG_IDS]-c.dof_pos[:,1,LEG_IDS]).square().mean(1)
    if kind=='slide':return (r.data.body_lin_vel_w[:,fids,:2].square().sum(-1)*(forces[:,:,2]>10)).sum(1).clamp(max=2)
    if kind=='height':return torch.exp(-(r.data.root_pos_w[:,2]-(c.root_state['root_pos'][:,1,2]-.005)).square()/.0025)
    raise ValueError(kind)

def containment(env):
    r,c,ids,fids,feet,fe,com,forces=state(env)
    return ((r.data.root_pos_w[:,:2]-c.root_state['root_pos'][:,1,:2]).norm(dim=1)>.3)|(fe.norm(dim=-1).amax(1)>.2)|(r.data.root_pos_w[:,2]<.55)

class LegAction(UnifiedReferenceResidualJointPositionAction):
    def apply_actions(self):
        super().apply_actions()
        c=self._env.command_manager.get_term('upper_body_joints')
        ids=[self._asset.joint_names.index(NAMES[i]) for i in UPPER_IDS]
        self._asset.set_joint_position_target(c.dof_pos[:,1,UPPER_IDS],joint_ids=ids)

@configclass
class LegActionCfg(UnifiedReferenceResidualJointPositionActionCfg):
    class_type:type=LegAction

@configclass
class BalanceObs(ObservationGroupCfg):
    state=ObservationTermCfg(func=balance_obs)
    def __post_init__(self):
        self.enable_corruption=False
        self.concatenate_terms=True
        self.history_length=1

@configclass
class NeutralStandGeometryCfg(G1GeneralTrackingV1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        c=self.commands.upper_body_joints
        c.demo_recording_path=[str(ROOT/'datasets/motions_csv/neutral_stand_geometry_v1/neutral.csv')]
        c.demo_start_range=[[0.,0.]]
        c.debug_vis=False
        self.actions.joint_pos=LegActionCfg(asset_name='robot',joint_names=[NAMES[i] for i in LEG_IDS],preserve_order=True,scale=.30,offset=0.,use_default_offset=False,command_name='upper_body_joints',reference_joint_ids=LEG_IDS,reference_time_index=1)
        self.observations.policy=BalanceObs()
        self.observations.critic=BalanceObs()
        for n in list(vars(self.events)):
            if not n.startswith('_') and n not in ('physics_material','reset_robot'):setattr(self.events,n,None)
        reset=self.events.reset_robot.params
        reset['pose_range']={k:(0.,0.) for k in reset['pose_range']}
        reset['velocity_range']={k:(0.,0.) for k in reset['velocity_range']}
        reset['jpos_range']=(0.,0.)
        reset['jvel_range']=(0.,0.)
        self.scene.robot.actuators['feet'].stiffness={'left_ankle_pitch_joint':20.,'right_ankle_pitch_joint':30.,'left_ankle_roll_joint':20.,'right_ankle_roll_joint':20.}
        for n in list(vars(self.rewards)):
            if not n.startswith('_'):setattr(self.rewards,n,None)
        for kind,weight in [('upright',4.),('root',3.),('speed',2.),('feet',2.),('com',1.),('contact',.5),('height',1.),('pose',-.5),('slide',-1.)]:
            setattr(self.rewards,kind,RewardTermCfg(func=balance_reward,weight=weight,params={'kind':kind}))
        self.rewards.action_rate=RewardTermCfg(func=mdp.action_rate_l2,weight=-.02)
        self.rewards.angular_speed=RewardTermCfg(func=mdp.ang_vel_xy_l2,weight=-.2)
        self.rewards.failure=RewardTermCfg(func=mdp.is_terminated,weight=-20.)
        self.terminations.containment=TerminationTermCfg(func=containment)
        self.terminations.bad_orientation.params['limit_angle']=.5
        self.episode_length_s=30.
        self.recorders=None

@configclass
class NeutralStandGeometryRunnerCfg(G1GeneralTrackingV1RunnerCfg):
    experiment_name='g1_neutral_stand_geometry_v1'
    num_steps_per_env=48
    save_interval=25
    max_iterations=300
    def __post_init__(self):
        self.policy.actor_hidden_dims=[256,256,128]
        self.policy.critic_hidden_dims=[256,256,128]
        self.policy.init_noise_std=.15
        self.algorithm.learning_rate=3e-4
        self.algorithm.num_learning_epochs=4
        self.algorithm.entropy_coef=.001

gym.register(id=TASK,entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,kwargs={'env_cfg_entry_point':NeutralStandGeometryCfg,'rsl_rl_cfg_entry_point':NeutralStandGeometryRunnerCfg})
