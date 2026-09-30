#new change
"""Fixed nominal Stand with online IK, wrist force, and trainable full legs.

Independent task: does not change any accepted model/task/physics settings.
"""
import copy
import json
import numpy as np
import torch
import gymnasium as gym
from isaaclab.utils import configclass
from isaaclab.managers import ActionTerm,ActionTermCfg,ObservationGroupCfg,ObservationTermCfg,RewardTermCfg
from dual540_recovery_window_cfg import RecoveryWindowCfg
from dual540_online_force_cfg import OnlineForceAction
from dual540_wholebody_force_cfg import wholebody_obs,ARM,WAIST,LOWER
from neutral_dual_adapt_cfg import ROOT,NAMES,UPPER
from neutral_stand_geometry_cfg import state,balance_obs,LEG_IDS

TASK='Isaac-G1-StandForceJointV1-v0'
CHECKPOINT540=LOWER.with_suffix('.pt')
NEUTRAL=ROOT/'datasets/motions_csv/neutral_stand_geometry_v1/neutral.csv'
UPPER_IDS=[i for i in range(29) if i not in LEG_IDS]
SCALES=[.30 if i in LEG_IDS else (.02 if i in ARM else .01) for i in range(29)]

def lower_obs(env):
    # Preserve exactly the pretrained 109D interface, including action history.
    value=balance_obs(env)
    return torch.cat((value[:,:-29],env._dual_delivered[:,LEG_IDS]),dim=-1)

def force_context(env):
    r,c,ids,fids,feet,fe,com,forces=state(env)
    value=wholebody_obs(env).clone()
    value[:,-1]=env._wb_release_age.clamp(max=5)/5
    mass=r.data.default_mass.to(env.device)
    com_velocity=(r.data.body_com_lin_vel_w*mass[:,:,None]).sum(1)/mass.sum(1)[:,None]
    # Explicitly distinguish root-link and whole-body COM velocities. Do not
    # silently replace the velocity in the pretrained lower observation.
    return torch.cat((value,r.data.root_link_lin_vel_w,com_velocity),dim=-1)

def anchor_mask(env):
    # Privileged training-only cohort flag; never an actor/critic input.
    return ((torch.arange(env.num_envs,device=env.device)%4)<round(env.cfg.zero_force_fraction*4)).float()[:,None]

class JointStandAction(ActionTerm):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.base=OnlineForceAction(cfg.base_cfg,env)
        self.teacher_lower=torch.jit.load(str(LOWER),map_location=self.device).eval()
        for parameter in self.teacher_lower.parameters():parameter.requires_grad_(False)
        self.joint_ids=[self._asset.joint_names.index(n) for n in NAMES]
        self._raw=torch.zeros((self.num_envs,29),device=self.device)
        self._processed=torch.zeros_like(self._raw);self._previous=torch.zeros_like(self._raw)
        self._upper_delta=torch.zeros_like(self._raw)
        self._had_force=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self._event_id=torch.full((self.num_envs,),-1,device=self.device,dtype=torch.long)
        env._wb_limits=torch.tensor(SCALES,device=self.device)
        env._wb_residual=torch.zeros_like(self._raw)
        env._wb_release_age=torch.full((self.num_envs,),10.,device=self.device)
    @property
    def action_dim(self):return 29
    @property
    def raw_actions(self):return self._raw
    @property
    def processed_actions(self):return self._processed
    def process_actions(self,actions):
        self._previous.copy_(self._processed);self._raw.copy_(actions)
        self._processed.copy_(torch.tanh(actions)*self._env._wb_limits)
        # Full pretrained leg action, NOT a +/-10mrad residual over frozen legs.
        self._processed[:,LEG_IDS]=.30*actions[:,LEG_IDS]
        self._upper_delta.copy_(self._processed);self._upper_delta[:,LEG_IDS]=0
        with torch.no_grad():old=self.teacher_lower(self._env.obs_buf['lower_policy'])
        self._env._wb_residual.copy_(self._upper_delta)
        self._env._wb_residual[:,LEG_IDS]=.30*(actions[:,LEG_IDS]-old)
        c=self._env.command_manager.get_term('upper_body_joints')
        force=c.force_world.norm(dim=-1)>1e-4
        age=self._env._wb_release_age
        age.add_(self._env.step_dt).clamp_(max=10.)
        released=(c._independent_event_index!=self._event_id)&(c._event_time<0)&self._had_force
        age[released]=0;self._had_force[released]=False
        self._had_force |= force;age[force]=10.
        self._event_id.copy_(c._independent_event_index)
        self.base.process_actions(actions[:,LEG_IDS])
        self._env._dual_delivered[:,ARM]+=self._upper_delta[:,ARM]/.15
        self._env._dual_delivered[:,WAIST]=self._upper_delta[:,WAIST]/.15
    def apply_actions(self):
        self.base.apply_actions()
        current=self._asset.data.joint_pos_target[:,self.joint_ids]
        self._asset.set_joint_position_target(current+self._upper_delta,joint_ids=self.joint_ids)
    def reset(self,env_ids=None):
        self.base.reset(env_ids)
        ix=slice(None) if env_ids is None else env_ids
        self._raw[ix]=0;self._processed[ix]=0;self._previous[ix]=0;self._upper_delta[ix]=0
        self._had_force[ix]=False;self._event_id[ix]=-1
        self._env._wb_residual[ix]=0;self._env._wb_release_age[ix]=10.

@configclass
class JointStandActionCfg(ActionTermCfg):
    class_type:type=JointStandAction
    base_cfg:object=None
@configclass
class LowerObs(ObservationGroupCfg):
    state=ObservationTermCfg(func=lower_obs)
    def __post_init__(self):self.enable_corruption=False;self.concatenate_terms=True
@configclass
class ContextObs(ObservationGroupCfg):
    state=ObservationTermCfg(func=force_context)
    def __post_init__(self):self.enable_corruption=False;self.concatenate_terms=True
@configclass
class AnchorObs(ObservationGroupCfg):
    state=ObservationTermCfg(func=anchor_mask)
    def __post_init__(self):self.enable_corruption=False;self.concatenate_terms=True

def joint_reward(env,kind):
    r,c,ids,fids,feet,fe,com,forces=state(env)
    if kind=='speed':return torch.exp(-r.data.root_link_lin_vel_w.square().sum(-1)/.0225)
    if kind=='residual':return (env._wb_residual/env._wb_limits).square().mean(-1)
    if kind=='recovery':
        quiet=(env._wb_release_age<5)&(c.force_world.norm(dim=-1)<=1e-4)
        x=(r.data.root_pos_w[:,:2]-c.root_state['root_pos'][:,1,:2]).norm(dim=-1)
        v=r.data.root_link_lin_vel_w[:,:2].norm(dim=-1)
        cost=(x/.04).square()+(v/.05).square()+r.data.projected_gravity_b[:,:2].square().sum(-1)/.12**2
        return quiet.float()*torch.exp(-cost)
    raise ValueError(kind)

@configclass
class StandForceJointCfg(RecoveryWindowCfg):
    def __post_init__(self):
        super().__post_init__()
        c=self.commands.upper_body_joints
        values=np.loadtxt(NEUTRAL,delimiter=',')
        np.testing.assert_allclose(values,np.repeat(values[:1],len(values),axis=0),atol=1e-11,rtol=0)
        c.demo_recording_path=[str(NEUTRAL)];c.demo_start_range=[[0.,0.]]
        c.demo_random_init=False;c.clip_length=None
        # No nominal wrist waveform in a fixed-pose task; original dynamic
        # task/correction remains untouched in its own configuration.
        self.pitch_probe_gain=0.
        self.actions.joint_pos=JointStandActionCfg(asset_name='robot',base_cfg=copy.deepcopy(self.actions.joint_pos.base_cfg))
        self.observations.lower_policy=LowerObs()
        self.observations.policy=ContextObs();self.observations.critic=ContextObs()
        self.observations.anchor_mask=AnchorObs()
        self.rewards.speed=RewardTermCfg(func=joint_reward,weight=2.,params={'kind':'speed'})
        self.rewards.recovery=RewardTermCfg(func=joint_reward,weight=2.,params={'kind':'recovery'})
        self.rewards.residual=RewardTermCfg(func=joint_reward,weight=-.02,params={'kind':'residual'})
        assert not getattr(self.events,'write_online_reference_kinematic_state',None)
        print('STAND_FORCE_JOINT_CONFIG',json.dumps({'nominal':str(NEUTRAL),'full_leg_policy':True,
          'force_N':[2,4],'recovery_wait_s':[3,5],'recovery_reward_s':5,'zero_fraction':self.zero_force_fraction}),flush=True)

gym.register(id=TASK,entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
             kwargs={'env_cfg_entry_point':StandForceJointCfg})
