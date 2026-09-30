#new change
"""Independent whole-body force coordination; original dual controllers untouched."""
import copy
import torch
import gymnasium as gym
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.managers import ActionTerm,ActionTermCfg,ObservationTermCfg,ObservationGroupCfg,RewardTermCfg
from dual540_force_train_cfg import LOWER,ARM,NAMES,compliance_reward
from dual540_paired_events_cfg import Dual540PairedEventsCfg
from dual540_online_force_cfg import OnlineForceAction
from neutral_stand_geometry_cfg import state,balance_obs,LEG_IDS
from dual540_force_model import runner_config as arm_runner_config

TASK='Isaac-G1-Dual540-WholeBodyForceResidual-v0'
WAIST=[i for i,n in enumerate(NAMES) if n.startswith('waist_')]
LIMITS=[.02 if i in ARM else .01 for i in range(29)]
assert len(ARM)==14 and len(LEG_IDS)==12 and len(WAIST)==3
assert sorted(ARM+LEG_IDS+WAIST)==list(range(29))

@configclass
class MixedFrozenCfg(Dual540PairedEventsCfg):
    """Identical base for both zero-residual and reference-only audit branches."""
    zero_force_fraction:float=.25
    def __post_init__(self):
        super().__post_init__()
        parent=self.commands.upper_body_joints.class_type
        fraction=self.zero_force_fraction
        assert fraction in (0.,.25,1.)
        class MixedEvents(parent):
            def _sample_events(self,env_ids):
                super()._sample_events(env_ids)
                # Stratified environment cohorts; all environments still draw
                # their own event seeds. No shared RNG or reset coupling.
                zero=env_ids[(env_ids%4)<round(fraction*4)]
                self._sampled_peak_force_world[zero]=0
        self.commands.upper_body_joints.class_type=MixedEvents

def frozen_lower_obs(env):
    v=balance_obs(env)
    return torch.cat((v[:,:-29],env._dual_delivered[:,LEG_IDS]),dim=-1)

def wholebody_obs(env):
    r,c,ids,fids,feet,fe,com,forces=state(env)
    torso=r.body_names.index('torso_link')
    gravity=torch.zeros_like(r.data.root_lin_vel_w);gravity[:,2]=-1
    force=quat_apply_inverse(r.data.root_quat_w,c.force_world)/8
    left=force*(c.selected_body_ids==r.body_names.index('left_wrist_yaw_link'))[:,None]
    right=force*(c.selected_body_ids==r.body_names.index('right_wrist_yaw_link'))[:,None]
    return torch.cat((r.data.joint_pos[:,ids],r.data.joint_pos[:,ids]-c.dof_pos[:,1],
        r.data.joint_vel[:,ids]*.1,c.dof_vel[:,1]*.1,left,right,
        r.data.projected_gravity_b,r.data.root_lin_vel_b,r.data.root_ang_vel_b*.25,
        quat_apply_inverse(r.data.body_quat_w[:,torso],gravity),
        r.data.body_ang_vel_w[:,torso]*.25,r.data.body_lin_vel_w[:,torso],
        (com-feet.mean(1))/.1,forces.flatten(1)/400,fe.flatten(1)/.05,
        (r.data.root_pos_w[:,:2]-c.root_state['root_pos'][:,1,:2])/.1,
        r.data.body_lin_vel_w[:,fids].flatten(1),
        env._wb_residual/env._wb_limits,env._wb_release_age.clamp(max=2)[:,None]/2),dim=-1)

class WholeBodyAction(ActionTerm):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.base=OnlineForceAction(cfg.base_cfg,env)
        self.lower=torch.jit.load(str(LOWER),map_location=self.device).eval()
        for p in self.lower.parameters():p.requires_grad_(False)
        self.joint_ids=[self._asset.joint_names.index(n) for n in NAMES]
        assert len(set(self.joint_ids))==29
        self._raw=torch.zeros((self.num_envs,29),device=self.device)
        self._processed=self._raw.clone();self._previous=self._raw.clone()
        self._last_force=torch.zeros(self.num_envs,device=self.device)
        env._wb_limits=torch.tensor(LIMITS,device=self.device)
        env._wb_residual=self._processed
        env._wb_release_age=torch.full((self.num_envs,),3.,device=self.device)
    @property
    def action_dim(self):return 29
    @property
    def raw_actions(self):return self._raw
    @property
    def processed_actions(self):return self._processed
    def process_actions(self,actions):
        self._previous.copy_(self._processed)
        self._raw.copy_(actions);self._processed.copy_(torch.tanh(actions)*self._env._wb_limits)
        c=self._env.command_manager.get_term('upper_body_joints')
        force=c.force_world.norm(dim=-1)
        age=self._env._wb_release_age
        age.add_(self._env.step_dt).clamp_(max=3)
        age[(self._last_force>1e-4)&(force<=1e-4)]=0
        age[force>1e-4]=3
        self._last_force.copy_(force)
        with torch.no_grad():lower=self.lower(self._env.obs_buf['lower_policy'])
        self.base.process_actions(lower)
        # Preserve accepted history units, now including delivered corrections.
        self._env._dual_delivered[:,LEG_IDS]+=self._processed[:,LEG_IDS]/.30
        self._env._dual_delivered[:,ARM]+=self._processed[:,ARM]/.15
        self._env._dual_delivered[:,WAIST]=self._processed[:,WAIST]/.15
    def apply_actions(self):
        self.base.apply_actions()
        before=self._asset.data.joint_pos_target[:,self.joint_ids]
        self._asset.set_joint_position_target(before+self._processed,joint_ids=self.joint_ids)
    def reset(self,env_ids=None):
        self.base.reset(env_ids)
        ix=slice(None) if env_ids is None else env_ids
        self._raw[ix]=0;self._processed[ix]=0;self._previous[ix]=0;self._last_force[ix]=0
        self._env._wb_release_age[ix]=3

@configclass
class WholeBodyActionCfg(ActionTermCfg):
    class_type:type=WholeBodyAction
    base_cfg:object=None
@configclass
class FrozenLowerObs(ObservationGroupCfg):
    state=ObservationTermCfg(func=frozen_lower_obs)
    def __post_init__(self):self.enable_corruption=False;self.concatenate_terms=True
@configclass
class WholeBodyObs(ObservationGroupCfg):
    state=ObservationTermCfg(func=wholebody_obs)
    def __post_init__(self):self.enable_corruption=False;self.concatenate_terms=True

def coordination_reward(env,kind):
    r,c,ids,fids,feet,fe,com,forces=state(env)
    root_error=(r.data.root_pos_w[:,:2]-c.root_state['root_pos'][:,1,:2]).norm(dim=-1)
    speed=r.data.root_lin_vel_w[:,:2].norm(dim=-1)
    if kind=='root':return torch.exp(-torch.relu(root_error-.02).square()/.05**2)
    if kind=='recovery':
        gate=(env._wb_release_age<2)&(c.force_world.norm(dim=-1)<=1e-4)
        cost=(root_error/.04).square()+(speed/.05).square()+r.data.projected_gravity_b[:,:2].square().sum(-1)/.12**2
        return gate.float()*torch.exp(-cost)
    term=env.action_manager.get_term('joint_pos')
    if kind=='residual':return (term.processed_actions/env._wb_limits).square().mean(-1)
    if kind=='smooth':return ((term.processed_actions-term._previous)/env._wb_limits).square().mean(-1)
    raise ValueError(kind)

@configclass
class WholeBodyForceCfg(MixedFrozenCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos=WholeBodyActionCfg(asset_name='robot',base_cfg=copy.deepcopy(self.actions.joint_pos))
        self.observations.lower_policy=FrozenLowerObs()
        self.observations.policy=WholeBodyObs();self.observations.critic=FrozenLowerObs()
        for side in ('left','right'):
            for kind,weight in [('joint',1.),('wrist',2.)]:
                setattr(self.rewards,side+'_'+kind,RewardTermCfg(func=compliance_reward,weight=weight,params={'side':side,'kind':kind}))
        self.rewards.root=RewardTermCfg(func=coordination_reward,weight=3.,params={'kind':'root'})
        self.rewards.recovery=RewardTermCfg(func=coordination_reward,weight=2.,params={'kind':'recovery'})
        self.rewards.residual=RewardTermCfg(func=coordination_reward,weight=-.02,params={'kind':'residual'})
        self.rewards.action_rate=RewardTermCfg(func=coordination_reward,weight=-.02,params={'kind':'smooth'})

def runner_config(seed=42):
    cfg=arm_runner_config(seed)
    cfg['save_interval']=5
    return cfg

gym.register(id=TASK,entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
             kwargs={'env_cfg_entry_point':WholeBodyForceCfg})
