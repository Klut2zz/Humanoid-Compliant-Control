#new change
"""Independent 14D force-conditioned residual; accepted controllers are immutable."""
import copy
import os
os.environ.setdefault('DUAL_ISOLATION', 'combined')
os.environ.setdefault('DUAL_FORCE_MAX', '4')
os.environ.setdefault('DIAGNOSTIC_ARM_SCALE', '1.0')
import torch
import gymnasium as gym
from isaaclab.utils import configclass
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from isaaclab.utils.math import quat_apply_inverse
from dual540_paired_events_cfg import Dual540PairedEventsCfg
from dual540_online_force_cfg import OnlineForceAction
from neutral_dual_adapt_cfg import ARM, ROOT, NAMES
from neutral_stand_geometry_cfg import balance_obs, LEG_IDS
from softmimic_gym.assets.robots.g1_compat import G1_SOFTMIMIC_KEYPOINT_BODY_NAMES

TASK = 'Isaac-G1-Dual540-ForceAware-ArmsResidual-v0'
LOWER = ROOT/'logs/rsl_rl/g1_neutral_stand_geometry_v1/2026-09-16_17-53-05_hold100_replay_lower_stability_v1/model_540.jit'

def lower_observation(env):
    # balance_obs originally ends in 12 lower actions. The outer action is now
    # 14 residuals; reconstruct exactly the accepted lower history interface.
    value = balance_obs(env)
    return torch.cat((value[:, :-14], env._dual_delivered[:, LEG_IDS]), dim=-1)

def force_observation(env):
    r = env.scene['robot']; c = env.command_manager.get_term('upper_body_joints')
    ids = c.control_joint_ids
    q = r.data.joint_pos[:, ids][:, ARM]
    dq = r.data.joint_vel[:, ids][:, ARM]
    # Simulated applied force, not future event timing nor an actual sensor.
    force_b = quat_apply_inverse(r.data.root_quat_w, c.force_world)/8.
    sided = []
    for side in ('left', 'right'):
        bid = r.body_names.index(side+'_wrist_yaw_link')
        sided.append(force_b*(c.selected_body_ids == bid)[:, None])
    return torch.cat((q, q-c.dof_pos[:, 1, ARM], dq*.1,
                      c.dof_vel[:, 1, ARM]*.1, *sided,
                      r.data.projected_gravity_b, r.data.root_lin_vel_b,
                      r.data.root_ang_vel_b*.25, env._force_residual/.02), dim=-1)

class ForceResidualAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.base = OnlineForceAction(cfg.base_cfg, env)
        self.lower = torch.jit.load(str(LOWER), map_location=self.device).eval()
        for p in self.lower.parameters(): p.requires_grad_(False)
        self._raw = torch.zeros((self.num_envs, 14), device=self.device)
        self._processed = self._raw.clone()
        env._force_residual = self._processed

    @property
    def action_dim(self): return 14
    @property
    def raw_actions(self): return self._raw
    @property
    def processed_actions(self): return self._processed

    def process_actions(self, actions):
        self._raw.copy_(actions)
        self._processed.copy_(.02*torch.tanh(actions))
        with torch.no_grad(): lower = self.lower(self._env.obs_buf['lower_policy'])
        self.base.process_actions(lower)
        # Preserve legacy observation units (.15 rad / arm action).
        self._env._dual_delivered[:, ARM] += self._processed/.15

    def apply_actions(self):
        self.base.apply_actions()
        # Base rewrites nominal+actor+5mrad targets each substep, so no accumulation.
        target = self._asset.data.joint_pos_target[:, self.base.arm_ids]+self._processed
        self._asset.set_joint_position_target(target, joint_ids=self.base.arm_ids)

    def reset(self, env_ids=None):
        self.base.reset(env_ids)
        ix = slice(None) if env_ids is None else env_ids
        self._raw[ix] = 0; self._processed[ix] = 0

@configclass
class ResidualActionCfg(ActionTermCfg):
    class_type: type = ForceResidualAction
    base_cfg: object = None

@configclass
class LowerObservation(ObservationGroupCfg):
    state = ObservationTermCfg(func=lower_observation)
    def __post_init__(self):
        self.enable_corruption = False; self.concatenate_terms = True

@configclass
class ForceObservation(ObservationGroupCfg):
    state = ObservationTermCfg(func=force_observation)
    def __post_init__(self):
        self.enable_corruption = False; self.concatenate_terms = True

def compliance_reward(env, side, kind):
    r = env.scene['robot']; c = env.command_manager.get_term('upper_body_joints')
    if kind == 'joint':
        ids = [i for i in ARM if NAMES[i].startswith(side+'_')]
        error = r.data.joint_pos[:, c.control_joint_ids][:, ids]-c.dof_pos[:, 1, ids]
        return torch.exp(-error.square().mean(-1)/.15**2)
    name = side+'_wrist_yaw_link'
    bid = r.body_names.index(name); kid = list(G1_SOFTMIMIC_KEYPOINT_BODY_NAMES).index(name)
    return torch.exp(-(r.data.body_pos_w[:, bid]-c.keypoints[:, 1, kid, :3]).square().sum(-1)/.03**2)

@configclass
class Dual540ForceTrainCfg(Dual540PairedEventsCfg):
    def __post_init__(self):
        super().__post_init__()
        assert os.environ['DUAL_ISOLATION'] == 'combined'
        assert self.arm_motion_scale == 1.
        base = copy.deepcopy(self.actions.joint_pos)
        self.actions.joint_pos = ResidualActionCfg(asset_name='robot', base_cfg=base)
        self.observations.lower_policy = LowerObservation()
        self.observations.policy = ForceObservation()
        self.observations.critic = LowerObservation()
        for side in ('left', 'right'):
            for kind, weight in [('joint', 1.), ('wrist', 2.)]:
                setattr(self.rewards, side+'_'+kind, RewardTermCfg(func=compliance_reward,
                        weight=weight, params={'side': side, 'kind': kind}))

gym.register(id=TASK, entry_point='isaaclab.envs:ManagerBasedRLEnv',
             disable_env_checker=True, kwargs={'env_cfg_entry_point': Dual540ForceTrainCfg})
