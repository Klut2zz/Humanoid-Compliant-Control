#new change
"""Train only the existing 12DOF lower network with frozen legacy arms."""
import copy
import torch
import gymnasium as gym
from isaaclab.utils import configclass
from neutral_stand_geometry_cfg import ROOT,NAMES,LEG_IDS,LegAction,NeutralStandGeometryCfg,NeutralStandGeometryRunnerCfg
from softmimic_gym.tasks.locomotion.tracking.config.g1.general_tracking_v1 import G1StandCoordinationAccuracyV2EnvCfg
from softmimic_gym.assets.robots.g1_compat import G1_SOFTMIMIC_BILATERAL_ARM_ACTION_IDS
ARM=list(G1_SOFTMIMIC_BILATERAL_ARM_ACTION_IDS)
UPPER=ROOT/'logs/rsl_rl/g1_stand_coordination_v1/2026-08-21_10-59-05_stand_coordination_accuracy_v2_alpha050_supervised_validation/model_1375.jit'
TASK='Isaac-G1-NeutralStand-FrozenArms-LowerAdapt-v0'

def delivered_actions(env):
    return getattr(env,'_dual_delivered',torch.zeros((env.num_envs,29),device=env.device))

class FrozenArmsAction(LegAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.upper=torch.jit.load(str(UPPER),map_location=self.device).eval()
        for p in self.upper.parameters():p.requires_grad_(False)
        self.arm_ids=[self._asset.joint_names.index(NAMES[i]) for i in ARM]
        self._env._dual_delivered=torch.zeros((self.num_envs,29),device=self.device)
        self._env._dual_arm_raw=torch.zeros((self.num_envs,len(ARM)),device=self.device)

    def process_actions(self,actions):
        # Read previous-step history before replacing it with delivered actions.
        obs=self._env.obs_buf
        with torch.no_grad():
            raw=self.upper(torch.cat([obs['legacy'],obs['left_arm_adapter'],obs['right_arm_adapter']],dim=1))
        self._env._dual_arm_raw=raw[:,ARM].detach()
        self._env._dual_delivered.zero_()
        self._env._dual_delivered[:,LEG_IDS]=actions.detach()
        self._env._dual_delivered[:,ARM]=raw[:,ARM]
        super().process_actions(actions)

    def apply_actions(self):
        super().apply_actions()
        c=self._env.command_manager.get_term('upper_body_joints')
        self._asset.set_joint_position_target(c.dof_pos[:,1,ARM]+self._env._dual_arm_raw*.15,joint_ids=self.arm_ids)

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self._env,'_dual_delivered'):
            self._env._dual_delivered[env_ids if env_ids is not None else slice(None)]=0
            self._env._dual_arm_raw[env_ids if env_ids is not None else slice(None)]=0

@configclass
class NeutralDualAdaptCfg(NeutralStandGeometryCfg):
    def __post_init__(self):
        super().__post_init__()
        old=G1StandCoordinationAccuracyV2EnvCfg()
        assert str(old.scene.robot.actuators['arms'])==str(self.scene.robot.actuators['arms'])
        self.observations.legacy=copy.deepcopy(old.observations.policy)
        self.observations.legacy.enable_corruption=False
        self.observations.legacy.actions.func=delivered_actions
        self.observations.legacy.actions.params={}
        self.observations.left_arm_adapter=copy.deepcopy(old.observations.left_arm_adapter)
        self.observations.right_arm_adapter=copy.deepcopy(old.observations.right_arm_adapter)
        self.actions.joint_pos.class_type=FrozenArmsAction

@configclass
class NeutralDualAdaptRunnerCfg(NeutralStandGeometryRunnerCfg):
    save_interval=10
    max_iterations=50
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate=1e-4
        self.algorithm.schedule='fixed'
        self.algorithm.clip_param=.1
        self.algorithm.num_learning_epochs=2
        self.algorithm.entropy_coef=0.
        self.algorithm.desired_kl=.005

gym.register(id=TASK,entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,kwargs={'env_cfg_entry_point':NeutralDualAdaptCfg,'rsl_rl_cfg_entry_point':NeutralDualAdaptRunnerCfg})
