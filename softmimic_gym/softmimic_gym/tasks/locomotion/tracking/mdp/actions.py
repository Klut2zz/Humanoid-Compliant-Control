"""Reference-relative action term for the general 29DOF controller."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass


# new change
class UnifiedReferenceResidualJointPositionAction(JointPositionAction):
    """Compose every controlled joint as ``q_ref + scaled actor residual``."""

    cfg: "UnifiedReferenceResidualJointPositionActionCfg"

    def __init__(self, cfg: "UnifiedReferenceResidualJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._reference_joint_ids = torch.as_tensor(
            cfg.reference_joint_ids, device=self.device, dtype=torch.long
        )
        if self._reference_joint_ids.numel() != self.action_dim:
            raise ValueError(
                "Unified reference-residual action requires one reference id per action: "
                f"actions={self.action_dim}, reference_ids={self._reference_joint_ids.numel()}."
            )

    def _current_reference(self) -> torch.Tensor:
        command = self._env.command_manager.get_term(self.cfg.command_name)
        reference = command.dof_pos[:, self.cfg.reference_time_index]
        reference = reference.index_select(1, self._reference_joint_ids)
        if reference.shape != self.raw_actions.shape:
            raise ValueError(
                "Action/reference dimensions disagree: "
                f"actions={tuple(self.raw_actions.shape)}, reference={tuple(reference.shape)}."
            )
        return reference

    def process_actions(self, actions: torch.Tensor) -> None:
        # Preserve IsaacLab's raw-action bookkeeping, then deliberately replace
        # the default-pose offset with the live motion reference.
        JointPositionAction.process_actions(self, actions)
        self._processed_actions = self._current_reference() + self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


# new change
@configclass
class UnifiedReferenceResidualJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the gate-free general 29DOF action term."""

    class_type: type[ActionTerm] = UnifiedReferenceResidualJointPositionAction
    command_name: str = MISSING
    reference_joint_ids: list[int] = MISSING
    reference_time_index: int = 1
