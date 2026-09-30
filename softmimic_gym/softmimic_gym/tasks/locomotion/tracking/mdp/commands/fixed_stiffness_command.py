# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Sub-module containing a command generator for reading augmented reference motions.
This command generator extends ReferenceCommand to handle CSV files that include
the original reference motion, an "adapted" motion, external forces/torques,
and forcefield metadata for reactive force computation.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

# Import the base class
from isaaclab.managers import CommandTerm
import isaaclab.utils.math as math_utils
from isaaclab.markers.visualization_markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import FixedStiffnessCommandCfg


class FixedStiffnessCommand(CommandTerm):
    """
    Command generator that samples motions from augmented AMASS-style datasets with forcefield info.

    This class extends `ReferenceCommand` to read from CSV files that contain four blocks of data:
    1. The original reference motion (e.g., from AMASS).
    2. An 'adapted' motion, which has the same format but is prefixed with 'adap_'.
    3. Force and Torque information, including the link ID and force/torque vectors.
    4. Forcefield metadata (stiffness, setpoint/origin, plane normal) for reactive force computation.

    The original reference motion is accessible, but by default, the main properties (`root_state`,
    `dof_pos`, etc.) are overridden to return the adapted motion and its derivatives. The new
    force/torque data are exposed through new properties. A config flag determines whether to
    use pre-computed feedforward forces or calculate them reactively from the forcefield data.
    """

    cfg: FixedStiffnessCommandCfg

    def __init__(self, cfg: FixedStiffnessCommand, env: ManagerBasedEnv):
        """Initialize the command generator.""" 
        
        super().__init__(cfg, env)

        self._stiffness = torch.ones((self.num_envs, 1), device=self.device) * cfg.desired_stiffness
        self._rotational_stiffness = torch.ones((self.num_envs, 1), device=self.device) * cfg.desired_rotational_stiffness
        self._ff_stiffness = torch.ones((self.num_envs, 1, 1), device=self.device) * cfg.forcefield_stiffness
        self._ff_rotational_stiffness = torch.ones((self.num_envs, 1, 1), device=self.device) * cfg.forcefield_rotational_stiffness

    def _update_command(self): pass
    def _resample_command(self, env_ids): pass
    def _update_metrics(self): pass

    @property
    def command(self) -> torch.Tensor:
        """Return the full command tensor."""
        return torch.zeros((self.num_envs, 1), device=self.device)

    @property
    def desired_stiffness(self) -> torch.Tensor:
        """Return the stiffness of the force field."""
        return self._stiffness[:, 0].clone()

    @property
    def desired_rotational_stiffness(self) -> torch.Tensor:
        """Return the rotational stiffness of the force field."""
        return self._rotational_stiffness[:, 0].clone()
    
    @property
    def forcefield_stiffness(self) -> torch.Tensor:
        """Alias for `ff_stiffness` for compatibility."""
        return self._ff_stiffness[:, 0, 0].clone()
    
    @property
    def forcefield_rotational_stiffness(self) -> torch.Tensor:
        """Alias for `ff_rotational_stiffness` for compatibility."""
        return self._ff_rotational_stiffness[:, 0, 0].clone()

