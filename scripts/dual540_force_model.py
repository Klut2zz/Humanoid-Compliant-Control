#new change
"""Zero deterministic residual with exploration confined to new arm outputs."""
import torch
from rsl_rl.models import MLPModel

class ZeroResidualModel(MLPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

def runner_config(seed=42):
    return dict(seed=seed, num_steps_per_env=48, save_interval=2, logger='tensorboard',
        obs_groups={'actor': ['policy'], 'critic': ['policy', 'critic']},
        actor=dict(class_name='dual540_force_model:ZeroResidualModel', hidden_dims=[128, 64],
                   activation='elu', obs_normalization=False,
                   distribution_cfg=dict(class_name='rsl_rl.modules:GaussianDistribution', init_std=.05)),
        critic=dict(class_name='rsl_rl.models:MLPModel', hidden_dims=[128, 64],
                    activation='elu', obs_normalization=False),
        algorithm=dict(class_name='rsl_rl.algorithms:PPO', learning_rate=2e-5,
            schedule='fixed', num_learning_epochs=1, num_mini_batches=4, clip_param=.05,
            entropy_coef=0., desired_kl=.002, gamma=.99, lam=.95, value_loss_coef=1.,
            max_grad_norm=1., use_clipped_value_loss=True, rnd_cfg=None, symmetry_cfg=None))
