# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import pathlib
import sys

# new change
# Keep general_softmimic self-contained even when the environment's editable
# softmimic-gym installation points to the archived project.
_LOCAL_EXTENSION_ROOT = pathlib.Path(__file__).absolute().parents[2] / "softmimic_gym"
sys.path.insert(0, str(_LOCAL_EXTENSION_ROOT))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--load_pulse_run", type=str, default=None, help="Run name to load PULSE model from.")
parser.add_argument("--load_teacher_run", type=str, default=None, help="Run name to load teacher model from.")
parser.add_argument("--load_teacher_checkpoint", type=str, default="model_.*.pt", help="Checkpoint to load teacher model from.")
parser.add_argument("--csv_file", type=str, default=None, help="Path to CSV file for motion reference.")
parser.add_argument("--csv_start", type=float, default=0.0, help="Start time for the CSV file.")
parser.add_argument("--csv_end", type=float, default=0.0, help="End time for the CSV file (0 means no limit).")
# new change
parser.add_argument(
    "--reset_optimizer",
    action="store_true",
    default=False,
    help="Load actor/critic and iteration while keeping the new task's fresh optimizer.",
)
# new change
parser.add_argument(
    "--reset_action_std",
    type=float,
    default=None,
    help="After loading a checkpoint, reset the state-independent Gaussian action std.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
# new change
import importlib.metadata as metadata
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.io import dump_pickle, dump_yaml
# new change
import pickle
from isaaclab.utils.io import dump_yaml


# new change
def dump_pickle(filename, data):
    """IsaacLab 5.1-compatible replacement for the removed helper."""
    with open(filename, "wb") as file:
        pickle.dump(data, file)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
# new change
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


# Import extensions to set up environment tasks
import softmimic_gym.tasks  # noqa: F401
# new change
from softmimic_gym.rsl_rl_extensions.policy_utils import (
    bilateral_structured_arm_adapter_zero_summary,
    configure_bilateral_structured_arm_adapter_finetune,
    configure_actor_output_row_finetune,
    # new change
    configure_online_compliance_adapter_finetune,
    # new change
    configure_online_compliance_arm_only_finetune,
    # new change
    configure_online_compliance_waist_only_finetune,
    # new change
    configure_proximal_arm_waist_adapter_finetune,
    # new change
    configure_structured_action_std,
    # new change
    configure_stationary_support_adapter_finetune,
    configure_waist_support_adapter_finetune,
    load_checkpoint_with_zero_adapter_compat,
    # new change
    online_compliance_adapter_zero_summary,
    support_adapter_zero_summary,
    waist_adapter_zero_summary,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# new change
def _reset_gaussian_action_std(policy, action_std: float) -> dict[str, float]:
    """Reset exploration without replacing the optimizer-visible parameter."""

    if not math.isfinite(action_std) or action_std <= 0.0:
        raise ValueError(f"action_std must be finite and positive, got {action_std!r}.")
    distribution = getattr(policy, "distribution", None)
    if distribution is None:
        raise TypeError("The loaded policy has no output distribution to reset.")
    if hasattr(distribution, "std_param"):
        parameter = distribution.std_param
        before = parameter.detach().clone()
        with torch.no_grad():
            parameter.fill_(action_std)
    elif hasattr(distribution, "log_std_param"):
        parameter = distribution.log_std_param
        before = parameter.detach().exp().clone()
        with torch.no_grad():
            parameter.fill_(math.log(action_std))
    else:
        raise TypeError("Expected a Gaussian distribution with std_param or log_std_param.")
    return {
        "before_mean": float(before.mean().item()),
        "before_min": float(before.min().item()),
        "before_max": float(before.max().item()),
        "after": float(action_std),
    }


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    # new change
    # Convert the repository's legacy policy block to the actor/critic blocks
    # required by the rsl_rl version bundled with IsaacLab 5.1.
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.load_teacher_run = args_cli.load_teacher_run
    agent_cfg.load_teacher_checkpoint = args_cli.load_teacher_checkpoint

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # turn off recorders for training
    env_cfg.recorders = None
    
    # Apply the csv file if specified
    if args_cli.csv_file is not None:
        if args_cli.csv_file == "LAFAN1_ALL":
            env_cfg.commands.upper_body_joints.demo_start_range = [[args_cli.csv_start, args_cli.csv_start] for _ in env_cfg.commands.upper_body_joints.demo_recording_path]
            # env_cfg.commands.upper_body_joints.demo_random_init = False
        else:
            env_cfg.commands.upper_body_joints.demo_recording_path = [args_cli.csv_file]
            env_cfg.commands.upper_body_joints.demo_start_range = [[args_cli.csv_start, args_cli.csv_start]]
            # env_cfg.commands.upper_body_joints.demo_random_init = False
        if args_cli.csv_end > 0.0:
            env_cfg.commands.upper_body_joints.resampling_time_range = [args_cli.csv_end-args_cli.csv_start, args_cli.csv_end-args_cli.csv_start]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    
    env.reset()
    env.env.env.command_manager.compute(dt=env.env.env.step_dt)

    import rsl_rl.runners.on_policy_runner
    from softmimic_gym.rsl_rl_extensions.runners import JitSavingOnPolicyRunner as OnPolicyRunner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    # runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        # runner.load(resume_path)
        # new change
        # A curriculum checkpoint may contain an optimizer whose adaptive PPO
        # learning rate has already decayed to its floor.  Loading only the
        # networks and iteration preserves the policy while allowing the new
        # stage to start from the optimizer settings in its runner config.
        if args_cli.reset_optimizer:
            load_summary = load_checkpoint_with_zero_adapter_compat(
                runner,
                resume_path,
                load_cfg={
                    "actor": True,
                    "critic": True,
                    "optimizer": False,
                    "iteration": True,
                    "rnd": False,
                },
            )
        else:
            load_summary = load_checkpoint_with_zero_adapter_compat(
                runner, resume_path
            )
        # new change
        if load_summary["legacy_bootstrap"]:
            print(
                "[GENERAL-TRACKING][adapter-bootstrap] "
                f"missing_keys={len(load_summary['missing_actor_keys'])}",
                flush=True,
            )

        # new change
        optimizer_lrs = [
            float(group["lr"]) for group in runner.alg.optimizer.param_groups
        ]
        print(
            "[GENERAL-TRACKING][resume] "
            f"reset_optimizer={args_cli.reset_optimizer} optimizer_lrs={optimizer_lrs}"
        )

        # new change
        if args_cli.reset_action_std is not None:
            std_summary = _reset_gaussian_action_std(
                runner.alg.get_policy(), args_cli.reset_action_std
            )
            print(
                "[GENERAL-TRACKING][action_std] "
                f"before_mean={std_summary['before_mean']:.6f} "
                f"before_min={std_summary['before_min']:.6f} "
                f"before_max={std_summary['before_max']:.6f} "
                f"after={std_summary['after']:.6f}"
            )

    # new change
    if args_cli.reset_action_std is not None and not agent_cfg.resume:
        raise ValueError("--reset_action_std requires --resume True and a checkpoint.")

    # new change
    # StationarySafeFineTune protects balance by freezing the actor feature
    # extractor and masking every unselected physical output row.  The critic
    # remains trainable and is never part of the deployed JIT policy.
    actor_output_finetune_action_ids = getattr(
        agent_cfg, "actor_output_finetune_action_ids", None
    )
    if actor_output_finetune_action_ids is not None:
        row_summary = configure_actor_output_row_finetune(
            runner.alg.get_policy(), actor_output_finetune_action_ids
        )
        print(
            "[GENERAL-TRACKING][actor-output-rows] "
            f"trainable={list(row_summary['trainable_action_ids'])} "
            f"frozen={list(row_summary['frozen_action_ids'])} "
            f"parameter_values={row_summary['trainable_parameter_values']}",
            flush=True,
        )

    # new change
    if getattr(agent_cfg, "bilateral_structured_arm_adapter_only", False):
        policy = runner.alg.get_policy()
        zero_summary = bilateral_structured_arm_adapter_zero_summary(policy)
        adapter_summary = configure_bilateral_structured_arm_adapter_finetune(policy)
        print(
            "[GENERAL-TRACKING][bilateral-structured-adapter] "
            f"zero_weight_max={zero_summary['output_weight_abs_max']:.9f} "
            f"zero_bias_max={zero_summary['output_bias_abs_max']:.9f} "
            f"left_params={adapter_summary['left_trainable_values']} "
            f"right_params={adapter_summary['right_trainable_values']} "
            f"left_limits={adapter_summary['left_limits']} "
            f"right_limits={adapter_summary['right_limits']}",
            flush=True,
        )

    # new change
    if getattr(agent_cfg, "proximal_arm_waist_adapter_only", False):
        policy = runner.alg.get_policy()
        waist_zero = waist_adapter_zero_summary(policy)
        coordination_summary = configure_proximal_arm_waist_adapter_finetune(policy)
        print(
            "[GENERAL-TRACKING][proximal-arm-waist-adapter] "
            f"waist_zero_weight_max={waist_zero['output_weight_abs_max']:.9f} "
            f"waist_zero_bias_max={waist_zero['output_bias_abs_max']:.9f} "
            f"left_proximal_params={coordination_summary['left_proximal_trainable_values']} "
            f"right_proximal_params={coordination_summary['right_proximal_trainable_values']} "
            f"waist_params={coordination_summary['waist_trainable_values']} "
            f"waist_limits={coordination_summary['waist_limits']}",
            flush=True,
        )

    # new change
    if getattr(agent_cfg, "waist_support_adapter_only", False):
        policy = runner.alg.get_policy()
        waist_zero = waist_adapter_zero_summary(policy)
        support_zero = support_adapter_zero_summary(policy)
        coordination_summary = configure_waist_support_adapter_finetune(policy)
        print(
            "[GENERAL-TRACKING][waist-support-adapter] "
            f"waist_zero_weight_max={waist_zero['output_weight_abs_max']:.9f} "
            f"waist_zero_bias_max={waist_zero['output_bias_abs_max']:.9f} "
            f"support_zero_weight_max={support_zero['output_weight_abs_max']:.9f} "
            f"support_zero_bias_max={support_zero['output_bias_abs_max']:.9f} "
            f"waist_params={coordination_summary['waist_trainable_values']} "
            f"support_params={coordination_summary['support_trainable_values']} "
            f"waist_limits={coordination_summary['waist_limits']} "
            f"support_limits={coordination_summary['support_limits']} "
            f"support_action_ids={coordination_summary['support_action_ids']}",
            flush=True,
        )

    # new change
    if getattr(agent_cfg, "stationary_support_adapter_only", False):
        policy = runner.alg.get_policy()
        support_zero = support_adapter_zero_summary(policy)
        support_summary = configure_stationary_support_adapter_finetune(policy)
        print(
            "[GENERAL-TRACKING][stationary-support-adapter] "
            f"zero_weight_max={support_zero['output_weight_abs_max']:.9f} "
            f"zero_bias_max={support_zero['output_bias_abs_max']:.9f} "
            f"support_params={support_summary['support_trainable_values']} "
            f"support_limits={support_summary['support_limits']} "
            f"support_action_ids={support_summary['support_action_ids']}",
            flush=True,
        )

    # new change
    if getattr(agent_cfg, "online_compliance_adapter_only", False):
        policy = runner.alg.get_policy()
        zero_summary = online_compliance_adapter_zero_summary(policy)
        compliance_summary = configure_online_compliance_adapter_finetune(policy)
        print(
            "[GENERAL-TRACKING][online-compliance-adapter] "
            f"zero_weight_max={zero_summary['output_weight_abs_max']:.9f} "
            f"zero_bias_max={zero_summary['output_bias_abs_max']:.9f} "
            f"trainable={compliance_summary['trainable_values']} "
            f"action_ids={compliance_summary['action_ids']} "
            f"limits={compliance_summary['limits']}",
            flush=True,
        )

    # new change
    if getattr(agent_cfg, "online_compliance_arm_only", False):
        policy = runner.alg.get_policy()
        zero_summary = online_compliance_adapter_zero_summary(policy)
        compliance_summary = configure_online_compliance_arm_only_finetune(policy)
        print(
            "[GENERAL-TRACKING][online-compliance-arm-only] "
            f"zero_weight_max={zero_summary['output_weight_abs_max']:.9f} "
            f"zero_bias_max={zero_summary['output_bias_abs_max']:.9f} "
            f"trainable={compliance_summary['trainable_values']} "
            f"frozen_waist={compliance_summary['frozen_waist_values']} "
            f"heads={compliance_summary['trainable_heads']} "
            f"action_ids={compliance_summary['action_ids']} "
            f"limits={compliance_summary['limits']}",
            flush=True,
        )

    # new change
    if getattr(agent_cfg, "online_compliance_waist_only", False):
        policy = runner.alg.get_policy()
        zero_summary = online_compliance_adapter_zero_summary(policy)
        coordination_summary = configure_online_compliance_waist_only_finetune(
            policy, waist_limit_rad=0.002
        )
        print(
            "[GENERAL-TRACKING][online-compliance-waist-only] "
            f"preload_zero_weight_max={zero_summary['output_weight_abs_max']:.9f} "
            f"preload_zero_bias_max={zero_summary['output_bias_abs_max']:.9f} "
            f"trainable={coordination_summary['trainable_values']} "
            f"frozen_arms={coordination_summary['frozen_arm_values']} "
            f"waist_limit_rad={coordination_summary['waist_limit_rad']:.6f} "
            f"action_ids={coordination_summary['action_ids']}",
            flush=True,
        )

    # new change
    structured_exploration_ids = getattr(
        agent_cfg, "structured_exploration_action_ids", None
    )
    if structured_exploration_ids is not None:
        std_summary = configure_structured_action_std(
            runner.alg.get_policy(),
            structured_exploration_ids,
            active_std=float(agent_cfg.structured_exploration_active_std),
            frozen_std=float(agent_cfg.structured_exploration_frozen_std),
        )
        print(
            "[GENERAL-TRACKING][structured-exploration] "
            f"active_action_ids={std_summary['active_action_ids']} "
            f"active_std={std_summary['active_std']:.7f} "
            f"frozen_std={std_summary['frozen_std']:.7f}",
            flush=True,
        )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # export policy to onnx/jit
    # export_policy_as_jit(
    #     runner.alg.policy, runner.obs_normalizer, path=log_dir, filename="policy.pt"
    # )
    # export_policy_as_onnx(
    #     runner.alg.policy, normalizer=runner.obs_normalizer, path=log_dir, filename="policy.onnx"
    # )
    # new change
    runner.export_policy_to_jit(log_dir, filename="policy.pt")
    runner.export_policy_to_onnx(log_dir, filename="policy.onnx")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
