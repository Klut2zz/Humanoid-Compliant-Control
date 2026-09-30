"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
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
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import matplotlib.pyplot as plt
from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import softmimic_gym.tasks  # noqa: F401
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    
    # env_cfg.scene.terrain.terrain_generator.num_rows = 1
    # env_cfg.scene.terrain.terrain_generator.num_cols = 1
    # env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    
    # turn off recorders for training
    env_cfg.recorders = None
    if hasattr(env_cfg, "recorders") and env_cfg.recorders is not None:
        data_log_dir = os.path.join(log_dir, "step_data")
        env_cfg.recorders.dataset_export_dir_path = data_log_dir
        env_cfg.recorders.dataset_filename = "simulation_data.csv"


    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
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

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.jit"
    )
    export_policy_as_onnx(
        ppo_runner.alg.policy, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # reset environment
    env.reset()
    # env.env.env.command_manager.compute(dt=env.env.env.step_dt)
    obs, extras = env.get_observations()
    # amp_obs = extras["observations"]["amp"]
    # print(obs)
    timestep = 0
    amp_reward=[]
    amp_obs_list=[]
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # actions+= torch.normal(0, 0.1, size=(1,29), device=agent_cfg.device)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
            # print(obs)
            # assert False
        if args_cli.video:
            timestep += 1
            if timestep % 25 == 0:
                print(timestep)
                print(rewards)
            
            # amp_obs = infos["observations"]["amp"]
            # cur_amp_reward = ppo_runner.alg.discriminator.predict_amp_reward(amp_obs, rewards, ppo_runner.alg.amp_normalizer)
            # amp_reward.append(cur_amp_reward[2].squeeze().cpu())
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                # plt.plot(amp_reward)
                # plt.title("Discriminator output (1=dataset, -1=policy)")
                # plt.savefig("line_plot.png", dpi=300, bbox_inches='tight')
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
