import gymnasium as gym
import numpy as np
import sapien
import time

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode


import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union
import torch

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PandaPutSpoonOnTableClothInScene-v1"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb+segmentation"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_ee_delta_pose"
    """Control mode"""

    render_mode: str = "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = None
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = 121
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

def main(args: Args):
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
        reconfiguration_freq=1, 
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
    env: BaseEnv = gym.make(
        args.env_id,
        **env_kwargs
    )
    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=gym_utils.find_max_episode_steps_value(env))

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    episode_id = 0 
    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True, episode_id = torch.tensor([episode_id])))
    if args.seed is not None and env.action_space is not None:
            env.action_space.seed(args.seed[0])
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()

    reset_init_time = time.time()
    object_pose_list=[]
    container_pose_list=[]

    object_pose_list.append(env.unwrapped.object["actor"][0].pose)
    container_pose_list.append(env.unwrapped.object["actor"][1].pose)

    try:
        while True:
            if time.time()-reset_init_time>2.0:
                episode_id += 1
                reset_init_time = time.time()
                obs, _ = env.reset(seed=args.seed, options=dict(episode_id=torch.tensor([episode_id])))
                object_pose_list.append(env.unwrapped.object["actor"][0].pose)
                container_pose_list.append(env.unwrapped.object["actor"][1].pose)

            if "bi" in args.control_mode:
                action = np.array([0.0,0,0,0,0,0,0,   0,0,0,0,0,0,0],dtype=np.float32) # env.action_space.sample() if env.action_space is not None else None
            else:
                action = np.array([0,0,0,0,0,0,1],dtype=np.float32) # env.action_space.sample() if env.action_space is not None else None
            obs, reward, terminated, truncated, info = env.step(action)
            # if verbose:
            #     print("reward", reward)
            #     print("terminated", terminated)
            #     print("truncated", truncated)
            #     print("info", info)
            if args.render_mode is not None:
                env.render()
            if args.render_mode is None or args.render_mode != "human":
                if (terminated | truncated).any():
                    break
    except KeyboardInterrupt:
        # print list
        print("Object poses:")
        for idx in range(len(object_pose_list)):
            print(f"object_{idx}       {object_pose_list[idx].p}")
        print("Container poses:")
        for idx in range(len(container_pose_list)):
            print(f"container_{idx}       {container_pose_list[idx].p}")
        env.close()
    env.close()

    if record_dir:
        print(f"Saving video to {record_dir}")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)