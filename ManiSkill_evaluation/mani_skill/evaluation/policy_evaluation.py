from collections import defaultdict
import json
import os
import signal
import time
import numpy as np
from typing import Annotated, Optional
import torch
from pathlib import Path
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video
from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles, euler_angles_to_matrix
signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c

import gymnasium as gym
import numpy as np
from PIL import Image
from mani_skill.envs.sapien_env import BaseEnv
import tyro
from dataclasses import dataclass
from mani_skill import MANISKILL_ROOT_DIR
import pickle

@dataclass
class Args:
    """
    This is a script to evaluate policies on real2sim environments.
    """

    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "TabletopPickPlaceEnv-v1"
    """The environment ID of the task you want to simulate. """

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    shader: str = "default"  # default, rt

    num_envs: int = 1
    """Number of environments to run. With more than 1 environment the environment will use the GPU backend 
    which runs faster enabling faster large-scale evaluations. Note that the overall behavior of the simulation
    will be slightly different between CPU and GPU backends."""

    num_episodes: int = 100
    """Number of episodes to run and record evaluation metrics over"""

    max_episode_len: int = 100
    """Max episode length"""

    record_dir: str = os.path.join(MANISKILL_ROOT_DIR,"videos")
    """The directory to save videos and results"""

    model: Optional[str] = "diffusion_policy" 
    """The model to evaluate on the given environment. If not given, random actions are sampled."""

    ckpt_path: str = ""
    """Checkpoint path for models."""

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Seed the model and environment. Default seed is 0"""

    info_on_video: bool = False
    """Whether to write info text onto the video"""

    save_video: bool = False
    """Whether to save videos"""

    save_data: bool = False
    """Whether to save collect data"""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Control mode"""

    obs_normalize_params_path: str = ""

    container_name: Optional[str] = None

    object_name: Optional[str] = None

    is_delta: bool = False
    
    is_table_green: bool = False
    
    initial_eps_count: int = 0
    """Initial episode count, used to avoid overwriting previous data when resuming evaluation."""

def main():
    args = tyro.cli(Args)
    if args.seed is not None:
        np.random.seed(args.seed)

    # Setup up the policy inference model
    print(f"model is {args.model}")

    env_kwargs = dict(
        num_envs=args.num_envs,
        obs_mode="rgb+segmentation",
        control_mode=args.control_mode,
        sim_backend="gpu",
        sim_config={
            "sim_freq": 1000,
            "control_freq": 25,
        },
        max_episode_steps=args.max_episode_len,
        sensor_configs={"shader_pack": args.shader},
        is_table_green = args.is_table_green,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))

    if args.env_id == "TabletopPickPlaceEnv-v1":
        env_kwargs["object_name"] = args.object_name
        env_kwargs["container_name"] = args.container_name
    elif args.env_id == "TabletopPickEnv-v1":
        env_kwargs["object_name"] = args.object_name

    env: BaseEnv = gym.make(
        args.env_id,
        **env_kwargs,
    )
    sim_backend = 'gpu' if env.unwrapped.device.type == 'cuda' else 'cpu'
    device = str(env.unwrapped._sim_device)

    if args.model == "diffusion_policy":
        from mani_skill.evaluation.policies.diffusion_policy.dp_infer import DPInference
        from mani_skill.evaluation.policies.diffusion_policy.dp_modules.utils.math import get_pose_from_rot_pos_batch
        model = DPInference(args.obs_normalize_params_path, args.ckpt_path)
    elif args.model == "cql":
        from mani_skill.evaluation.policies.diffusion_policy.dp_modules.utils.math import get_pose_from_rot_pos_batch
        from mani_skill.evaluation.policies.cql.cql_new_infer import CQLInference
        model = CQLInference(saved_model_path=args.ckpt_path, device=device)
    else:
        raise NotImplementedError

    eval_metrics = defaultdict(list)
    initial_eps_count = args.initial_eps_count
    eps_count = 0

    print(f"Running Real2Sim Evaluation of model {args.model} on environment {args.env_id}")
    print(f"Using {args.num_envs} environments on the {sim_backend} simulation backend")

    timers = {"env.step+inference": 0, "env.step": 0, "inference": 0, "total": 0}
    total_start_time = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    model.pre_init(env) # get init pose to use target control

    while eps_count < args.num_episodes:
        seed = args.seed + eps_count

        # data dump
        datas = [{
            "image": [],  # obs_t: [0, T-1]
            "instruction": "",
            "action": [],  # a_t: [0, T-1]
            "info": [],  # info after executing a_t: [1, T]
        } for idx in range(args.num_envs)]

        # env and policy reset
        env_reset_options = {
            "reconfigure": True,
            "episode_id": torch.arange(args.num_envs) + eps_count + initial_eps_count,
        }
        obs, info = env.reset(seed=seed, options=env_reset_options)
        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8) # on cuda:0
        instruction = env.unwrapped.get_language_instruction()
        model.reset(instruction)

        print("instruction[0]:", instruction[0])

        # data dump: instruction
        for idx in range(args.num_envs):
            datas[idx]["instruction"] = instruction[idx]

        elapsed_steps = 0
        predicted_terminated, truncated = False, False
        while not (predicted_terminated or truncated):
        # inference
            start_time = time.time()
            # only for diffusion policy
            raw_actions, actions = model.step(env, obs_image, instruction) # actually only env is needed
            timers["inference"] += time.time() - start_time

            actions_list = []
            if args.model == "diffusion_policy":
                for i in range(10): # dp generate 8
                    B = actions.shape[0] # B indicates the environment number
                    action = actions[:,i,:] # [B, 10]
                    mat_6 = action[:,3:9].reshape(action.shape[0],3,2) # [B ,3, 2]
                    mat_6[:, :, 0] = mat_6[:, :, 0] / np.linalg.norm(mat_6[:, :, 0]) # [B, 3]
                    mat_6[:, :, 1] = mat_6[:, :, 1] / np.linalg.norm(mat_6[:, :, 1]) # [B, 3]
                    z_vec = np.cross(mat_6[:, :, 0], mat_6[:, :, 1]) # [B, 3]
                    z_vec = z_vec[:, :, np.newaxis]  # (B, 3, 1)
                    mat = np.concatenate([mat_6, z_vec], axis=2) # [B, 3, 3]
                    pos = action[:, :3] # [B, 3]
                    gripper_width = action[:, -1, np.newaxis] # [B, 1]
                    if args.is_delta:
                        init_to_desired_pose = model.pose_at_obs @ get_pose_from_rot_pos_batch(mat, pos) # for delta_action in base frame 
                    else:
                        init_to_desired_pose = get_pose_from_rot_pos_batch(mat, pos) # for abs_action
                    pose_action = np.concatenate(
                        [
                            init_to_desired_pose[:, :3, 3],
                            matrix_to_euler_angles(torch.from_numpy(init_to_desired_pose[:, :3, :3]),"XYZ").numpy(),
                            gripper_width
                        ],
                        axis=1) # [B, 7]
                    actions_list.append(pose_action)
            else:
                if args.is_delta:
                    pose_action = model.get_abs_actions_via_delta_actions(torch.from_numpy(actions).to(device)) # abs_actions
                else:
                    pose_action = actions # [B ,7]
                actions_list.append(pose_action)

            for i in range(len(actions_list)):
                # step
                start_time = time.time()
                action = actions_list[i]
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"step {elapsed_steps} ee_pose_action:", action)
                obs_image_new = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
                info = {k: v.cpu().numpy() for k, v in info.items()}
                truncated = bool(truncated.any())  # note that all envs truncate and terminate at the same time.

                timers["env.step"] += time.time() - start_time
                # print info
                info_dict = {k: v.mean().tolist() for k, v in info.items()}
                # print(f"step {elapsed_steps}: {info_dict}")

                # data dump: image, action, info
                for i in range(args.num_envs):
                    log_image = obs_image[i].cpu().numpy()
                    log_action = action[i].tolist()
                    log_info = {k: v[i].tolist() for k, v in info.items()}
                    datas[i]["image"].append(log_image)
                    datas[i]["action"].append(log_action)
                    datas[i]["info"].append(log_info)
                # add count
                obs_image = obs_image_new
                elapsed_steps += 1

        # data dump: last image
        for i in range(args.num_envs):
            log_image = obs_image[i].cpu().numpy()
            datas[i]["image"].append(log_image)

        # save video
        if args.save_video:
            if args.container_name != None and args.object_name != None:
                temp_name = f"put_{args.object_name}_on_{args.container_name}"
                exp_vis_dir = Path(args.record_dir) / f"visualize/{Path(args.ckpt_path).stem}/{args.env_id}" / temp_name / timestamp
                exp_vis_dir.mkdir(parents=True, exist_ok=True)
            elif args.object_name != None:
                temp_name = f"pick_the_{args.object_name}_up"
                exp_vis_dir = Path(args.record_dir) / f"visualize/{Path(args.ckpt_path).stem}/{args.env_id}" / temp_name / timestamp
                exp_vis_dir.mkdir(parents=True, exist_ok=True)
            else:
                exp_vis_dir = Path(args.record_dir) / f"visualize/{Path(args.ckpt_path).stem}/{args.env_id}" / timestamp
                exp_vis_dir.mkdir(parents=True, exist_ok=True)

            for i in range(args.num_envs):
                images = datas[i]["image"]
                infos = datas[i]["info"]
                assert len(images) == len(infos) + 1

                if args.info_on_video:
                    for j in range(len(infos)):
                        images[j + 1] = visualization.put_info_on_image(images[j + 1], infos[j])

                success = np.sum([d["success"] for d in infos]) >= 6
                images_to_video(images, str(exp_vis_dir), f"video_eps_{eps_count + initial_eps_count + i}_success={success}",
                                fps=30, verbose=True)

        # save data
        if args.save_data:
            exp_data_dir = Path(args.record_dir) / f"collect/{Path(args.ckpt_path).stem}/{args.env_id}" / timestamp
            exp_data_dir.mkdir(parents=True, exist_ok=True)

            for i in range(args.num_envs):
                if np.sum([d["success"] for d in datas[i]["info"]]) < 6:
                    continue
                res = datas[i].copy()
                res["image"] = [Image.fromarray(im).convert("RGB") for im in res["image"]]
                np.save(exp_data_dir / f"data_eps_{eps_count + initial_eps_count + i:0>4d}.npy", res)

        # metrics log and print
        for k, v in info.items():
            for i in range(args.num_envs):
                info_i = np.array([inf[k] for inf in datas[i]["info"]]).sum() >= 6
                eval_metrics[k].append(int(info_i))
            print(f"{k}: {np.mean(eval_metrics[k])}")

        eps_count += args.num_envs

    # Print timing information
    timers["total"] = time.time() - total_start_time
    timers["env.step+inference"] = timers["env.step"] + timers["inference"]

    print("\nTiming Info:")
    for key, value in timers.items():
        print(f"{key}: {value:.2f} seconds")

    mean_metrics = {k: np.mean(v) for k, v in eval_metrics.items()}
    mean_metrics["initial_episode_idx"] = initial_eps_count
    mean_metrics["total_episodes"] = eps_count
    mean_metrics["total_steps"] = eps_count * args.max_episode_len
    mean_metrics["time/episodes_per_second"] = eps_count / timers["total"]

    metrics_path = exp_vis_dir / f"eval_metrics.json"
    json.dump(mean_metrics, open(metrics_path, "w"), indent=4)
    print(f"Evaluation complete. Results saved to {exp_vis_dir}. Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
