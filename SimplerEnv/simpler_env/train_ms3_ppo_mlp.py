import os
import pprint
import random
import gc
import signal
from collections import defaultdict
import time
from pathlib import Path
from typing import Annotated, Optional
import torch
import numpy as np
import tyro
import wandb
from dataclasses import dataclass
import yaml
from tqdm import tqdm
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video

from simpler_env.env.simpler_wrapper_mlp import MLPMS3Wrapper
from simpler_env.utils.replay_buffer_mlp import SeparatedReplayBufferMLP

signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "TabletopPickPlaceEnv-v1"
    """The environment ID of the task you want to simulate. Can be one of
    PutCarrotOnPlateInScene-v1, PutSpoonOnTableClothInScene-v1, StackGreenCubeOnYellowCubeBakedTexInScene-v1, PutEggplantInBasketScene-v1"""

    """Number of environments to run. With more than 1 environment the environment will use the GPU backend 
    which runs faster enabling faster large-scale evaluations. Note that the overall behavior of the simulation
    will be slightly different between CPU and GPU backends."""

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Seed the model and environment. Default seed is 0"""

    name: str = "PPO-test"

    # env
    num_envs: int = 64
    episode_len: int = 200
    shader: str = "default"
    use_same_init: bool = False
    control_mode: str = "pd_ee_pose"
    is_table_green: bool = False
    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None

    object_name: Optional[str] = "green_bell_pepper"
    container_name: Optional[str] = "plate"

    steps_max: int = 2000000
    interval_eval: int = 2
    interval_save: int = 1


    # buffer
    buffer_inferbatch: int = num_envs
    buffer_minibatch: int = 128
    buffer_gamma: float = 0.99
    buffer_lambda: float = 0.95


    # ppo & grpo
    alg_name: str = "ppo"  # ppo, grpo
    alg_grpo_fix: bool = True
    alg_gradient_accum: int = 20
    alg_ppo_epoch: int = 1
    alg_entropy_coef: float = 0.0
    alg_lr: float = 3e-5


    # other
    wandb: bool = True
    only_render: bool = False
    render_info: bool = False
    device: str = "cuda"
    device_other: str = "cuda"

    # mlp
    mlp_embedding_size: int = 512
    pretrained_mlp_path: Optional[str] = None  # Path to pretrained MLP model from SFT
    



def process_image_for_model(obs_image):
    """Convert image from (B, H, W, C) to (B, C, H, W) and normalize"""
    # 确保输入是tensor
    if not isinstance(obs_image, torch.Tensor):
        obs_image = torch.tensor(obs_image)
    
    # 检查是否为BHWC格式 (batch, height, width, channels)
    if len(obs_image.shape) == 4 and obs_image.shape[-1] == 3:
        obs_image = obs_image.float() / 255.0  # normalize to [0, 1]
        obs_image = obs_image.permute(0, 3, 1, 2)  # BHWC -> BCHW
    
    # 检查是否已经是BCHW格式 (batch, channels, height, width)
    elif len(obs_image.shape) == 4 and obs_image.shape[1] == 3:
        obs_image = obs_image.float() / 255.0 if obs_image.dtype != torch.float32 else obs_image
    
    # 处理其他可能的格式
    else:
        raise ValueError(f"Unexpected image shape: {obs_image.shape}. Expected (B, H, W, 3) or (B, 3, H, W)")
    
    return obs_image

class Runner:
    def __init__(self, all_args: Args):
        self.args = all_args

        # alg_name
        assert self.args.alg_name in ["ppo", "grpo"]

        # set seed
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        # set wandb
        wandb.init(
            config=all_args.__dict__,
            project="WMPPO-MLP",
            name=self.args.name,
            mode="online" if self.args.wandb else "offline",
        )
        self.save_dir = Path(wandb.run.dir)
        self.glob_dir = Path(wandb.run.dir) / ".." / "glob"
        self.glob_dir.mkdir(parents=True, exist_ok=True)

        yaml.dump(all_args.__dict__, open(self.glob_dir / "config.yaml", "w"))

        # policy
        from simpler_env.policies.MLP.MLP_train import MLPPolicy, MLPPPO
        self.device = torch.device(self.args.device)
        if torch.cuda.device_count() > 1:
            self.device_env = torch.device(self.args.device_other)
        else:
            self.device_env = self.device

        self.policy = MLPPolicy(all_args, self.device)
        
        # Load pretrained model if provided
        if self.args.pretrained_mlp_path is not None:
            print(f"Loading pretrained MLP model from: {self.args.pretrained_mlp_path}")
            self.policy.load(Path(self.args.pretrained_mlp_path))
            print("Successfully loaded pretrained MLP model!")

        self.alg = MLPPPO(all_args, self.policy)

        # env
        self.env_wrapper = MLPMS3Wrapper(self.args, self.device_env)

        # buffer
        self.buffer = SeparatedReplayBufferMLP(
            all_args,
            obs_dim=(480, 640, 3),
            state_dim=0,  # No state input
            act_dim=7,  # 3 pos + 3 euler + 1 gripper
        )
        minibatch_count = self.buffer.get_minibatch_count()
        print(f"Buffer minibatch count: {minibatch_count}")

    @torch.no_grad()
    def _get_action(self, obs: dict, deterministic=False, render_mode=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert isinstance(obs, dict)
        assert isinstance(obs["image"], torch.Tensor)
        # No longer need state assertion
        
        total_batch = obs["image"].shape[0]

        values = []
        actions = []
        logprobs = []

        # Use smaller batch size for render to save memory
        batch_size = 16 if render_mode else self.args.buffer_inferbatch
        
        for i in range(0, total_batch, batch_size):
            # Only pass image data
            obs_batch = {"image": obs["image"][i:i + batch_size]}
            value, action, logprob = self.policy.get_action(obs_batch, deterministic)
            values.append(value)
            actions.append(action)
            logprobs.append(logprob)

        values = torch.cat(values, dim=0).to(device=self.device)
        actions = torch.cat(actions, dim=0).to(device=self.device)
        logprobs = torch.cat(logprobs, dim=0).to(device=self.device)

        return values, actions, logprobs

    def collect(self):
        self.policy.prep_rollout()

        obs_image = self.buffer.obs[self.buffer.step]
        obs_image = torch.tensor(obs_image).to(self.device)
        obs_image = process_image_for_model(obs_image)

        # No state needed
        obs = dict(image=obs_image)
        value, action, logprob = self._get_action(obs)

        return value, action, logprob

    def insert(self, data):
        obs_img, actions, logprob, value_preds, rewards, terminated, truncated = data
        masks = 1.0 - truncated.to(torch.float32)

        obs_img = obs_img
        actions = actions.to(torch.int32).cpu().numpy()
        logprob = logprob.to(torch.float32).cpu().numpy()
        value_preds = value_preds.to(torch.float32).cpu().numpy()
        rewards = rewards.cpu().numpy()
        masks = masks.cpu().numpy()

        # Pass empty state (None) since we don't use state
        self.buffer.insert(obs_img, None, actions, logprob, value_preds, rewards, masks)

    def compute_endup(self):
        self.policy.prep_rollout()

        obs_image = torch.tensor(self.buffer.obs[-1]).to(self.device)
        obs_image = process_image_for_model(obs_image)
            
        # No state needed
        obs = dict(image=obs_image)
        with torch.no_grad():
            next_value, _, _ = self._get_action(obs)
        next_value = next_value.to(torch.float32).cpu().numpy()

        self.buffer.endup(next_value)

    def train(self):
        self.policy.prep_training()

        if self.args.alg_name == "ppo":
            train_info = self.alg.train_ppo(self.buffer)
        else:
            raise ValueError(f"Unknown alg_name: {self.args.alg_name}")

        info = {f"train/{k}": v for k, v in train_info.items()}
        info["buffer/reward_mean"] = np.mean(self.buffer.rewards)
        info["buffer/mask_mean"] = np.mean(1.0 - self.buffer.masks)

        return info

    @torch.no_grad()
    def eval(self) -> dict:
        self.policy.prep_rollout()
        env_infos = defaultdict(lambda: [])


        obs_img, _, info = self.env_wrapper.reset(eps_count=0)  # ignore state
        obs_img = torch.tensor(obs_img).to(self.device)

        for _ in range(self.args.episode_len):
            obs_img_processed = process_image_for_model(obs_img)
            obs = dict(image=obs_img_processed)  # only image
            value, action, logprob = self._get_action(obs, deterministic=True) # need tensor
            print(f"!!action: {action[0]}")

            obs_img, _, reward, terminated, truncated, env_info = self.env_wrapper.step(action)  # ignore state
            obs_img = torch.tensor(obs_img).to(self.device)

            # info
            print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items() if k != "episode"})
            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v

        # infos
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats = env_stats.copy()

        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")

        return env_stats

    @torch.no_grad()
    def render(self, epoch: int) -> dict:
        self.policy.prep_rollout()

        # init logger
        env_infos = defaultdict(lambda: [])
        datas = [{
            "image": [],  # obs_t: [0, T-1]
            "state": [],  # state_t: [0, T-1]
            "action": [],  # a_t: [0, T-1]
            "info": [],  # info after executing a_t: [1, T]
        } for idx in range(self.args.num_envs)]

        obs_img, _, info = self.env_wrapper.reset(eps_count=0) # obs_img: np.ndarray, ignore state, info: dict
        obs_img = torch.tensor(obs_img).to(self.device)

        # No state data to dump since we don't use state

        for step in range(self.args.episode_len):
            obs_img_processed = process_image_for_model(obs_img)
            obs = dict(image=obs_img_processed) # obs: dict, image: tensor
            value, action, logprob = self._get_action(obs, deterministic=True, render_mode=True) # action: tensor

            #action: tensor, obs_img: np.ndarray, reward: tensor, terminated: tensor, truncated: tensor, env_info: dict
            obs_img_new, _, reward, terminated, truncated, env_info = self.env_wrapper.step(action)  # ignore state
            obs_img_new = torch.tensor(obs_img_new).to(self.device) 

            # info
            print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items() if k != "episode"})
            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v

            for i in range(self.args.num_envs):
                post_action = self.env_wrapper._process_action(action) # need tensor
                log_image = obs_img[i].cpu().numpy()  # Convert to numpy for video generation
                log_action = post_action[i].tolist()
                log_info = {k: v[i].tolist() for k, v in env_info.items() if k != "episode"}
                datas[i]["image"].append(log_image)
                datas[i]["action"].append(log_action)
                datas[i]["info"].append(log_info)

            # update obs_img
            obs_img = obs_img_new 

        # data dump: last image
        for i in range(self.args.num_envs):
            log_image = obs_img[i].cpu().numpy()  # Convert to numpy for video generation
            datas[i]["image"].append(log_image)

        # save video
        exp_dir = Path(self.glob_dir) / f"vis_{epoch}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.args.num_envs):
            images = datas[i]["image"]
            infos = datas[i]["info"]
            assert len(images) == len(infos) + 1

            if self.args.render_info:
                for j in range(len(infos)):
                    images[j + 1] = visualization.put_info_on_image(
                        images[j + 1], infos[j],
                        extras=[]  # No state info since we don't use state
                    )

            success = int(infos[-1]["success"])
            images_to_video(images, str(exp_dir), f"video_{i}-s_{success}",
                            fps=10, verbose=False)

        # infos
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats_ret = env_stats.copy()

        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")

        # save stats
        last_info = {
            idx: {k: env_infos[k][idx] for k in env_infos.keys()}
            for idx in range(self.args.num_envs)
        }

        save_stats = {}
        save_stats["env_name"] = self.args.env_id
        save_stats["ep_len"] = self.args.episode_len
        save_stats["epoch"] = epoch
        save_stats["stats"] = {k: v.item() if hasattr(v, 'item') else v for k, v in env_stats.items()}
        # No state to save since we don't use state
        save_stats["last_info"] = last_info

        yaml.dump(save_stats, open(exp_dir / "stats.yaml", "w"))

        # Clear render data from memory
        del datas, env_infos, last_info, save_stats
        torch.cuda.empty_cache()

        return env_stats_ret

    def run(self):
        max_episodes = self.args.steps_max // self.args.episode_len // self.args.num_envs
        print(f"Starting training with {max_episodes} total episodes")

        for episode in tqdm(range(max_episodes), desc="Training Episodes", position=0):
            env_infos = defaultdict(lambda: [])
            ep_time = time.time() 

            obs_img, _, info = self.env_wrapper.reset(eps_count=episode)  # ignore state

            # TODO: check buffer warmup
            self.buffer.warmup(obs_img, None)  # no state

            for step in tqdm(range(self.args.episode_len), 
                           desc=f"Episode {episode+1}/{max_episodes} - Rollout", 
                           position=1, 
                           leave=False):
                value, action, logprob = self.collect()
                obs_img, _, reward, terminated, truncated, env_info = self.env_wrapper.step(action)  # ignore state

                data = (obs_img, action, logprob, value, reward, terminated, truncated)  # removed state
                self.insert(data)

                # info
                if "episode" in env_info.keys():
                    for k, v in env_info["episode"].items():
                        env_infos[f"{k}"] += v

            # steps
            steps = (episode + 1) * self.args.episode_len * self.args.num_envs
            
            # train and process infos
            self.compute_endup()
            del value, action, logprob, obs_img, reward, terminated, truncated
            gc.collect()
            torch.cuda.empty_cache()

            # train
            infos = self.train()
            for k, v in env_infos.items():
                infos[f"env/{k}"] = np.mean(v)

            # log
            wandb.log(infos, step=steps)

            elapsed_time = time.time() - ep_time
            
            # Format training information output
            env_metrics = {k: round(np.mean(v), 4) for k, v in env_infos.items()}
            success_rate = env_metrics.get('success', 0.0)
            
            print(f"\nEpisode {episode+1:04d}/{max_episodes} completed!")
            print(f"   Success rate: {success_rate:.1%}")
            print(f"   Total steps: {steps:,}")
            print(f"   Duration: {elapsed_time:.2f}s")
            if env_metrics:
                print(f"   Env metrics: {env_metrics}")
            
            # Show key training metrics
            key_train_metrics = {k: round(v, 4) for k, v in infos.items() 
                               if any(x in k.lower() for x in ['loss', 'lr', 'reward', 'success'])}
            if key_train_metrics:
                print(f"   Train metrics: {key_train_metrics}")

            # eval
            if episode % self.args.interval_eval == self.args.interval_eval - 1 or episode == max_episodes - 1:
                print(f"\nStarting evaluation (Episode {episode+1}, Steps {steps:,})")
                sval_stats = self.eval()
                eval_success_rate = sval_stats.get('success', 0.0)
                print(f"   Eval success rate: {eval_success_rate:.1%}")
                
                sval_stats = {f"eval/{k}": v for k, v in sval_stats.items()}
                wandb.log(sval_stats, step=steps)

            # save
            if episode % self.args.interval_save == self.args.interval_save - 1 or episode == max_episodes - 1:
                print(f"\nSaving model and videos (Episode {episode+1}, Steps {steps:,})")
                save_path = self.glob_dir / f"steps_{episode:0>4d}"
                self.policy.save(save_path)
                print(f"   Model saved to: {save_path}")

                print("   Generating training video...")
                self.render(epoch=episode)
                print(f"   Videos saved to: {self.glob_dir}/vis_{episode}/")
                
                # Clear GPU memory after rendering
                print("   Clearing GPU memory...")
                torch.cuda.empty_cache()
                gc.collect()
                print("   GPU memory cleared")


def main():
    args = tyro.cli(Args)
    runner = Runner(args)

    if args.only_render:
        runner.render(epoch=0)
    else:
        runner.run()


if __name__ == "__main__":
    main()
