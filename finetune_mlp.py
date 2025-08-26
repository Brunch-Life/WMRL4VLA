"""
finetune_mlp.py

eg:
    python finetune_mlp.py --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                          --dataset_name <DATASET_NAME> \
                          --run_root_dir <PATH/TO/LOGS/DIR> \
                          ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import gc
import time
import glob
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from scipy.spatial.transform import Rotation as R
from transformers import AutoProcessor
from PIL import Image
import draccus

# import MLP related from SimplerEnv
from simpler_env.policies.MLP.MLP_train import MLPPolicy

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class MLPFinetuneConfig:
    # fmt: off
    
    # Directory Paths
    data_root_dir: Path = Path("/home/chenyinuo/data/dataset/bingwen/data_for_success/green_bell_pepper_plate_wooden/success")  # Path to NPZ files directory
    dataset_name: str = "green_bell_pepper"                        # Name of fine-tuning dataset (for logging)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    
    # Fine-tuning Parameters
    batch_size: int = 256                                          # Fine-tuning batch size
    max_steps: int = 50_000                                         # Max number of fine-tuning steps
    eval_steps: int = 1000                                           # Interval for evaluation
    save_steps: str = "1,5000,10000,20000,50000"                     # Steps to save checkpoints
    learning_rate: float = 1e-4                                    # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size
    
    # MLP Model Parameters  
    mlp_embedding_size: int = 512                                   # MLP embedding size
    action_dim: int = 7                                             # Action dimension (3 pos + 3 euler + 1 gripper)
    state_dim: int = 0                                              # State dimension (not used)
    
    # Loss Weights
    action_loss_weight: float = 1.0                                 # Weight for action prediction loss
    value_loss_weight: float = 0.1                                  # Weight for value prediction loss (optional)
    
    # Tracking Parameters
    wandb_project: str = "mlp-sft"                                  # Name of W&B project to log to
    run_id_note: Optional[str] = None                               # Extra note for logging
    
    # Pre-trained model loading
    pretrained_mlp_path: Optional[str] = None                       # Path to pre-trained MLP model
    freeze_backbone: bool = False                                   # Whether to freeze ResNet backbone
    
    # fmt: on
    unnorm_key: Optional[str] = None


def _load_single_file(file_path: str, image_size: tuple = (480, 640)):
    """
    multi-process safe single file loading function
    Args:
        file_path: NPZ path
        image_size: target image size (H, W)
    Returns:
        tuple: (processed_images, actions) or None
    """
    try:
        data = np.load(file_path, allow_pickle=True)["arr_0"].tolist()
        
        # parse action data
        position = data["action"]["end"]["position"].squeeze(1)  # (T, 3)
        orientation_quat = data["action"]["end"]["orientation"].squeeze(1)  # (T, 4) quaternion
        gripper = data["action"]["effector"]["position_gripper"]  # (T, 1)
        
        # Convert quaternion to euler angles
        # orientation_quat is in (x, y, z, w) format, scipy expects (x, y, z, w)
        euler_angles = R.from_quat(orientation_quat).as_euler('xyz', degrees=False)  # (T, 3)
        
        # Concatenate: position (3) + euler (3) + gripper (1) = 7D action
        actions = np.concatenate([
            position,      # (T, 3)
            euler_angles,  # (T, 3) 
            gripper        # (T, 1)
        ], axis=1).astype(np.float32)  # (T, 7)
        
        # parse image data
        raw_images = data["observation"]["rgb"]
        
        # preprocess images (resize)
        processed_images = []
        for img in raw_images:
            img_array = np.asarray(img)
            
            # ensure image is uint8 format and has 3 channels
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                if img_array.shape[:2] != image_size:
                    # use PIL to resize
                    from PIL import Image
                    img_array = np.array(Image.fromarray(img_array).resize(
                        (image_size[1], image_size[0])  # PIL use (width, height)
                    ))
                processed_images.append(img_array)
            else:
                print(f"Warning: Skipping invalid image shape: {img_array.shape}")
                continue
        
        processed_images = np.stack(processed_images, axis=0)  # (N, H, W, C)
        
        # 确保数据长度一致
        min_len = min(len(actions), len(processed_images))
        actions = actions[:min_len]
        processed_images = processed_images[:min_len]
        
        return processed_images, actions
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


class NPZDataset(IterableDataset):
    """
    preload all NPZ data into memory dataset class, optimize training speed
    """
    def __init__(self, data_dir: Path, train: bool = True, image_size=(480, 640)):
        self.data_dir = Path(data_dir)
        self.train = train
        self.image_size = image_size
        
        print(f"Loading dataset from: {self.data_dir}")
        
        # find all NPZ files
        self.npz_files = sorted(glob.glob(str(self.data_dir / "*.npz")))
        print(f"Found {len(self.npz_files)} NPZ files in {self.data_dir}")
        
        if len(self.npz_files) == 0:
            raise ValueError(f"No NPZ files found in {self.data_dir}")
        
        # split train/eval set (80/20 split)
        split_idx = int(0.8 * len(self.npz_files))
        if train:
            self.npz_files = self.npz_files[:split_idx]
        else:
            self.npz_files = self.npz_files[split_idx:]
            
        print(f"Using {len(self.npz_files)} files for {'train' if train else 'eval'}")
        
        # multi-core parallel load all data into memory
        print("Loading all data into memory using multiprocessing...")
        start_time = time.time()
        
        # get CPU core number
        num_workers = min(mp.cpu_count(), len(self.npz_files), 16)  # limit max 16 processes
        print(f"Using {num_workers} processes for parallel loading")
        
        # multi-process parallel load
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # submit all tasks
            future_to_file = {
                executor.submit(_load_single_file, file_path, self.image_size): file_path 
                for file_path in self.npz_files
            }
            
            self.all_images = []
            self.all_actions = []
            total_samples = 0
            
            # collect results
            for i, future in enumerate(as_completed(future_to_file)):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result is not None:
                        processed_images, actions = result
                        self.all_images.append(processed_images)
                        self.all_actions.append(actions)
                        total_samples += len(actions)
                        print(f"  Loaded {len(actions)} samples from {file_path} ({i+1}/{len(self.npz_files)})")
                    
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
                    continue
        
        # merge all data
        if self.all_images:
            print("Concatenating all data...")
            concat_start = time.time()
            self.all_images = np.concatenate(self.all_images, axis=0)  # (Total_N, H, W, C)
            self.all_actions = np.concatenate(self.all_actions, axis=0)  # (Total_N, 8)
            concat_time = time.time() - concat_start
            
            total_time = time.time() - start_time
            print(f"Final dataset shapes:")
            print(f"  Images: {self.all_images.shape}")
            print(f"  Actions: {self.all_actions.shape}")
            print(f"  Total samples loaded: {len(self.all_images)}")
            print(f"  Loading time: {total_time:.1f}s (concat: {concat_time:.1f}s)")
        else:
            raise ValueError("No valid data loaded!")
        
        # create sample indices
        self.sample_indices = list(range(len(self.all_images)))
        print(f"Dataset ready with {len(self.sample_indices)} samples!")
    
    def __len__(self):
        """return dataset size"""
        return len(self.sample_indices)
    
    def __iter__(self):
        # infinite loop iterator
        while True:
            # shuffle indices for each epoch
            indices = self.sample_indices.copy()
            if self.train:
                random.shuffle(indices)
            
            for idx in indices:
                try:
                    sample = self._get_sample(idx)
                    if sample is not None:
                        yield sample
                except Exception as e:
                    print(f"Warning: Failed to get sample {idx}: {e}")
                    continue
    
    def _get_sample(self, idx: int):
        """get single sample from memory"""
        try:
            image = self.all_images[idx]  # (H, W, C) uint8
            action = self.all_actions[idx]  # (8,) float32
            
            return {
                "image": image,    # (H, W, C) uint8
                "action": action,  # (8,) float32 
                "language": "put green bell pepper on plate",  # task description
            }
            
        except Exception as e:
            print(f"Error getting sample {idx}: {e}")
            return None


def create_args_mock(mlp_embedding_size, alg_lr, action_dim=7):
    """create a mock args object for MLPPolicy"""
    class MockArgs:
        def __init__(self, mlp_embedding_size, alg_lr, action_dim):
            self.mlp_embedding_size = mlp_embedding_size
            self.alg_lr = alg_lr
            self.action_dim = action_dim
    
    return MockArgs(mlp_embedding_size, alg_lr, action_dim)


@draccus.wrap()
def finetune_mlp(cfg: MLPFinetuneConfig) -> None:
    print(f"Fine-tuning MLP Model on `{cfg.dataset_name}`")
    
    # parse save steps
    save_step_list = [int(x) for x in cfg.save_steps.split(",") if x.strip() != ""]
    
    # [Validate] Ensure GPU Available & Set Device
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    
    # single GPU device setting
    device_id = 0
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    print(f"Using single GPU training on device {device_id}")
    
    # Configure Unique Experiment ID & Log Directory
    exp_id = f"mlp_sft_steps_{cfg.max_steps}"
    if not cfg.image_aug:
        exp_id += "-no_aug"
    
    # Start =>> Build Directories
    run_dir = cfg.run_root_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)
    
    # create MLP policy
    mock_args = create_args_mock(cfg.mlp_embedding_size, cfg.learning_rate, cfg.action_dim)
    mlp_policy = MLPPolicy(mock_args, torch.device(device_id))
    
    # load pre-trained model (if provided)
    if cfg.pretrained_mlp_path is not None:
        print(f"Loading pre-trained MLP model from {cfg.pretrained_mlp_path}")
        mlp_policy.load(Path(cfg.pretrained_mlp_path))
    
    # freeze backbone (if needed)
    if cfg.freeze_backbone:
        print("Freezing ResNet backbone")
        for param in mlp_policy.resnet.parameters():
            param.requires_grad = False
    
    # create optimizer
    optimizer = AdamW(mlp_policy.parameters(), lr=cfg.learning_rate)
    
    # load dataset
    print(f"Loading dataset from: {cfg.data_root_dir}")
    
    # train dataset - load directly from NPZ files
    # note: original image may be 640x480, we change it to 480x640 to match ResNet expected input
    train_dataset = NPZDataset(
        cfg.data_root_dir,
        train=True,
        image_size=(480, 640)  # (height, width)
    )
    
    # eval dataset
    eval_dataset = NPZDataset(
        cfg.data_root_dir,
        train=False,
        image_size=(480, 640)  # (height, width)
    )
    
    # create custom collator
    def mlp_collate_fn(batch_list):
        """custom batch processing function"""
        batch_size = len(batch_list)
        
        # collect all data
        images = []
        actions = []
        languages = []
        
        for item in batch_list:
            images.append(item["image"])
            actions.append(item["action"])
            languages.append(item["language"])
        
        # convert to tensors
        images = np.stack(images)  # (B, H, W, C)
        actions = np.stack(actions)  # (B, action_dim)
        
        # ensure image is correct dimension
        if len(images.shape) == 4 and images.shape[-1] == 3:
            # convert image: normalize to [0,1], and convert to CHW format
            images = images.astype(np.float32) / 255.0  # normalize to [0,1]
            images = np.transpose(images, (0, 3, 1, 2))  # BHWC -> BCHW
        else:
            raise ValueError(f"Unexpected image shape: {images.shape}")
        
        return {
            "image": torch.from_numpy(images).float(),
            "action": torch.from_numpy(actions).float(),
            "language": languages
        }
    
    # create data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=mlp_collate_fn,
        num_workers=0,  # data is in memory, no need for multi-process
        pin_memory=True,  # accelerate GPU transfer
        drop_last=True,  # keep batch size consistent
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=mlp_collate_fn,
        num_workers=0,
        pin_memory=True,
        drop_last=False,  # keep all data for evaluation
    )
    
    # initialize W&B logging
    name = f"{cfg.dataset_name}-{exp_id}"
    wandb.init(project=cfg.wandb_project, name=name)
    
    # queue for storing recent losses (for smoothing during gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_losses = deque(maxlen=cfg.grad_accumulation_steps)
    
    # Training Loop
    print("Start training...")
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        mlp_policy.train()
        optimizer.zero_grad()
        
        step_count = 0
        
        for epoch in range(100):  # max epochs
            for batch_idx, batch in enumerate(train_dataloader):
                if step_count >= cfg.max_steps:
                    break
                    
                # move data to device
                batch_obs = {
                    "image": batch["image"].to(device_id)
                }
                target_actions = batch["action"].to(device_id)
                
                # forward pass
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # get policy output
                    value, pred_actions, log_prob = mlp_policy.get_action(batch_obs, deterministic=True)
                    
                    # calculate action loss (MSE)
                    action_loss = F.mse_loss(pred_actions, target_actions)
                    
                    # total loss
                    total_loss = cfg.action_loss_weight * action_loss
                
                # normalize loss to consider gradient accumulation
                normalized_loss = total_loss / cfg.grad_accumulation_steps
                
                # backward pass
                normalized_loss.backward()
                
                # store recent losses
                recent_losses.append(total_loss.item())
                recent_action_losses.append(action_loss.item())
                
                # calculate gradient step index
                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
                
                # calculate smoothed loss
                smoothened_loss = sum(recent_losses) / len(recent_losses)
                smoothened_action_loss = sum(recent_action_losses) / len(recent_action_losses)
                
                # push metrics to W&B (every 10 gradient steps)
                if gradient_step_idx % 10 == 0:
                    wandb.log(
                        {
                            "train_loss": smoothened_loss,
                            "action_loss": smoothened_action_loss,
                        },
                        step=step_count,
                    )
                
                # optimizer step
                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update()
                    step_count += 1
                
                # evaluation
                if step_count % cfg.eval_steps == 0 and step_count > 0:
                    print(f"Evaluating at step {step_count}")
                    mlp_policy.eval()
                    
                    eval_losses = []
                    eval_action_losses = []
                    
                    with torch.no_grad():
                        eval_steps = min(100, len(eval_dataloader))  # limit eval steps
                        for eval_idx, eval_batch in enumerate(eval_dataloader):
                            if eval_idx >= eval_steps:
                                break
                                
                            eval_obs = {
                                "image": eval_batch["image"].to(device_id)
                            }
                            eval_target_actions = eval_batch["action"].to(device_id)
                            
                            # forward pass
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                # get policy output (single GPU)
                                _, eval_pred_actions, _ = mlp_policy.get_action(eval_obs, deterministic=True)
                                eval_action_loss = F.mse_loss(eval_pred_actions, eval_target_actions)
                                eval_total_loss = cfg.action_loss_weight * eval_action_loss
                            
                            eval_losses.append(eval_total_loss.item())
                            eval_action_losses.append(eval_action_loss.item())
                    
                    avg_eval_loss = sum(eval_losses) / len(eval_losses)
                    avg_eval_action_loss = sum(eval_action_losses) / len(eval_action_losses)
                    
                    wandb.log(
                        {
                            "eval_loss": avg_eval_loss,
                            "eval_action_loss": avg_eval_action_loss,
                        },
                        step=step_count,
                    )
                    
                    print(f"Eval at step {step_count}: loss={avg_eval_loss:.4f}, action_loss={avg_eval_action_loss:.4f}")
                    mlp_policy.train()
                
                # save model checkpoint
                if step_count in save_step_list:
                    print(f"Saving model checkpoint for step {step_count}")
                    save_path = run_dir / f"step_{step_count:06d}"
                    
                    # save model (single GPU)
                    mlp_policy.save(save_path)
                    
                    # save optimizer state
                    torch.save({
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': step_count,
                    }, save_path / "optimizer.pt")
                
                # check if reached max steps
                if step_count >= cfg.max_steps:
                    print(f"Reached max steps {cfg.max_steps}! Stopping training...")
                    break
                    
                # memory cleanup
                if step_count % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
            if step_count >= cfg.max_steps:
                break
    
    # final save
    print("Saving final model...")
    final_save_path = run_dir / "final"
    mlp_policy.save(final_save_path)
    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step_count,
    }, final_save_path / "optimizer.pt")
    
    print("Training completed!")


if __name__ == "__main__":
    finetune_mlp()
