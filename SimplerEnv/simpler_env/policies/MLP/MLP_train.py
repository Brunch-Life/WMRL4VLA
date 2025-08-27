"""
This file implements the MLP policy for robot control, as described by the user.
The policy processes image and state observations to produce actions.

The overall architecture is as follows:
1.  **Observation Preprocessing (`_get_embedding`)**:
    -   Image observations are passed through a pretrained ResNet-18 to extract features.
    -   State observations (7D: 3D position, 3D Euler angles, 1D gripper width) are processed:
        -   Euler angles are converted to a 6D rotation representation.
        -   The position, 6D rotation, and gripper state are concatenated into a 10D vector.
        -   This vector is passed through an MLP to get state features.
    -   Image and state features are concatenated and passed through a final MLP to produce a unified embedding.

2.  **Action Generation (`get_action`)**:
    -   The policy network (actor) outputs a 10D action (3D delta position, 6D rotation representation, 1D absolute gripper width).
    -   This 10D action is post-processed to compute a 7D absolute target pose:
        -   The 6D rotation is converted to a full rotation matrix.
        -   A homogeneous transformation matrix for the delta pose is created.
        -   This delta transform is applied to the current robot pose (from observation) to get the absolute target pose.
        -   The target pose is decomposed back into 3D position, 3D Euler angles, and gripper width.

3.  **Training (`MLPPPO`)**:
    -   The policy is trained using PPO. The `evaluate_actions` method is crucial for this, as it computes the log-probabilities and entropy of actions taken during rollouts.
    -   A critic network estimates the value function from the unified embedding.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from scipy.spatial.transform import Rotation as R
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def process_image_for_model_training(obs_image):
    """Convert image from (B, H, W, C) to (B, C, H, W) and normalize for training"""
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


def huber_loss(e, d):
    a = (abs(e) <= d).to(torch.float32)
    b = (abs(e) > d).to(torch.float32)
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


class MLPPolicy(nn.Module):
    def __init__(self, all_args, device: torch.device):
        super().__init__()
        self.args = all_args
        self.device = device
        self.tpdv = dict(device=self.device, dtype=torch.float32)

        # --- Model Components ---
        # 1. ResNet for image feature extraction
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.resnet.to(**self.tpdv)

        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        resnet_feature_dim = 512  # ResNet18 final feature dimension

        # 2. Image processing MLP - 5 layers with 1024 hidden units each
        final_embedding_dim = self.args.mlp_embedding_size
        self.image_mlp = nn.Sequential(
            nn.Linear(resnet_feature_dim, 1024),  # 512 -> 1024
            nn.ReLU(),
            nn.Linear(1024, 1024),  # 1024 -> 1024
            nn.ReLU(),
            nn.Linear(1024, 1024),  # 1024 -> 1024
            nn.ReLU(),
            nn.Linear(1024, 1024),  # 1024 -> 1024
            nn.ReLU(),
            nn.Linear(1024, final_embedding_dim)  # 1024 -> final_embedding_dim
        ).to(**self.tpdv)

        # 4. Actor and Critic Heads - enlarged to match the increased network capacity
        # Use action_dim from args if available, otherwise default to 7
        self.action_dim = getattr(self.args, 'action_dim', 7)  # 3 (pos) + 3 (euler) + 1 (gripper)
        self.actor = nn.Sequential(
            nn.Linear(final_embedding_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.action_dim)
        ).to(**self.tpdv)
        
        self.critic = nn.Sequential(
            nn.Linear(final_embedding_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        ).to(**self.tpdv)

        # Learnable parameter for the standard deviation of the action distribution
        self.action_log_std = nn.Parameter(torch.zeros(1, self.action_dim, device=self.device))
        
        # Note: Optimizer will be created externally
    
    def get_trainable_params(self):
        """Get all trainable parameters for optimizer creation"""
        return list(self.parameters())

    # --- Pose Transformation Utilities ---
    def _euler_to_rotation_matrix(self, euler: torch.Tensor) -> torch.Tensor:
        """Converts a batch of Euler angles to rotation matrices."""
        euler_np = euler.cpu().numpy()
        r = R.from_euler('xyz', euler_np, degrees=False)
        return torch.from_numpy(r.as_matrix()).to(self.device, dtype=torch.float32)

    def _rotation_matrix_to_euler(self, mat: torch.Tensor) -> torch.Tensor:
        """Converts a batch of rotation matrices to Euler angles."""
        mat_np = mat.cpu().numpy()
        r = R.from_matrix(mat_np)
        return torch.from_numpy(r.as_euler('xyz', degrees=False)).to(self.device, dtype=torch.float32)

    def _6d_to_rotation_matrix(self, rot_6d: torch.Tensor) -> torch.Tensor:
        """Converts 6D rotation representation to a 3x3 rotation matrix."""
        a1 = rot_6d[:, 0:3]
        a2 = rot_6d[:, 3:6]
        b1 = nn.functional.normalize(a1, dim=1)
        b2 = nn.functional.normalize(a2 - torch.sum(b1 * a2, dim=1, keepdim=True) * b1, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=2)

    # --- Core Policy Methods ---
    def _get_embedding(self, x: dict) -> torch.Tensor:
        """Processes raw observation dict to image feature embedding."""
        # Process image: expect (B, C, H, W) format already preprocessed and normalized
        images = x['image'].to(self.device, dtype=torch.float32)
        
        with torch.no_grad():
            img_features = self.resnet(images)
            img_features = self.avgpool(img_features)
            img_features = torch.flatten(img_features, 1)

        # Process image features through MLP
        return self.image_mlp(img_features)

    def _get_action_dist(self, embedding: torch.Tensor) -> torch.distributions.Normal:
        """Gets the action distribution from the embedding."""
        action_mean = self.actor(embedding)
        action_std = torch.exp(self.action_log_std)
        return torch.distributions.Normal(action_mean, action_std)

    # def _postprocess_action(self, raw_action: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
    #     """Converts a 10D raw action to a 7D absolute action."""
    #     delta_pos = raw_action[:, :3]
    #     rot_6d = raw_action[:, 3:9]
    #     abs_gripper = raw_action[:, 9].unsqueeze(-1)
        
    #     R_delta = self._6d_to_rotation_matrix(rot_6d)
    #     T_delta = torch.eye(4, **self.tpdv).unsqueeze(0).repeat(R_delta.shape[0], 1, 1)
    #     T_delta[:, :3, :3] = R_delta
    #     T_delta[:, :3, 3] = delta_pos
        
    #     current_pos, current_euler = current_state[:, :3], current_state[:, 3:6]
    #     R_current = self._euler_to_rotation_matrix(current_euler)
    #     T_current = torch.eye(4, **self.tpdv).unsqueeze(0).repeat(R_current.shape[0], 1, 1)
    #     T_current[:, :3, :3] = R_current
    #     T_current[:, :3, 3] = current_pos
        
    #     T_abs_target = torch.bmm(T_current, T_delta)
        
    #     abs_pos = T_abs_target[:, :3, 3]
    #     abs_euler = self._rotation_matrix_to_euler(T_abs_target[:, :3, :3])
        
    #     return torch.cat([abs_pos, abs_euler, abs_gripper], dim=1)

    def get_action(self, x: dict, deterministic=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Only use image, no state needed
        obs_dict = {"image": x["image"]} if "image" in x else x
        embedding = self._get_embedding(obs_dict)
        dist = self._get_action_dist(embedding)

        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.sample()
        
        log_prob = dist.log_prob(raw_action).sum(axis=-1, keepdim=True)
        value = self.critic(embedding)
        
        final_action = raw_action  # 7D: 3 pos + 3 euler + 1 gripper
        
        return value, final_action, log_prob

    def get_value(self, x: dict) -> torch.Tensor:
        # Only use image, no state needed
        obs_dict = {"image": x["image"]} if "image" in x else x
        embedding = self._get_embedding(obs_dict)
        return self.critic(embedding)

    def evaluate_actions(self, x: dict, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates actions for PPO training."""
        # Only use image, no state needed
        obs_dict = {"image": x["image"]} if "image" in x else x
        embedding = self._get_embedding(obs_dict)
        dist = self._get_action_dist(embedding)
        
        # For image-only model, we directly use the action without state-based transformation
        log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        entropy = dist.entropy().sum(axis=-1, keepdim=True)
        value = self.critic(embedding)
        
        return log_prob, entropy, value

    def prep_rollout(self):
        """Set models to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
        self.image_mlp.eval()

    def prep_training(self):
        """Set models to training mode."""
        self.actor.train()
        self.critic.train()
        self.image_mlp.train()

    def save(self, path: Path):
        """Saves the model states."""
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
        }, path / "mlp_policy.pt")

    def load(self, path: Path):
        """Loads the model states."""
        checkpoint = torch.load(path / "mlp_policy.pt", map_location=self.device)
        if 'model_state_dict' in checkpoint:
            # New format
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Legacy format - try to load individual components
            if 'image_mlp_state_dict' in checkpoint:
                self.image_mlp.load_state_dict(checkpoint['image_mlp_state_dict'])
            if 'actor_state_dict' in checkpoint:
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
            if 'critic_state_dict' in checkpoint:
                self.critic.load_state_dict(checkpoint['critic_state_dict'])
            if 'action_log_std' in checkpoint:
                self.action_log_std.data.copy_(checkpoint['action_log_std'])


class MLPPPO:
    def __init__(self, all_args, policy: MLPPolicy):
        self.args = all_args
        self.policy = policy
        self.ppo_clip = 0.2
        self.ppo_grad_norm = 10.0
        self.ppo_entropy_coef = self.args.alg_entropy_coef
        self.ppo_huber_delta = 10.0
        self.tpdv = self.policy.tpdv
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.args.alg_lr)

    def train_ppo_step(self, idx, total, batch):
        obs_image, obs_state, actions, value_preds, returns, masks, old_logprob, advantages = batch

        # Only use image, ignore state
        # Process image format from BHWC to BCHW before passing to model
        processed_image = process_image_for_model_training(obs_image)
        obs = dict(
            image=processed_image.to(**self.tpdv)
        )
        actions = torch.tensor(actions).to(**self.tpdv)
        value_preds = torch.tensor(value_preds).to(**self.tpdv)
        returns = torch.tensor(returns).to(**self.tpdv)
        old_logprob = torch.tensor(old_logprob).to(**self.tpdv)
        advantages = torch.tensor(advantages).to(**self.tpdv)

        # Policy loss
        logprob, entropy, values = self.policy.evaluate_actions(obs, actions)
        ratio = torch.exp(logprob - old_logprob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_pred_clipped = value_preds + (values - value_preds).clamp(-self.ppo_clip, self.ppo_clip)
        error_clipped = returns - value_pred_clipped
        error_original = returns - values
        value_loss_clipped = huber_loss(error_clipped, self.ppo_huber_delta)
        value_loss_original = huber_loss(error_original, self.ppo_huber_delta)
        value_loss = torch.max(value_loss_original, value_loss_clipped).mean()
        
        # Entropy loss
        entropy_loss = entropy.mean()

        # Total loss
        loss = policy_loss + value_loss - self.ppo_entropy_coef * entropy_loss
        
        # Gradient accumulation
        loss /= self.args.alg_gradient_accum
        loss.backward()

        if idx % self.args.alg_gradient_accum == (self.args.alg_gradient_accum - 1) or idx == (total - 1):
            grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.ppo_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            grad_norm = None

        info = dict(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy_loss=entropy_loss.item(),
            ratio=ratio.mean().item(),
        )
        if grad_norm is not None:
            info["grad_norm"] = grad_norm.item()

        return info

    def train_ppo(self, buffer):
        train_info = defaultdict(lambda: [])
        buffer.compute_returns_ppo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()
            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="train"):
                info = self.train_ppo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        final_info = {key: np.mean(value) for key, value in train_info.items()}
        return final_info
