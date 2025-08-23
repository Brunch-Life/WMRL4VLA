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


def huber_loss(e, d):
    a = (abs(e) <= d).to(torch.float32)
    b = (abs(e) > d).to(torch.float32)
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


class MLPPolicy:
    def __init__(self, all_args, device: torch.device):
        self.args = all_args
        self.device = device
        self.tpdv = dict(device=self.device, dtype=torch.float32)

        # --- Model Components ---
        # 1. ResNet for image feature extraction
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.resnet.to(**self.tpdv)

        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        resnet_feature_dim = 64

        # 2. MLP for state processing
        state_input_dim = 10  # 3 (pos) + 6 (rot) + 1 (gripper)
        self.state_mlp = nn.Sequential(
            nn.Linear(state_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, resnet_feature_dim)
        ).to(**self.tpdv)

        # 3. Fusion MLP to combine image and state features
        fusion_input_dim = 512 + resnet_feature_dim
        final_embedding_dim = self.args.mlp_embedding_size
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, final_embedding_dim)
        ).to(**self.tpdv)

        # 4. Actor and Critic Heads
        self.action_dim = 10  # 3 (delta_pos) + 6 (rot_6d) + 1 (gripper)
        self.actor = nn.Sequential(
            nn.Linear(final_embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_dim)
        ).to(**self.tpdv)
        
        self.critic = nn.Sequential(
            nn.Linear(final_embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        ).to(**self.tpdv)

        # Learnable parameter for the standard deviation of the action distribution
        self.action_log_std = nn.Parameter(torch.zeros(1, self.action_dim, device=self.device))
        
        # --- Optimizer ---
        self.trainable_params = list(self.state_mlp.parameters()) + \
                               list(self.fusion_mlp.parameters()) + \
                               list(self.actor.parameters()) + \
                               list(self.critic.parameters()) + \
                               [self.action_log_std]
        self.optimizer = torch.optim.Adam(self.trainable_params, lr=self.args.alg_lr)

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
        """Processes raw observation dict to a fused feature embedding."""
        # Process image: (B, H, W, C) -> (B, 512)
        image_obs = x['image'].to(self.device, dtype=torch.uint8)
        image_obs_float = image_obs.to(torch.float32) / 255.0
        images = image_obs_float.permute(0, 3, 1, 2)
        with torch.no_grad():
            img_features = self.resnet(images)
            img_features = self.avgpool(img_features)
            img_features = torch.flatten(img_features, 1)

        # Process state: (B, 7) -> (B, 10) -> (B, 512)
        state_obs = x['state'].to(**self.tpdv)
        pos, euler, gripper = state_obs[:, :3], state_obs[:, 3:6], state_obs[:, 6].unsqueeze(-1)
        rot_mat = self._euler_to_rotation_matrix(euler)
        rot_6d = rot_mat[:, :2, :].reshape(-1, 6)
        
        processed_state = torch.cat([pos, rot_6d, gripper], dim=1)
        state_features = self.state_mlp(processed_state)

        # Fuse features
        fused_features = torch.cat([img_features, state_features], dim=1)
        return self.fusion_mlp(fused_features)

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
        embedding = self._get_embedding(x)
        dist = self._get_action_dist(embedding)

        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.sample()
        
        log_prob = dist.log_prob(raw_action).sum(axis=-1, keepdim=True)
        value = self.critic(embedding)
        
        final_action = raw_action #10D
        
        return value, final_action, log_prob

    def get_value(self, x: dict) -> torch.Tensor:
        embedding = self._get_embedding(x)
        return self.critic(embedding)

    def evaluate_actions(self, x: dict, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates actions for PPO training."""
        embedding = self._get_embedding(x)
        dist = self._get_action_dist(embedding)
        
        # Reverse the action post-processing to get the raw action
        abs_pos, abs_6d, abs_gripper = action[:, :3], action[:, 3:9], action[:, 9].unsqueeze(-1)
        R_abs_target = self._6d_to_rotation_matrix(abs_6d)
        T_abs_target = torch.eye(4, **self.tpdv).unsqueeze(0).repeat(action.shape[0], 1, 1)
        T_abs_target[:, :3, :3] = R_abs_target
        T_abs_target[:, :3, 3] = abs_pos
        
        current_state = x['state'].to(**self.tpdv)
        current_pos, current_6d = current_state[:, :3], current_state[:, 3:9]
        R_current = self._6d_to_rotation_matrix(current_6d)
        T_current = torch.eye(4, **self.tpdv).unsqueeze(0).repeat(action.shape[0], 1, 1)
        T_current[:, :3, :3] = R_current
        T_current[:, :3, 3] = current_pos
        
        R_current_inv = R_current.transpose(1, 2)
        T_current_inv = torch.eye(4, **self.tpdv).unsqueeze(0).repeat(action.shape[0], 1, 1)
        T_current_inv[:, :3, :3] = R_current_inv
        T_current_inv[:, :3, 3] = -torch.bmm(R_current_inv, current_pos.unsqueeze(-1)).squeeze(-1)
        T_delta = torch.bmm(T_current_inv, T_abs_target)

        delta_pos = T_delta[:, :3, 3]
        R_delta = T_delta[:, :3, :3]
        rot_6d = R_delta[:, :2, :].reshape(-1, 6)
        
        raw_action_reconstructed = torch.cat([delta_pos, rot_6d, abs_gripper], dim=1)

        log_prob = dist.log_prob(raw_action_reconstructed).sum(axis=-1, keepdim=True)
        entropy = dist.entropy().sum(axis=-1, keepdim=True)
        value = self.critic(embedding)
        
        return log_prob, entropy, value

    def prep_rollout(self):
        """Set models to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
        self.state_mlp.eval()
        self.fusion_mlp.eval()

    def prep_training(self):
        """Set models to training mode."""
        self.actor.train()
        self.critic.train()
        self.state_mlp.train()
        self.fusion_mlp.train()

    def save(self, path: Path):
        """Saves the model and optimizer states."""
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_mlp_state_dict': self.state_mlp.state_dict(),
            'fusion_mlp_state_dict': self.fusion_mlp.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'action_log_std': self.action_log_std,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path / "mlp_policy.pt")

    def load(self, path: Path):
        """Loads the model and optimizer states."""
        checkpoint = torch.load(path / "mlp_policy.pt", map_location=self.device)
        self.state_mlp.load_state_dict(checkpoint['state_mlp_state_dict'])
        self.fusion_mlp.load_state_dict(checkpoint['fusion_mlp_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.action_log_std.data.copy_(checkpoint['action_log_std'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class MLPPPO:
    def __init__(self, all_args, policy: MLPPolicy):
        self.args = all_args
        self.policy = policy
        self.ppo_clip = 0.2
        self.ppo_grad_norm = 10.0
        self.ppo_entropy_coef = self.args.alg_entropy_coef
        self.ppo_huber_delta = 10.0
        self.tpdv = self.policy.tpdv

    def train_ppo_step(self, idx, total, batch):
        obs_image, obs_state, actions, value_preds, returns, masks, old_logprob, advantages = batch

        obs = dict(
            image=torch.tensor(obs_image).to(**self.tpdv),
            state=torch.tensor(obs_state).to(**self.tpdv)
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
            grad_norm = nn.utils.clip_grad_norm_(self.policy.trainable_params, self.ppo_grad_norm)
            self.policy.optimizer.step()
            self.policy.optimizer.zero_grad()
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
