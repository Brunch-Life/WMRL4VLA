import d3rlpy
import os
import torch
import numpy as np
import torch.nn as nn
import torchvision
from torchvision import transforms
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import Pose
import dataclasses
from d3rlpy.models.encoders import register_encoder_factory
from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles, quaternion_to_matrix, euler_angles_to_matrix

# 自定义 ResNet18 编码器
class ResNet18Encoder(nn.Module):
    def __init__(self, observation_shape, feature_size=512):
        super().__init__()
        self.feature_size = feature_size
        # 加载预训练的 ResNet18，去掉最后的分类层
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        # 冻结预训练参数（可选）
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # 预处理变换
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 最终的特征映射层
        self.fc = nn.Linear(512, feature_size)
        
    def forward(self, x):
        # x shape: (batch_size, 480, 640, 3) - HWC格式
        # d3rlpy 在 predict 时，输入的 observation 是 (1, H, W, C)
        # 但在内部处理时，可能会变成 (1, 1, H, W, C) 或 (N, 1, H, W, C)
        # 我们需要确保输入到 permute 前是 (N, H, W, C)
        if x.ndim == 5 and x.shape[1] == 1: # (N, 1, H, W, C)
            x = x.squeeze(1) # (N, H, W, C)
        elif x.ndim == 4: # (N, H, W, C)
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")


        batch_size = x.shape[0]
        
        # 转换为CHW格式：(batch_size, H, W, C) -> (batch_size, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # 预处理并提取特征
        x = self.preprocess(x)
        features = self.resnet(x)
        features = features.view(batch_size, -1)
        
        return self.fc(features)

# 为支持 actor-critic 的动作条件编码器
class ResNet18EncoderWithAction(nn.Module):
    def __init__(self, observation_shape, action_size, feature_size=512):
        super().__init__()
        self.feature_size = feature_size
        self.resnet_encoder = ResNet18Encoder(observation_shape, 256)
        self.action_fc = nn.Linear(action_size, 256)
        self.final_fc = nn.Linear(512, feature_size)
        
    def forward(self, x, action):
        # x shape: (batch_size, 1, 480, 640, 3)
        # action shape: (batch_size, action_size)
        obs_features = self.resnet_encoder(x)  # (batch_size, 256)
        action_features = self.action_fc(action)  # (batch_size, 256)
        combined = torch.cat([obs_features, action_features], dim=1)  # (batch_size, 512)
        return self.final_fc(combined)

@dataclasses.dataclass()
class ResNet18EncoderFactory(d3rlpy.models.EncoderFactory):
    feature_size: int = 512

    def create(self, observation_shape):
        # observation_shape 在 predict 时可能是 (1, H, W, C) 或 (H, W, C)
        # ResNet18Encoder 期望的是 (N, H, W, C)
        # 但 d3rlpy 内部传递给 create 的 observation_shape 是去除 batch 维度后的
        # 例如 (480, 640, 3)
        return ResNet18Encoder(observation_shape, self.feature_size)

    def create_with_action(self, observation_shape, action_size, discrete_action=False):
        return ResNet18EncoderWithAction(observation_shape, action_size, self.feature_size)

    @staticmethod
    def get_type() -> str:
        return "resnet18"

class CQLInference:
    def __init__(
        self,
        saved_model_path: str = "/nvme_data/bingwen/Documents/temp/rl/model_16400.d3",
        device: str = "cuda:0",
    ) -> None:
        self.device = device
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        register_encoder_factory(ResNet18EncoderFactory)
        self.policy = d3rlpy.load_learnable(saved_model_path, device=device)
        self.command_action = None # only for pd_ee_pose

    def pre_init(self, env: BaseEnv) -> None:
        B = env.unwrapped.agent.ee_pose_at_robot_base.shape[0]
        tcp_pose = env.unwrapped.agent.ee_pose_at_robot_base
        self.init_pose_mat = tcp_pose.to_transformation_matrix().cpu().numpy()
        self.command_action = torch.concat([tcp_pose.p, 
                                            matrix_to_euler_angles(quaternion_to_matrix(tcp_pose.q), "XYZ"),
                                            torch.ones(B, 1).to(self.device)], dim=-1)

    def step(
        self, env: BaseEnv, images: torch.Tensor, task_description: list[str]
    ) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor]]:
        # inference
        observation = images.cpu().numpy()
        raw_actions = self.policy.predict(observation) # observation, numpy [B, 480, 640, 3]
        # raw_actions -> shape [B, 8]
        # actios -> shape [B, 7]
        euler_angles = matrix_to_euler_angles(quaternion_to_matrix(torch.from_numpy(raw_actions[:, 3:7])), "XYZ").cpu().numpy()  # [B, 3]
        # concatenate euler angles with raw actions
        actions = np.concatenate([raw_actions[:, :3], euler_angles, raw_actions[:, -1:]], axis=-1) # [B, 7]
        return raw_actions, actions

    def get_abs_actions_via_delta_actions(self, delta_actions: torch.Tensor) -> torch.Tensor:
        delta_xyz = delta_actions[:, :3]  # [B, 3]
        delta_euler = delta_actions[:, 3:6] # [B, 3]
        gripper_pos = delta_actions[:, -1:] # [B, 1]
        # Compute target xyz
        command_xyz = self.command_action[:, :3] + delta_xyz
        # Compute target euler angles
        delta_rotation = euler_angles_to_matrix(delta_euler,'XYZ')
        command_rotation = torch.matmul(euler_angles_to_matrix(self.command_action[:, 3:6],'XYZ'), delta_rotation)
        command_euler = matrix_to_euler_angles(command_rotation, 'XYZ')
        # Compute target gripper
        command_gripper = gripper_pos
        # Concat
        actions = torch.cat([command_xyz, command_euler, command_gripper], dim=-1)  # [B, 7]
        self.command_action = actions
        return actions

    def reset(self, task_description: str) -> None:
        pass

if __name__ == "__main__":
    cql = CQLInference()
    images = np.random.rand(5, 480, 640, 3).astype(np.float32)
    raw_actions, actions = cql.step(None, torch.from_numpy(images), None)
    print(actions)
    print(f"shape: {actions.shape}")