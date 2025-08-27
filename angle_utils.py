"""
angle_utils.py

处理欧拉角的相位偏移和恢复，解决-π/π边界跳跃问题
核心思想：
1. 训练时：将-π~π范围的角度加π，然后wrap到-π~π，等效于将不连续点从±π移到0
2. 推理时：将预测结果减π，然后wrap到-π~π，恢复原始角度空间
"""

import numpy as np
import torch
from typing import Union


def wrap_angle_to_pi(angle: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    将角度wrap到[-π, π]范围内
    
    Args:
        angle: 角度值（弧度）
    Returns:
        wrapped_angle: wrap后的角度值
    """
    if isinstance(angle, torch.Tensor):
        return torch.atan2(torch.sin(angle), torch.cos(angle))
    else:
        return np.arctan2(np.sin(angle), np.cos(angle))


def apply_phase_shift_for_training(euler_angles: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    对euler_x和euler_z应用相位偏移，用于训练
    将原本的-π~π范围加π后wrap，使不连续点从±π移动到0
    
    Args:
        euler_angles: (..., 3) 形状的欧拉角 [euler_x, euler_y, euler_z]
    Returns:
        shifted_angles: 相位偏移后的角度
    """
    if isinstance(euler_angles, torch.Tensor):
        shifted_angles = euler_angles.clone()
    else:
        shifted_angles = euler_angles.copy()
    
    # 对euler_x (索引0) 和 euler_z (索引2) 应用相位偏移
    # euler_y已经在0附近，不需要处理
    shifted_angles[..., 0] = wrap_angle_to_pi(euler_angles[..., 0] + np.pi)  # euler_x
    shifted_angles[..., 2] = wrap_angle_to_pi(euler_angles[..., 2] + np.pi)  # euler_z
    
    return shifted_angles


def restore_phase_shift_for_action(euler_angles: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    恢复相位偏移，用于将训练后的预测结果转换回原始角度空间
    
    Args:
        euler_angles: (..., 3) 形状的相位偏移后的欧拉角
    Returns:
        restored_angles: 恢复到原始空间的角度
    """
    if isinstance(euler_angles, torch.Tensor):
        restored_angles = euler_angles.clone()
    else:
        restored_angles = euler_angles.copy()
    
    # 对euler_x (索引0) 和 euler_z (索引2) 恢复相位偏移
    restored_angles[..., 0] = wrap_angle_to_pi(euler_angles[..., 0] - np.pi)  # euler_x
    restored_angles[..., 2] = wrap_angle_to_pi(euler_angles[..., 2] - np.pi)  # euler_z
    
    return restored_angles


def preprocess_action_for_training(actions: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    预处理action用于训练
    对7维action的euler角部分应用相位偏移
    
    Args:
        actions: (..., 7) 形状的action [pos_x, pos_y, pos_z, euler_x, euler_y, euler_z, gripper]
    Returns:
        processed_actions: 处理后的action
    """
    if isinstance(actions, torch.Tensor):
        processed_actions = actions.clone()
    else:
        processed_actions = actions.copy()
    
    # 提取euler角部分 (索引3:6)
    euler_part = actions[..., 3:6]
    
    # 应用相位偏移
    shifted_euler = apply_phase_shift_for_training(euler_part)
    
    # 更新action
    processed_actions[..., 3:6] = shifted_euler
    
    return processed_actions


def postprocess_action_for_env(actions: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    后处理action用于环境执行
    将训练空间的action恢复到原始角度空间
    
    Args:
        actions: (..., 7) 形状的训练空间action
    Returns:
        env_actions: 适用于环境的action
    """
    if isinstance(actions, torch.Tensor):
        env_actions = actions.clone()
    else:
        env_actions = actions.copy()
    
    # 提取euler角部分 (索引3:6)
    euler_part = actions[..., 3:6]
    
    # 恢复相位偏移
    restored_euler = restore_phase_shift_for_action(euler_part)
    
    # 更新action
    env_actions[..., 3:6] = restored_euler
    
    return env_actions


def robust_unwrap_angles(angles: np.ndarray, threshold: float = np.pi) -> np.ndarray:
    """
    鲁棒的角度unwrap，处理角度序列中的跳跃
    
    Args:
        angles: 1D角度序列
        threshold: 跳跃检测阈值
    Returns:
        unwrapped_angles: unwrap后的角度序列
    """
    if len(angles) <= 1:
        return angles.copy()
    
    unwrapped = angles.copy()
    
    for i in range(1, len(angles)):
        diff = angles[i] - unwrapped[i-1]
        
        # 检测跳跃并修正
        if diff > threshold:
            unwrapped[i:] -= 2 * np.pi
        elif diff < -threshold:
            unwrapped[i:] += 2 * np.pi
    
    return unwrapped


def test_phase_shift_functions():
    """测试相位偏移功能的正确性"""
    print("Testing angle phase shift functions...")
    
    # 测试数据：包含边界值的角度
    test_angles = np.array([
        [-np.pi, 0, np.pi],      # 边界情况
        [-np.pi/2, 0.1, np.pi/2], # 正常情况
        [3.1, -0.05, -3.1],     # 接近边界
    ])
    
    print("Original angles:")
    print(test_angles)
    
    # 应用相位偏移
    shifted = apply_phase_shift_for_training(test_angles)
    print("\nAfter phase shift:")
    print(shifted)
    
    # 恢复相位偏移
    restored = restore_phase_shift_for_action(shifted)
    print("\nAfter restoration:")
    print(restored)
    
    # 检查是否正确恢复
    diff = np.abs(wrap_angle_to_pi(test_angles) - wrap_angle_to_pi(restored))
    print(f"\nMax difference after round-trip: {np.max(diff):.6f}")
    
    if np.max(diff) < 1e-6:
        print("✅ Phase shift functions work correctly!")
    else:
        print("❌ Phase shift functions have errors!")
    
    # 测试完整的action处理
    print("\n" + "="*50)
    print("Testing full action processing...")
    
    test_actions = np.array([
        [0.4, 0.0, 0.1, -np.pi, 0.0, np.pi, 1.0],      # 边界角度
        [0.5, 0.1, 0.2, -np.pi/2, 0.001, np.pi/2, -1.0], # 正常角度
    ])
    
    print("Original actions:")
    print(test_actions)
    
    # 训练预处理
    train_actions = preprocess_action_for_training(test_actions)
    print("\nPreprocessed for training:")
    print(train_actions)
    
    # 环境后处理
    env_actions = postprocess_action_for_env(train_actions)
    print("\nPostprocessed for environment:")
    print(env_actions)
    
    # 检查位置和夹爪维度是否不变
    pos_gripper_diff = np.max(np.abs(test_actions[:, [0,1,2,6]] - env_actions[:, [0,1,2,6]]))
    euler_diff = np.max(np.abs(wrap_angle_to_pi(test_actions[:, 3:6]) - wrap_angle_to_pi(env_actions[:, 3:6])))
    
    print(f"\nPosition & gripper max diff: {pos_gripper_diff:.6f}")
    print(f"Euler angles max diff: {euler_diff:.6f}")
    
    if pos_gripper_diff < 1e-6 and euler_diff < 1e-6:
        print("✅ Action processing functions work correctly!")
    else:
        print("❌ Action processing functions have errors!")


if __name__ == "__main__":
    test_phase_shift_functions()
