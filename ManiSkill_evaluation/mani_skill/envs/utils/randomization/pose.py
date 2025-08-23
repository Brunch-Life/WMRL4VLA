import numpy as np
import torch

from transforms3d.euler import euler2quat
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from mani_skill.utils.structs.types import Device


def random_quaternions(
    n: int,
    device: Device = None,
    lock_x: bool = False,
    lock_y: bool = False,
    lock_z: bool = False,
    bounds=(0, np.pi * 2),
):
    """
    Generates random quaternions by generating random euler angles uniformly, with each of
    the X, Y, Z angles ranging from bounds[0] to bounds[1] radians. Can optionally
    choose to fix X, Y, and/or Z euler angles to 0 via lock_x, lock_y, lock_z arguments
    """
    dist = bounds[1] - bounds[0]
    xyz_angles = torch.rand((n, 3), device=device) * (dist) + bounds[0]
    if lock_x:
        xyz_angles[:, 0] *= 0
    if lock_y:
        xyz_angles[:, 1] *= 0
    if lock_z:
        xyz_angles[:, 2] *= 0
    return matrix_to_quaternion(euler_angles_to_matrix(xyz_angles, convention="XYZ"))

def generate_random_pos(xy_center, fixed_z_value, half_edge_length_x, half_edge_length_y):
    """
    Generate a random position within a rectangular region in the XY plane at a fixed Z height.

    Parameters:
    xy_center (array, tuple or list of float): The (x, y) coordinates of the center of the rectangle.
    z_value (float): The fixed Z-coordinate of the generated position.
    half_edge_length_x (float): Half of the rectangle's edge length along the X-axis.
    half_edge_length_y (float): Half of the rectangle's edge length along the Y-axis.

    Returns:
    np.ndarray: A NumPy array of shape (3,) representing the (x, y, z) coordinates of the random position.
    """
    random_x = np.random.uniform(xy_center[0] - half_edge_length_x, xy_center[0] + half_edge_length_x)
    random_y = np.random.uniform(xy_center[1] - half_edge_length_y, xy_center[1] + half_edge_length_y)
    return np.array([random_x, random_y, fixed_z_value])

def generate_random_pos_batch(xy_center, fixed_z_value, half_edge_length_x, half_edge_length_y, batch_size):
    random_x = np.random.uniform(
        xy_center[0] - half_edge_length_x, xy_center[0] + half_edge_length_x, size=(batch_size,)
    )
    random_y = np.random.uniform(
        xy_center[1] - half_edge_length_y, xy_center[1] + half_edge_length_y, size=(batch_size,)
    )
    z = np.full((batch_size,), fixed_z_value)
    return np.stack([random_x, random_y, z], axis=1)  # (b, 3)

def get_objs_random_pose_batch(xy_center, half_edge_length_x, half_edge_length_y, z_value,
                               extents_x, extents_y, quats, b, threshold_scale=1.0):
    xy_center = np.array(xy_center)
    half_edge_length_x = np.array(half_edge_length_x)
    half_edge_length_y = np.array(half_edge_length_y)
    z_value = np.array(z_value)
    extents_x = np.array(extents_x)
    extents_y = np.array(extents_y)

    if np.linalg.norm([half_edge_length_x[0], half_edge_length_y[0]]) >= np.linalg.norm([half_edge_length_x[1], half_edge_length_y[1]]):
        first_idx, second_idx = 1, 0  
    else:
        first_idx, second_idx = 0, 1

    half_extents_x = extents_x / 2.0
    half_extents_y = extents_y / 2.0

    # threshold = threshold_scale * (max(half_extents_x) + max(half_extents_y)) # 规则长方体/正方体
    threshold = threshold_scale * np.linalg.norm([2 * max(half_extents_x), 2 * max(half_extents_y)]) # 物体的形状不规则
    # threshold = threshold_scale * np.mean([half_extents_x[0] + half_extents_x[1], half_extents_y[0] + half_extents_y[1]]) # 物体大小相差比较大

    # Generate batch positions for first object
    first_obj_xyz = generate_random_pos_batch(
        xy_center[first_idx], z_value[first_idx],
        half_edge_length_x[first_idx], half_edge_length_y[first_idx], b
    )

    second_obj_xyz = np.zeros((b, 3))
    n_candidates = 500
    for i in range(b):
        candidates = generate_random_pos_batch(
            xy_center[second_idx], z_value[second_idx],
            half_edge_length_x[second_idx], half_edge_length_y[second_idx],
            n_candidates
        )
        distances = np.linalg.norm(candidates[:, :2] - first_obj_xyz[i, :2], axis=1)

        valid_idx = np.where(distances >= threshold)[0]
        if len(valid_idx) > 0:
            second_obj_xyz[i] = candidates[valid_idx[0]]
        else:
            max_idx = np.argmax(distances)
            second_obj_xyz[i] = candidates[max_idx]


    # Reorder to source/target
    if first_idx == 0:
        source_obj_xyz = first_obj_xyz
        target_obj_xyz = second_obj_xyz
    else:
        source_obj_xyz = second_obj_xyz
        target_obj_xyz = first_obj_xyz

    # Batch quats
    source_obj_quat = np.array([
        q if q is not None else euler2quat(0, 0, np.random.uniform(-np.pi, np.pi), "sxyz")
        for q in [quats[0]] * b
    ])
    target_obj_quat = np.array([
        q if q is not None else euler2quat(0, 0, np.random.uniform(-np.pi, np.pi), "sxyz")
        for q in [quats[1]] * b
    ])

    return (
        torch.from_numpy(source_obj_xyz).float(),
        torch.from_numpy(source_obj_quat).float(),
        torch.from_numpy(target_obj_xyz).float(),
        torch.from_numpy(target_obj_quat).float(),
    )


def get_single_obj_grid_pose_batch(xy_center, half_edge_length_x, half_edge_length_y, z_value, step_size, batch_size, episode_idx=None):
    xy_center = np.array(xy_center)
    x_range = np.array([-half_edge_length_x, half_edge_length_x])
    y_range = np.array([-half_edge_length_y, half_edge_length_y])

    # x_vals = np.random.uniform(x_range[0]+xy_center[0], x_range[1]+xy_center[0], 110)
    # y_vals = np.random.uniform(y_range[0]+xy_center[1], y_range[1]+xy_center[1], 110)

    x_vals = np.arange(x_range[0] + xy_center[0], x_range[1] + xy_center[0] + 1e-6, step_size)
    y_vals = np.arange(y_range[0] + xy_center[1], y_range[1] + xy_center[1] + 1e-6, step_size)

    X, Y = np.meshgrid(x_vals, y_vals)
    grid_positions = np.vstack([X.ravel(), Y.ravel()]).T  # (N, 2)

    total_positions = grid_positions.shape[0]
    if isinstance(episode_idx, int):
        pos_ids = (np.arange(batch_size) + episode_idx) % total_positions
    else:
        pos_ids = np.random.choice(total_positions, size=batch_size, replace=False)

    xy_batch = grid_positions[pos_ids]
    z_batch = np.zeros((batch_size, 1)) + z_value
    xyz_batch = np.hstack([xy_batch, z_batch])  # (b, 3)

    # default unit quaternion [1, 0, 0, 0] for all
    quat_batch = np.tile(np.array([1, 0, 0, 0]), (batch_size, 1))  # (b, 4)

    return xyz_batch, quat_batch




def create_position_grid(xy_center, half_edge_length_x, half_edge_length_y, grid_size):
    """创建位置网格"""
    x_min = xy_center[0] - half_edge_length_x
    x_max = xy_center[0] + half_edge_length_x
    y_min = xy_center[1] - half_edge_length_y
    y_max = xy_center[1] + half_edge_length_y
    
    x_positions = np.linspace(x_min, x_max, grid_size)
    y_positions = np.linspace(y_min, y_max, grid_size)
    
    # 创建所有可能的位置组合
    positions = []
    for x in x_positions:
        for y in y_positions:
            positions.append([x, y])
    
    positions = np.array(positions)
    print(f"Grid created: center={xy_center}, x_range=[{x_min:.2f}, {x_max:.2f}], y_range=[{y_min:.2f}, {y_max:.2f}]")
    print(f"Grid size: {grid_size}x{grid_size} = {len(positions)} positions")
    return positions

def generate_deterministic_position(xy_center, half_edge_length_x, half_edge_length_y, 
                                   config_id, max_positions=100000):
    """
    基于config_id生成确定性位置,提供更多变化
    使用伪随机但确定性的方法在区域内生成位置
    """
    # 使用config_id作为种子来生成确定性的"随机"位置
    # 这里使用简单的线性同余生成器原理
    seed = config_id
    
    # 生成x坐标
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    x_ratio = (seed % max_positions) / max_positions
    x = xy_center[0] - half_edge_length_x + 2 * half_edge_length_x * x_ratio
    
    # 生成y坐标
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    y_ratio = (seed % max_positions) / max_positions
    y = xy_center[1] - half_edge_length_y + 2 * half_edge_length_y * y_ratio
    
    return np.array([x, y])

def get_objs_deterministic_pose_batch(xy_center, half_edge_length_x, half_edge_length_y, z_value,
                                    extents_x, extents_y, quats, b, 
                                    config_ids, grid_size=100, 
                                    predefined_quats=None, threshold_scale=0.8,
                                    use_continuous_positions=True, verbose=False):
    """
    基于config_ids生成确定性的物体位置和姿态 # config_id = episode_id
    
    Args:
        config_ids: 配置ID列表,长度为b
        grid_size: 网格大小,决定每个维度有多少个离散位置(仅在use_continuous_positions=False时使用)
        predefined_quats: 预定义的四元数列表,如果为None则使用固定的角度
        use_continuous_positions: 如果True,使用连续位置生成;如果False,使用网格位置
    """
    xy_center = np.array(xy_center)
    half_edge_length_x = np.array(half_edge_length_x)
    half_edge_length_y = np.array(half_edge_length_y)
    z_value = np.array(z_value)
    extents_x = np.array(extents_x)
    extents_y = np.array(extents_y)
    
    # 确定哪个物体先放置（基于区域大小）
    if np.linalg.norm([half_edge_length_x[0], half_edge_length_y[0]]) >= np.linalg.norm([half_edge_length_x[1], half_edge_length_y[1]]):
        first_idx, second_idx = 1, 0  
    else:
        first_idx, second_idx = 0, 1

    if verbose:
        print(f"First object index: {first_idx}, Second object index: {second_idx}")
    
    half_extents_x = extents_x / 2.0
    half_extents_y = extents_y / 2.0
    threshold = threshold_scale * np.linalg.norm([2 * max(half_extents_x), 2 * max(half_extents_y)])
    if verbose:
        print(f"Distance threshold: {threshold:.3f}")
    
    # 创建固定的预定义四元数（确定性）
    if predefined_quats is None:
        # 创建一些固定的旋转角度
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)  # 16个方向
        predefined_quats = [euler2quat(0, 0, angle, "sxyz") for angle in angles]
        if verbose:
            print(f"Created {len(predefined_quats)} predefined quaternions")
    
    first_obj_xyz = np.zeros((b, 3))
    second_obj_xyz = np.zeros((b, 3))
    source_obj_quat = np.zeros((b, 4))
    target_obj_quat = np.zeros((b, 4))
    
    if use_continuous_positions:
        # 使用连续位置生成
        for i in range(b):
            config_id = config_ids[i]
            if verbose:
                print(f"\nProcessing batch {i}, config_id: {config_id}")
            
            # 生成第一个物体的位置
            first_pos_2d = generate_deterministic_position(
                xy_center[first_idx], half_edge_length_x[first_idx], 
                half_edge_length_y[first_idx], config_id
            )
            first_obj_xyz[i] = [first_pos_2d[0], first_pos_2d[1], z_value[first_idx]]
            if verbose:
                print(f"First object (idx={first_idx}) position: {first_obj_xyz[i]}")
            
            # 为第二个物体找位置,尝试多个候选位置
            max_attempts = 1000
            placed = False
            best_pos_2d = None
            best_distance = 0
            
            for attempt in range(max_attempts):
                # 使用不同的种子来生成候选位置
                candidate_id = config_id * 1000 + attempt
                candidate_pos_2d = generate_deterministic_position(
                    xy_center[second_idx], half_edge_length_x[second_idx], 
                    half_edge_length_y[second_idx], candidate_id
                )
                
                distance = np.linalg.norm(candidate_pos_2d - first_pos_2d)
                
                if distance >= threshold:
                    second_obj_xyz[i] = [candidate_pos_2d[0], candidate_pos_2d[1], z_value[second_idx]]
                    placed = True
                    if verbose:
                        print(f"  ✓ Placed at {second_obj_xyz[i]} after {attempt + 1} attempts, distance: {distance:.3f}")
                    break
                
                # 记录最佳候选位置
                if distance > best_distance:
                    best_distance = distance
                    best_pos_2d = candidate_pos_2d
            
            # 如果没有找到满足约束的位置,使用最远的位置
            if not placed:
                second_obj_xyz[i] = [best_pos_2d[0], best_pos_2d[1], z_value[second_idx]]
                if verbose:
                    print(f"  ⚠ No valid position found, using best: {second_obj_xyz[i]}, distance: {best_distance:.3f}")
            
            # 确定性地选择四元数
            quat_idx = config_id % len(predefined_quats)
            first_quat = predefined_quats[quat_idx]
        
            quat_idx = (config_id // len(predefined_quats)) % len(predefined_quats)
            second_quat = predefined_quats[quat_idx]
            
            # 根据first_idx分配到source/target
            if first_idx == 0:
                source_obj_quat[i] = first_quat
                target_obj_quat[i] = second_quat
            else:
                source_obj_quat[i] = second_quat
                target_obj_quat[i] = first_quat

            if quats[0] is not None:
                source_obj_quat[i] = quats[0]
            if quats[1] is not None:
                target_obj_quat[i] = quats[1]
    
    else:
        # 使用原来的网格方法
        # 为两个物体创建位置网格
        first_obj_grid = create_position_grid(
            xy_center[first_idx], half_edge_length_x[first_idx], 
            half_edge_length_y[first_idx], grid_size
        )
        second_obj_grid = create_position_grid(
            xy_center[second_idx], half_edge_length_x[second_idx], 
            half_edge_length_y[second_idx], grid_size
        )
        
        for i in range(b):
            config_id = config_ids[i]
            print(f"\nProcessing batch {i}, config_id: {config_id}")
            
            # 使用config_id来确定性地选择位置
            first_pos_idx = config_id % len(first_obj_grid)
            first_pos_2d = first_obj_grid[first_pos_idx]
            first_obj_xyz[i] = [first_pos_2d[0], first_pos_2d[1], z_value[first_idx]]
            print(f"First object (idx={first_idx}) position: {first_obj_xyz[i]}")
            
            # 为第二个物体找到满足距离约束的位置
            config_offset = (config_id // len(first_obj_grid)) % len(second_obj_grid)
            print(f"Second object config_offset: {config_offset}")
            
            # 检查所有可能的第二个物体位置,从config_offset开始
            placed = False
            for j in range(len(second_obj_grid)):
                second_pos_idx = (config_offset + j) % len(second_obj_grid)
                second_pos_2d = second_obj_grid[second_pos_idx]
                
                # 检查距离约束
                distance = np.linalg.norm(second_pos_2d - first_pos_2d)
                if distance >= threshold:
                    second_obj_xyz[i] = [second_pos_2d[0], second_pos_2d[1], z_value[second_idx]]
                    placed = True
                    print(f"  ✓ Placed at {second_obj_xyz[i]}")
                    break
            
            # 如果没有找到满足约束的位置,选择距离最远的位置
            if not placed:
                distances = np.linalg.norm(second_obj_grid - first_pos_2d, axis=1)
                best_idx = np.argmax(distances)
                best_pos_2d = second_obj_grid[best_idx]
                second_obj_xyz[i] = [best_pos_2d[0], best_pos_2d[1], z_value[second_idx]]
                print(f"  ⚠ No valid position found, using farthest: {second_obj_xyz[i]}")
            
            # 确定性地选择四元数
            if quats[0] is not None:
                first_quat = quats[0]
            else:
                quat_idx = config_id % len(predefined_quats)
                first_quat = predefined_quats[quat_idx]
            
            if quats[1] is not None:
                second_quat = quats[1]
            else:
                quat_idx = (config_id // len(predefined_quats)) % len(predefined_quats)
                second_quat = predefined_quats[quat_idx]
            
            # 根据first_idx分配到source/target
            if first_idx == 0:
                source_obj_quat[i] = first_quat
                target_obj_quat[i] = second_quat
            else:
                source_obj_quat[i] = second_quat
                target_obj_quat[i] = first_quat
    
    # 重新排序位置以匹配source/target
    if first_idx == 0:
        source_obj_xyz = first_obj_xyz
        target_obj_xyz = second_obj_xyz
    else:
        source_obj_xyz = second_obj_xyz
        target_obj_xyz = first_obj_xyz
    
    return (
        torch.from_numpy(source_obj_xyz).float(),
        torch.from_numpy(source_obj_quat).float(),
        torch.from_numpy(target_obj_xyz).float(),
        torch.from_numpy(target_obj_quat).float(),
    )

def get_objs_mixed_pose_batch(xy_center, half_edge_length_x, half_edge_length_y, z_value,
                             extents_x, extents_y, quats, b, 
                             config_ids=None, use_deterministic=True, 
                             grid_size=100, predefined_quats=None, threshold_scale=0.8):
    """
    混合模式：可以选择使用确定性或随机生成
    
    Args:
        config_ids: 如果提供且use_deterministic=True, 则使用确定性生成
        use_deterministic: 是否使用确定性生成
    """
    if use_deterministic and config_ids is not None:
        return get_objs_deterministic_pose_batch(
            xy_center, half_edge_length_x, half_edge_length_y, z_value,
            extents_x, extents_y, quats, b, config_ids, 
            grid_size, predefined_quats, threshold_scale
        )
    else:
        # 如果没有提供config_ids或选择随机模式,生成随机config_ids
        if config_ids is None:
            config_ids = np.random.randint(0, 10000, size=b)
        return get_objs_deterministic_pose_batch(
            xy_center, half_edge_length_x, half_edge_length_y, z_value,
            extents_x, extents_y, quats, b, config_ids, 
            grid_size, predefined_quats, threshold_scale
        )

# 使用示例 - 调试版本
if __name__ == "__main__":
    # 参数设置
    xy_center = [[0, 0], [1, 1]]
    half_edge_length_x = [2.0, 1.5]
    half_edge_length_y = [2.0, 1.5]
    z_value = [0.5, 0.5]
    extents_x = [0.3, 0.2]
    extents_y = [0.3, 0.2]
    quats = [None, None]  # 使用随机四元数
    b = 3  # 减少批量用于调试
    
    print("=== 参数信息 ===")
    print(f"xy_center: {xy_center}")
    print(f"half_edge_length_x: {half_edge_length_x}")
    print(f"half_edge_length_y: {half_edge_length_y}")
    print(f"extents_x: {extents_x}, extents_y: {extents_y}")
    
    # 生成确定性的config_ids
    config_ids = [101, 102, 103]
    
    print(f"\n=== 生成位置 (config_ids: {config_ids}) ===")
    # 生成位置
    source_xyz, source_quat, target_xyz, target_quat = get_objs_deterministic_pose_batch(
        xy_center, half_edge_length_x, half_edge_length_y, z_value,
        extents_x, extents_y, quats, b, config_ids, grid_size=100, use_continuous_positions=True,
    )
    
    print("\n=== 结果 ===")
    print("Source positions:", source_xyz)
    print("Target positions:", target_xyz)
    print("Source quaternions:", source_quat)
    print("Target quaternions:", target_quat)
    
    print("\n=== 可复现性测试 ===")
    # 验证可复现性 - 相同的config_id应该产生相同的结果
    source_xyz_2, source_quat_2, target_xyz_2, target_quat_2 = get_objs_deterministic_pose_batch(
        xy_center, half_edge_length_x, half_edge_length_y, z_value,
        extents_x, extents_y, quats, b, config_ids, grid_size=100,use_continuous_positions=True,
    )
    
    print("Position reproducibility check:")
    print("Source positions match:", torch.allclose(source_xyz, source_xyz_2, atol=1e-6))
    print("Target positions match:", torch.allclose(target_xyz, target_xyz_2, atol=1e-6))
    print("Source quaternions match:", torch.allclose(source_quat, source_quat_2, atol=1e-6))
    print("Target quaternions match:", torch.allclose(target_quat, target_quat_2, atol=1e-6))