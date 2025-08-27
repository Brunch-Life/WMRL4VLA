"""
simple_action_analysis.py

使用rlvla_env环境现有依赖的action分布分析脚本
只使用numpy和matplotlib，避免对seaborn和pandas的依赖
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import json

# 导入角度处理工具
from angle_utils import preprocess_action_for_training, postprocess_action_for_env

def load_dataset_actions(data_dir: Path, max_files: int = None, apply_preprocessing: bool = False):
    """
    加载数据集的所有action数据
    Args:
        data_dir: 数据集目录
        max_files: 最大加载文件数
        apply_preprocessing: 是否应用角度预处理
    Returns:
        all_actions: 原始或预处理后的action数据
        file_info: 文件信息
    """
    npz_files = sorted(glob.glob(str(data_dir / "*.npz")))
    if max_files:
        npz_files = npz_files[:max_files]
    
    processing_type = "with preprocessing" if apply_preprocessing else "original"
    print(f"Loading {len(npz_files)} NPZ files from {data_dir} ({processing_type})")
    
    all_actions = []
    file_info = []
    
    for file_path in tqdm(npz_files, desc=f"Loading files ({processing_type})"):
        try:
            data = np.load(file_path, allow_pickle=True)["arr_0"].tolist()
            
            # 解析action数据
            position = data["action"]["end"]["position"].squeeze(1)  # (T, 3)
            orientation_quat = data["action"]["end"]["orientation"].squeeze(1)  # (T, 4)
            gripper = data["action"]["effector"]["position_gripper"]  # (T, 1)
            
            # 将四元数转换为欧拉角
            euler_angles = R.from_quat(orientation_quat).as_euler('xyz', degrees=False)  # (T, 3)
            
            # 拼接：position (3) + euler (3) + gripper (1) = 7D action
            actions = np.concatenate([
                position,      # (T, 3)
                euler_angles,  # (T, 3) 
                gripper        # (T, 1)
            ], axis=1).astype(np.float32)  # (T, 7)
            
            # 如果需要，应用角度预处理
            if apply_preprocessing:
                actions = preprocess_action_for_training(actions)
            
            all_actions.append(actions)
            file_info.append({
                'file': Path(file_path).name,
                'num_steps': len(actions),
                'action_shape': actions.shape
            })
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if all_actions:
        all_actions = np.concatenate(all_actions, axis=0)
        print(f"Total actions loaded: {all_actions.shape}")
        return all_actions, file_info
    else:
        raise ValueError("No valid actions loaded!")

def compute_detailed_statistics(actions: np.ndarray):
    """计算详细的统计信息"""
    action_names = ['pos_x', 'pos_y', 'pos_z', 'euler_x', 'euler_y', 'euler_z', 'gripper']
    
    stats_dict = {}
    for i, name in enumerate(action_names):
        data = actions[:, i]
        
        # 基本统计
        stats_dict[name] = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75)),
            'range': float(np.max(data) - np.min(data)),
        }
        
        # 检查数据质量
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        outlier_threshold = 3 * iqr
        outliers = np.sum((data < q1 - outlier_threshold) | (data > q3 + outlier_threshold))
        stats_dict[name]['outlier_count'] = int(outliers)
        stats_dict[name]['outlier_percentage'] = float(outliers / len(data) * 100)
    
    return stats_dict

def create_comprehensive_visualization(actions: np.ndarray, output_dir: Path):
    """创建全面的可视化分析"""
    action_names = ['pos_x', 'pos_y', 'pos_z', 'euler_x', 'euler_y', 'euler_z', 'gripper']
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    
    # 1. 分布直方图
    fig1, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()
    
    for i, (name, color) in enumerate(zip(action_names, colors)):
        ax = axes[i]
        data = actions[:, i]
        
        # 绘制直方图
        n, bins, patches = ax.hist(data, bins=50, alpha=0.7, color=color, edgecolor='black', density=True)
        
        # 添加统计线
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.8, label=f'+1σ: {mean_val + std_val:.3f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.8, label=f'-1σ: {mean_val - std_val:.3f}')
        
        # 添加中位数
        median_val = np.median(data)
        ax.axvline(median_val, color='green', linestyle='-.', linewidth=1.5, label=f'Median: {median_val:.3f}')
        
        ax.set_title(f'{name} Distribution\n(Range: {np.min(data):.3f} to {np.max(data):.3f})', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{name} Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 删除多余的子图
    for i in range(7, 9):
        fig1.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detailed_action_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 箱线图对比
    fig2, ax = plt.subplots(figsize=(14, 8))
    
    # 标准化数据以便比较（每个维度减去均值除以标准差）
    normalized_actions = []
    for i in range(7):
        data = actions[:, i]
        normalized = (data - np.mean(data)) / np.std(data)
        normalized_actions.append(normalized)
    
    bp = ax.boxplot(normalized_actions, labels=action_names, patch_artist=True)
    
    # 设置颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Normalized Action Dimensions Comparison (Box Plots)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Action Dimension')
    ax.set_ylabel('Standardized Value (z-score)')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'action_boxplots_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 相关性热力图
    fig3, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = np.corrcoef(actions.T)
    
    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # 添加文本标注
    for i in range(len(action_names)):
        for j in range(len(action_names)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.5 else "white")
    
    ax.set_xticks(range(len(action_names)))
    ax.set_yticks(range(len(action_names)))
    ax.set_xticklabels(action_names, rotation=45)
    ax.set_yticklabels(action_names)
    ax.set_title('Action Dimensions Correlation Matrix', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'action_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 轨迹示例（前5个episode）
    fig4, axes = plt.subplots(4, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # 假设每200步为一个episode
    episode_length = 200
    max_episodes = min(5, len(actions) // episode_length)
    
    for i, (name, color) in enumerate(zip(action_names, colors)):
        ax = axes[i]
        for ep in range(max_episodes):
            start_idx = ep * episode_length
            end_idx = start_idx + episode_length
            if end_idx <= len(actions):
                episode_data = actions[start_idx:end_idx, i]
                ax.plot(episode_data, alpha=0.8, linewidth=1.5, label=f'Episode {ep+1}', color=plt.cm.Set1(ep))
        
        ax.set_title(f'{name} Trajectories', fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel(f'{name} Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 删除最后一个空的子图
    fig4.delaxes(axes[7])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'action_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ All visualizations saved to {output_dir}")

def create_preprocessing_comparison(original_actions: np.ndarray, processed_actions: np.ndarray, output_dir: Path):
    """
    创建预处理前后的对比可视化
    """
    action_names = ['pos_x', 'pos_y', 'pos_z', 'euler_x', 'euler_y', 'euler_z', 'gripper']
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    
    # 分布对比图
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, (name, color) in enumerate(zip(action_names, colors)):
        ax = axes[i]
        
        # 原始数据
        orig_data = original_actions[:, i]
        proc_data = processed_actions[:, i]
        
        # 绘制重叠的直方图
        ax.hist(orig_data, bins=50, alpha=0.6, color='blue', 
               label=f'Original (Range: {np.max(orig_data)-np.min(orig_data):.3f})', 
               density=True, edgecolor='black', linewidth=0.5)
        
        ax.hist(proc_data, bins=50, alpha=0.6, color='red', 
               label=f'Preprocessed (Range: {np.max(proc_data)-np.min(proc_data):.3f})', 
               density=True, edgecolor='black', linewidth=0.5)
        
        # 添加统计线
        orig_mean = np.mean(orig_data)
        proc_mean = np.mean(proc_data)
        
        ax.axvline(orig_mean, color='blue', linestyle='--', linewidth=2, alpha=0.8,
                  label=f'Orig Mean: {orig_mean:.3f}')
        ax.axvline(proc_mean, color='red', linestyle='--', linewidth=2, alpha=0.8,
                  label=f'Proc Mean: {proc_mean:.3f}')
        
        # 特别标注euler_x和euler_z的改善
        if name in ['euler_x', 'euler_z']:
            # 计算不连续性改善
            orig_range = np.max(orig_data) - np.min(orig_data)
            proc_range = np.max(proc_data) - np.min(proc_data)
            improvement = orig_range / proc_range if proc_range > 0 else float('inf')
            
            ax.text(0.02, 0.98, f'Range improvement: {improvement:.1f}x', 
                   transform=ax.transAxes, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                   verticalalignment='top')
        
        ax.set_title(f'{name} Distribution Comparison\n(Blue: Original, Red: Preprocessed)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{name} Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 删除多余的子图
    for i in range(7, 9):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'preprocessing_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 专门针对euler角的详细对比
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
    euler_names = ['euler_x', 'euler_y', 'euler_z']
    euler_indices = [3, 4, 5]
    
    for i, (name, idx) in enumerate(zip(euler_names, euler_indices)):
        ax = axes[i]
        
        orig_data = original_actions[:, idx]
        proc_data = processed_actions[:, idx]
        
        # 绘制密度图
        ax.hist(orig_data, bins=60, alpha=0.6, color='blue', 
               label='Original', density=True, edgecolor='black', linewidth=0.5)
        ax.hist(proc_data, bins=60, alpha=0.6, color='red', 
               label='Preprocessed', density=True, edgecolor='black', linewidth=0.5)
        
        # 标记关键点
        ax.axvline(-np.pi, color='purple', linestyle=':', alpha=0.8, label='-π')
        ax.axvline(np.pi, color='purple', linestyle=':', alpha=0.8, label='+π')
        ax.axvline(0, color='green', linestyle='--', alpha=0.8, label='0')
        
        # 统计信息
        orig_std = np.std(orig_data)
        proc_std = np.std(proc_data)
        
        ax.set_title(f'{name} Detailed Comparison\nSTD: {orig_std:.3f} → {proc_std:.3f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{name} Value (radians)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置x轴范围
        ax.set_xlim(-4, 4)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'euler_angles_detailed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Preprocessing comparison visualizations saved to {output_dir}")

def main():
    # 使用配置文件中的数据路径
    data_dir = Path("/home/chenyinuo/data/dataset/bingwen/data_for_success/green_bell_pepper_plate_wooden/success")
    output_dir = Path("action_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"🔍 Analyzing action distribution in: {data_dir}")
    print(f"📂 Results will be saved to: {output_dir}")
    
    # 加载原始数据
    print("\n📥 Loading original action data...")
    original_actions, file_info = load_dataset_actions(data_dir, max_files=50, apply_preprocessing=False)
    
    # 加载预处理后的数据
    print("\n📥 Loading preprocessed action data...")
    processed_actions, _ = load_dataset_actions(data_dir, max_files=50, apply_preprocessing=True)
    
    # 验证往返一致性（预处理后再后处理）
    print("\n🔄 Testing round-trip consistency...")
    restored_actions = postprocess_action_for_env(processed_actions)
    
    # 检查往返误差
    pos_gripper_orig = original_actions[:, [0,1,2,6]]
    pos_gripper_restored = restored_actions[:, [0,1,2,6]]
    pos_gripper_diff = np.max(np.abs(pos_gripper_orig - pos_gripper_restored))
    
    # 检查欧拉角误差（考虑周期性）
    def min_angle_diff(a1, a2):
        diff = np.abs(a1 - a2)
        return np.minimum(diff, 2*np.pi - diff)
    
    euler_orig = original_actions[:, 3:6]
    euler_restored = restored_actions[:, 3:6]
    euler_diff = np.max(min_angle_diff(euler_orig, euler_restored))
    
    print(f"   Position & gripper max difference: {pos_gripper_diff:.8f}")
    print(f"   Euler angles max difference: {euler_diff:.8f}")
    
    if pos_gripper_diff < 1e-5 and euler_diff < 1e-5:
        print("   ✅ Round-trip consistency verified!")
    else:
        print("   ⚠️  Round-trip consistency issues detected!")
    
    # 计算原始数据统计信息
    print("\n📊 Computing statistics for original data...")
    original_stats = compute_detailed_statistics(original_actions)
    
    # 计算预处理数据统计信息
    print("\n📊 Computing statistics for preprocessed data...")
    processed_stats = compute_detailed_statistics(processed_actions)
    
    # 生成原始数据可视化
    print("\n🎨 Creating visualizations for original data...")
    create_comprehensive_visualization(original_actions, output_dir)
    
    # 重命名原始可视化文件
    original_files = [
        'detailed_action_distributions.png',
        'action_boxplots_comparison.png', 
        'action_correlation_matrix.png',
        'action_trajectories.png'
    ]
    for file in original_files:
        old_path = output_dir / file
        new_path = output_dir / f"original_{file}"
        if old_path.exists():
            old_path.rename(new_path)
    
    # 生成预处理数据可视化
    print("\n🎨 Creating visualizations for preprocessed data...")
    create_comprehensive_visualization(processed_actions, output_dir)
    
    # 重命名预处理可视化文件
    for file in original_files:
        old_path = output_dir / file
        new_path = output_dir / f"preprocessed_{file}"
        if old_path.exists():
            old_path.rename(new_path)
    
    # 生成对比可视化
    print("\n🎨 Creating preprocessing comparison visualizations...")
    create_preprocessing_comparison(original_actions, processed_actions, output_dir)
    
    # 保存统计结果
    output_data = {
        'dataset_info': {
            'total_files': len(file_info),
            'total_steps': sum(info['num_steps'] for info in file_info),
            'action_shape': original_actions.shape,
            'files': file_info
        },
        'original_statistics': original_stats,
        'preprocessed_statistics': processed_stats,
        'round_trip_verification': {
            'position_gripper_max_diff': float(pos_gripper_diff),
            'euler_angles_max_diff': float(euler_diff),
            'consistency_check_passed': bool(pos_gripper_diff < 1e-5 and euler_diff < 1e-5)
        }
    }
    
    with open(output_dir / 'comprehensive_action_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 打印对比报告
    print_comparison_report(original_stats, processed_stats, original_actions, processed_actions)
    
    print(f"\n🎉 Complete analysis with preprocessing comparison finished!")
    print(f"📂 Results saved to: {output_dir}")
    print(f"🔍 Check the following files:")
    print(f"   - original_detailed_action_distributions.png (original data)")
    print(f"   - preprocessed_detailed_action_distributions.png (preprocessed data)")  
    print(f"   - preprocessing_comparison.png (side-by-side comparison)")
    print(f"   - euler_angles_detailed_comparison.png (detailed euler angles)")
    print(f"   - comprehensive_action_analysis.json (complete statistics)")

def print_detailed_report(stats_dict: dict, actions: np.ndarray):
    """打印详细的分析报告"""
    action_names = ['pos_x', 'pos_y', 'pos_z', 'euler_x', 'euler_y', 'euler_z', 'gripper']
    
    print("\n" + "="*100)
    print("COMPREHENSIVE ACTION DISTRIBUTION ANALYSIS REPORT")
    print("="*100)
    
    # 基本统计表格
    print(f"\n{'Dimension':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Range':<10} {'Outliers':<8}")
    print("-" * 88)
    
    for name in action_names:
        s = stats_dict[name]
        print(f"{name:<12} {s['mean']:<10.4f} {s['std']:<10.4f} {s['min']:<10.4f} "
              f"{s['max']:<10.4f} {s['range']:<10.4f} {s['outlier_percentage']:<8.1f}%")
    
    # 数据质量分析
    print("\n" + "="*100)
    print("DATA QUALITY ANALYSIS")
    print("="*100)
    
    print("🔍 Position Analysis:")
    pos_data = actions[:, :3]
    workspace_volume = np.prod([
        stats_dict['pos_x']['range'],
        stats_dict['pos_y']['range'], 
        stats_dict['pos_z']['range']
    ])
    print(f"   - Workspace volume: {workspace_volume:.6f} cubic units")
    print(f"   - Position center: ({np.mean(pos_data[:, 0]):.3f}, {np.mean(pos_data[:, 1]):.3f}, {np.mean(pos_data[:, 2]):.3f})")
    
    print("\n🔄 Orientation Analysis:")
    euler_data = actions[:, 3:6]
    print(f"   - Roll range: {stats_dict['euler_x']['range']:.3f} rad ({np.degrees(stats_dict['euler_x']['range']):.1f}°)")
    print(f"   - Pitch range: {stats_dict['euler_y']['range']:.3f} rad ({np.degrees(stats_dict['euler_y']['range']):.1f}°)")
    print(f"   - Yaw range: {stats_dict['euler_z']['range']:.3f} rad ({np.degrees(stats_dict['euler_z']['range']):.1f}°)")
    
    print("\n🤖 Gripper Analysis:")
    gripper_data = actions[:, 6]
    gripper_unique = np.unique(gripper_data)
    open_ratio = np.sum(gripper_data > 0) / len(gripper_data) * 100
    print(f"   - Unique values: {len(gripper_unique)} ({gripper_unique})")
    print(f"   - Open ratio: {open_ratio:.1f}% (gripper > 0)")
    print(f"   - Close ratio: {100-open_ratio:.1f}% (gripper ≤ 0)")
    
    # 相关性分析
    print("\n📈 Correlation Analysis:")
    corr_matrix = np.corrcoef(actions.T)
    high_correlations = []
    for i in range(len(action_names)):
        for j in range(i+1, len(action_names)):
            if abs(corr_matrix[i, j]) > 0.3:  # 显著相关性阈值
                high_correlations.append((action_names[i], action_names[j], corr_matrix[i, j]))
    
    if high_correlations:
        print("   - Significant correlations (|r| > 0.3):")
        for dim1, dim2, corr in high_correlations:
            print(f"     * {dim1} ↔ {dim2}: r = {corr:.3f}")
    else:
        print("   - No significant correlations found (all |r| ≤ 0.3)")

def print_comparison_report(original_stats: dict, processed_stats: dict, original_actions: np.ndarray, processed_actions: np.ndarray):
    """打印预处理前后的对比报告"""
    action_names = ['pos_x', 'pos_y', 'pos_z', 'euler_x', 'euler_y', 'euler_z', 'gripper']
    
    print("\n" + "="*120)
    print("PREPROCESSING COMPARISON ANALYSIS REPORT")
    print("="*120)
    
    # 对比统计表格
    print(f"\n{'Dimension':<12} {'Original Range':<15} {'Processed Range':<16} {'Range Reduction':<15} {'Orig STD':<10} {'Proc STD':<10} {'STD Change':<10}")
    print("-" * 118)
    
    for name in action_names:
        orig_s = original_stats[name]
        proc_s = processed_stats[name]
        
        orig_range = orig_s['range']
        proc_range = proc_s['range']
        range_reduction = orig_range / proc_range if proc_range > 0 else float('inf')
        
        orig_std = orig_s['std']
        proc_std = proc_s['std']
        std_change = orig_std / proc_std if proc_std > 0 else float('inf')
        
        print(f"{name:<12} {orig_range:<15.4f} {proc_range:<16.4f} {range_reduction:<15.2f} "
              f"{orig_std:<10.4f} {proc_std:<10.4f} {std_change:<10.2f}")
    
    # 重点分析euler角的改善
    print("\n" + "="*120)
    print("EULER ANGLES PREPROCESSING BENEFITS")
    print("="*120)
    
    euler_names = ['euler_x', 'euler_y', 'euler_z']
    for name in euler_names:
        orig_s = original_stats[name]
        proc_s = processed_stats[name]
        
        print(f"\n📐 {name.upper()}:")
        print(f"   Original range: [{orig_s['min']:.3f}, {orig_s['max']:.3f}] = {orig_s['range']:.3f} rad ({np.degrees(orig_s['range']):.1f}°)")
        print(f"   Processed range: [{proc_s['min']:.3f}, {proc_s['max']:.3f}] = {proc_s['range']:.3f} rad ({np.degrees(proc_s['range']):.1f}°)")
        
        if name in ['euler_x', 'euler_z']:
            range_improvement = orig_s['range'] / proc_s['range'] if proc_s['range'] > 0 else float('inf')
            std_improvement = orig_s['std'] / proc_s['std'] if proc_s['std'] > 0 else float('inf')
            
            print(f"   🚀 Range improvement: {range_improvement:.1f}x smaller")
            print(f"   📉 Standard deviation: {orig_s['std']:.3f} → {proc_s['std']:.3f} ({std_improvement:.1f}x improvement)")
            
            # 检查是否成功移除了不连续性
            if orig_s['range'] > 6.0 and proc_s['range'] < 4.0:
                print(f"   ✅ Successfully eliminated ±π discontinuity!")
            else:
                print(f"   ⚠️  Partial discontinuity reduction")
        else:
            print(f"   ℹ️  euler_y: No processing applied (already continuous)")
    
    # 分析跳跃减少效果
    print("\n" + "="*120)
    print("DISCONTINUITY REDUCTION ANALYSIS")
    print("="*120)
    
    # 计算序列中的跳跃
    for name, idx in [('euler_x', 3), ('euler_z', 5)]:
        orig_data = original_actions[:, idx]
        proc_data = processed_actions[:, idx]
        
        # 计算相邻步骤间的跳跃
        orig_jumps = np.abs(np.diff(orig_data))
        proc_jumps = np.abs(np.diff(proc_data))
        
        # 统计大跳跃（> π）
        orig_large_jumps = np.sum(orig_jumps > np.pi)
        proc_large_jumps = np.sum(proc_jumps > np.pi)
        
        print(f"\n📊 {name.upper()} Sequential Jumps:")
        print(f"   Original large jumps (> π): {orig_large_jumps} ({orig_large_jumps/len(orig_jumps)*100:.1f}%)")
        print(f"   Processed large jumps (> π): {proc_large_jumps} ({proc_large_jumps/len(proc_jumps)*100:.1f}%)")
        print(f"   Jump reduction: {orig_large_jumps - proc_large_jumps} fewer large jumps")
        
        avg_orig_jump = np.mean(orig_jumps)
        avg_proc_jump = np.mean(proc_jumps)
        print(f"   Average jump size: {avg_orig_jump:.3f} → {avg_proc_jump:.3f} ({avg_orig_jump/avg_proc_jump:.1f}x reduction)")
    
    # 总结
    print("\n" + "="*120)
    print("PREPROCESSING EFFECTIVENESS SUMMARY")
    print("="*120)
    
    euler_x_improvement = original_stats['euler_x']['range'] / processed_stats['euler_x']['range']
    euler_z_improvement = original_stats['euler_z']['range'] / processed_stats['euler_z']['range']
    
    print(f"✅ Successfully applied phase shift preprocessing!")
    print(f"📈 Euler_x range reduced by {euler_x_improvement:.1f}x")
    print(f"📈 Euler_z range reduced by {euler_z_improvement:.1f}x")
    print(f"🎯 Discontinuity points moved from ±π to 0")
    print(f"🚀 This should significantly improve training stability!")
    print(f"💡 Position, euler_y, and gripper dimensions remain unchanged")

if __name__ == "__main__":
    main()
