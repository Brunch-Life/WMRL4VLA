"""
analyze_action_distribution.py

用于分析NPZ数据集中7维action的分布情况，并生成可视化图表
包括：
1. 每维action的分布直方图
2. 统计信息（均值、标准差、最大最小值等）
3. 相关性分析
4. 时间序列分析（可选）
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import json
from scipy import stats
from scipy.spatial.transform import Rotation as R
import pandas as pd
from tqdm import tqdm
import argparse

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_single_npz_file(file_path: str):
    """
    加载单个NPZ文件并解析action数据
    Args:
        file_path: NPZ文件路径
    Returns:
        actions: (T, 7) numpy数组，包含position(3) + euler(3) + gripper(1)
    """
    try:
        data = np.load(file_path, allow_pickle=True)["arr_0"].tolist()
        
        # 解析action数据
        position = data["action"]["end"]["position"].squeeze(1)  # (T, 3)
        orientation_quat = data["action"]["end"]["orientation"].squeeze(1)  # (T, 4) quaternion
        gripper = data["action"]["effector"]["position_gripper"]  # (T, 1)
        
        # 将四元数转换为欧拉角
        euler_angles = R.from_quat(orientation_quat).as_euler('xyz', degrees=False)  # (T, 3)
        
        # 拼接：position (3) + euler (3) + gripper (1) = 7D action
        actions = np.concatenate([
            position,      # (T, 3)
            euler_angles,  # (T, 3) 
            gripper        # (T, 1)
        ], axis=1).astype(np.float32)  # (T, 7)
        
        return actions
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_dataset_actions(data_dir: Path, max_files: int = None):
    """
    加载整个数据集的所有action数据
    Args:
        data_dir: 数据集目录
        max_files: 最大加载文件数（用于快速测试）
    Returns:
        all_actions: (N, 7) numpy数组
        file_info: 每个文件的信息列表
    """
    npz_files = sorted(glob.glob(str(data_dir / "*.npz")))
    if max_files:
        npz_files = npz_files[:max_files]
    
    print(f"Found {len(npz_files)} NPZ files in {data_dir}")
    
    all_actions = []
    file_info = []
    
    for file_path in tqdm(npz_files, desc="Loading files"):
        actions = load_single_npz_file(file_path)
        if actions is not None:
            all_actions.append(actions)
            file_info.append({
                'file': Path(file_path).name,
                'num_steps': len(actions),
                'action_shape': actions.shape
            })
    
    if all_actions:
        all_actions = np.concatenate(all_actions, axis=0)
        print(f"Total actions loaded: {all_actions.shape}")
        return all_actions, file_info
    else:
        raise ValueError("No valid actions loaded!")


def compute_statistics(actions: np.ndarray):
    """
    计算action各维度的统计信息
    """
    action_names = ['pos_x', 'pos_y', 'pos_z', 'euler_x', 'euler_y', 'euler_z', 'gripper']
    
    stats_dict = {}
    for i, name in enumerate(action_names):
        data = actions[:, i]
        stats_dict[name] = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75)),
            'range': float(np.max(data) - np.min(data)),
            'skewness': float(stats.skew(data)),
            'kurtosis': float(stats.kurtosis(data))
        }
    
    return stats_dict


def plot_action_distributions(actions: np.ndarray, save_dir: Path):
    """
    绘制action各维度的分布直方图
    """
    action_names = ['pos_x', 'pos_y', 'pos_z', 'euler_x', 'euler_y', 'euler_z', 'gripper']
    
    # 创建子图
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes[:7], action_names)):
        data = actions[:, i]
        
        # 绘制直方图和密度曲线
        ax.hist(data, bins=50, density=True, alpha=0.7, color=f'C{i}', edgecolor='black')
        
        # 添加统计信息
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'±1σ')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
        
        ax.set_title(f'{name} Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{name} Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 删除多余的子图
    for i in range(7, 9):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'action_distributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'action_distributions.pdf', bbox_inches='tight')
    print(f"Saved action distributions to {save_dir}")


def plot_action_boxplots(actions: np.ndarray, save_dir: Path):
    """
    绘制action各维度的箱线图
    """
    action_names = ['pos_x', 'pos_y', 'pos_z', 'euler_x', 'euler_y', 'euler_z', 'gripper']
    
    # 创建DataFrame用于seaborn绘图
    df_data = []
    for i, name in enumerate(action_names):
        df_data.extend([{'dimension': name, 'value': val} for val in actions[:, i]])
    
    df = pd.DataFrame(df_data)
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='dimension', y='value')
    plt.title('Action Dimensions Box Plots', fontsize=14, fontweight='bold')
    plt.xlabel('Action Dimension')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'action_boxplots.png', dpi=300, bbox_inches='tight')
    print(f"Saved action boxplots to {save_dir}")


def plot_correlation_matrix(actions: np.ndarray, save_dir: Path):
    """
    绘制action各维度之间的相关性矩阵
    """
    action_names = ['pos_x', 'pos_y', 'pos_z', 'euler_x', 'euler_y', 'euler_z', 'gripper']
    
    # 计算相关性矩阵
    corr_matrix = np.corrcoef(actions.T)
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角
    
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt='.3f',
                mask=mask,
                xticklabels=action_names,
                yticklabels=action_names,
                cmap='coolwarm',
                center=0,
                square=True,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('Action Dimensions Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'action_correlation.png', dpi=300, bbox_inches='tight')
    print(f"Saved correlation matrix to {save_dir}")


def plot_action_trajectories(actions: np.ndarray, save_dir: Path, max_episodes: int = 5):
    """
    绘制action轨迹示例（假设每个episode有相同长度）
    """
    action_names = ['pos_x', 'pos_y', 'pos_z', 'euler_x', 'euler_y', 'euler_z', 'gripper']
    
    # 估计episode长度（简单假设前几百步为一个episode）
    episode_len = min(200, len(actions) // max_episodes)
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes[:7], action_names)):
        for ep in range(max_episodes):
            start_idx = ep * episode_len
            end_idx = start_idx + episode_len
            if end_idx <= len(actions):
                episode_data = actions[start_idx:end_idx, i]
                ax.plot(episode_data, alpha=0.7, label=f'Episode {ep+1}')
        
        ax.set_title(f'{name} Trajectories', fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel(f'{name} Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 删除最后一个多余的子图
    fig.delaxes(axes[7])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'action_trajectories.png', dpi=300, bbox_inches='tight')
    print(f"Saved action trajectories to {save_dir}")


def save_statistics_to_json(stats_dict: dict, file_info: list, save_dir: Path):
    """
    保存统计信息到JSON文件
    """
    output_data = {
        'dataset_info': {
            'total_files': len(file_info),
            'total_steps': sum(info['num_steps'] for info in file_info),
            'files': file_info
        },
        'action_statistics': stats_dict
    }
    
    with open(save_dir / 'action_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved analysis results to {save_dir / 'action_analysis.json'}")


def print_summary_report(stats_dict: dict):
    """
    打印统计信息摘要报告
    """
    print("\n" + "="*80)
    print("ACTION DISTRIBUTION ANALYSIS REPORT")
    print("="*80)
    
    action_names = ['pos_x', 'pos_y', 'pos_z', 'euler_x', 'euler_y', 'euler_z', 'gripper']
    
    print(f"{'Dimension':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Range':<10}")
    print("-" * 70)
    
    for name in action_names:
        s = stats_dict[name]
        print(f"{name:<12} {s['mean']:<10.4f} {s['std']:<10.4f} {s['min']:<10.4f} {s['max']:<10.4f} {s['range']:<10.4f}")
    
    print("\n" + "="*80)
    print("KEY OBSERVATIONS:")
    print("="*80)
    
    # 分析位置维度
    pos_ranges = [stats_dict[f'pos_{axis}']['range'] for axis in ['x', 'y', 'z']]
    print(f"1. Position ranges: X={pos_ranges[0]:.3f}, Y={pos_ranges[1]:.3f}, Z={pos_ranges[2]:.3f}")
    
    # 分析欧拉角
    euler_ranges = [stats_dict[f'euler_{axis}']['range'] for axis in ['x', 'y', 'z']]
    print(f"2. Euler angle ranges: X={euler_ranges[0]:.3f}, Y={euler_ranges[1]:.3f}, Z={euler_ranges[2]:.3f}")
    
    # 分析夹爪
    gripper_stats = stats_dict['gripper']
    print(f"3. Gripper range: [{gripper_stats['min']:.3f}, {gripper_stats['max']:.3f}]")
    
    # 检查数据质量
    print("\n4. Data Quality Checks:")
    for name in action_names:
        s = stats_dict[name]
        if abs(s['skewness']) > 2:
            print(f"   - {name}: High skewness ({s['skewness']:.3f}) - data may be unbalanced")
        if abs(s['kurtosis']) > 3:
            print(f"   - {name}: High kurtosis ({s['kurtosis']:.3f}) - data has heavy tails")


def main():
    parser = argparse.ArgumentParser(description='Analyze action distribution in NPZ dataset')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/chenyinuo/data/dataset/bingwen/data_for_success/green_bell_pepper_plate_wooden/success',
                       help='Directory containing NPZ files')
    parser.add_argument('--output_dir', type=str, default='action_analysis_results',
                       help='Output directory for results')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # 加载数据
    actions, file_info = load_dataset_actions(data_dir, args.max_files)
    
    # 计算统计信息
    print("\nComputing statistics...")
    stats_dict = compute_statistics(actions)
    
    # 生成可视化
    print("\nGenerating visualizations...")
    plot_action_distributions(actions, output_dir)
    plot_action_boxplots(actions, output_dir)
    plot_correlation_matrix(actions, output_dir)
    plot_action_trajectories(actions, output_dir)
    
    # 保存结果
    save_statistics_to_json(stats_dict, file_info, output_dir)
    
    # 打印报告
    print_summary_report(stats_dict)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
