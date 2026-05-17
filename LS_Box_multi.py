import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def load_multiple_yaml_data(yaml_paths: List[str]) -> Tuple[Dict, List[Dict]]:
    """
    加载多个YAML文件数据
    返回: (训练集数据, 多个生成集数据列表)
    """
    all_data = []
    for path in yaml_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            all_data.append(data)
    
    # 假设所有YAML的训练集数据相同，取第一个文件的训练集数据
    train_data = all_data[0]
    
    # 提取多个生成集数据
    gen_data_list = []
    for data in all_data:
        gen_data = {
            'porosity': extract_porosity_data(data)[1],  # 生成集孔隙度统计
            'tpf': extract_tpf_data(data, norm=False)['gen'],
            'norm_tpf': extract_tpf_data(data, norm=True)['gen'],
            'lpf': extract_lpf_data(data)['gen']
        }
        gen_data_list.append(gen_data)
    
    return train_data, gen_data_list


def plot_multi_porosity_boxplot(
    train_stats: tuple, 
    gen_stats_list: List[tuple], 
    model_labels: List[str],
    save_path: str
):
    """
    多模型孔隙度箱式图
    train_stats: 训练集统计量 (min, 25%, 50%, 75%, max)
    gen_stats_list: 多个生成集的统计量列表
    model_labels: 各生成集的标签（模型名称）
    """
    print('\n# ------------------- 多模型孔隙度箱式图 ------------------- #')
    plt.figure(figsize=(10, 8))
    
    # 箱体位置（训练集在0，生成集依次排列）
    positions = [0] + list(range(1, len(gen_stats_list) + 1))
    
    # 所有箱体的统计数据（训练集 + 所有生成集）
    all_stats = [train_stats] + gen_stats_list
    
    # 标签（训练集 + 各模型名称）
    all_labels = ['Train'] + model_labels
    
    # 颜色设置（训练集蓝色，生成集不同颜色）
    colors = ['#1E88E5'] + plt.cm.tab10(np.linspace(0, 1, len(gen_stats_list))[:, :3].tolist()
    
    # 绘制箱式图
    boxplot = plt.boxplot(
        all_stats,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"color": "black", "linewidth": 1.5},
        capprops={"color": "black", "linewidth": 1.5}
    )
    
    # 设置箱体颜色
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    
    # 添加标签与标题
    plt.xticks(positions, all_labels, fontsize=12)
    plt.ylabel('Porosity ($\phi$)', fontsize=14)
    plt.title('Multi-Model Porosity Distribution Comparison', fontsize=16, pad=20)
    plt.grid(True, linestyle="--", alpha=0.7, axis='y')
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ 多模型孔隙度箱式图已保存到：{save_path}")
    plt.show()
                                        
                                        
def plot_multi_correlation_comparison(
    train_data: Dict, 
    gen_data_list: List[Dict], 
    model_labels: List[str],
    save_path: str,
    correlation_type: str,  # 'tpf', 'norm_tpf' 或 'lpf'
    x_label: str,
    y_label: str,
    title: str
):
    """
    多模型相关性函数对比图（TPF/归一化TPF/LPF）
    """
    print(f'\n# ------------------- {title} ------------------- #')
    plt.figure(figsize=(12, 8))
    
    # 绘制训练集曲线（均值±标准差）
    x_key = 'r' if correlation_type != 'lpf' else 'l'
    plt.plot(
        train_data[x_key],
        train_data['mean'],
        label="Train (Mean)",
        color="#1E88E5",
        linewidth=3
    )
    plt.fill_between(
        train_data[x_key],
        train_data['mean'] - train_data['std'],
        train_data['mean'] + train_data['std'],
        color="#BBDEFB",
        alpha=0.5,
        label="Train (Mean±Std)"
    )
    
    # 绘制多个生成集曲线（均值）
    colors = plt.cm.tab10(np.linspace(0, 1, len(gen_data_list)))
    for i, gen_data in enumerate(gen_data_list):
        plt.plot(
            gen_data[correlation_type][x_key],
            gen_data[correlation_type]['mean'],
            label=f"{model_labels[i]} (Mean)",
            color=colors[i],
            linewidth=2.5,
            linestyle='--'
        )
    
    # 设置标签与标题
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=16, pad=20)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ {title}对比图已保存到：{save_path}")
    plt.show()
                                        
                                    