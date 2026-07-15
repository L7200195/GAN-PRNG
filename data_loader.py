"""
GAN-PRNG 数据加载模块
负责生成训练数据和噪声
"""

import torch
import numpy as np


def generate_uniform_1to1(batch_size, seq_len, device='cpu', low=-1.0, high=1.0):
    """
    生成真实训练数据（浮点型）

    Args:
        batch_size: 批大小
        seq_len: 序列长度
        device: 设备
        low: 最小值 (默认 -1.0)
        high: 最大值 (默认 1.0)

    Returns:
        真实数据张量，范围 [low, high]
    """
    return torch.rand(batch_size, seq_len, device=device) * (high - low) + low

def generate_uniform_int8(batch_size, seq_len, device='cpu', low=0, high=256):
    """
    生成均匀分布整数数据 (int8 range)

    Args:
        batch_size: 批大小
        seq_len: 序列长度
        device: 设备
        low: 最小值 (默认 0)
        high: 最大值 (默认 256)

    Returns:
        均匀分布整数张量，范围 [low, high-1]
    """
    return torch.rand(batch_size, seq_len, device=device) * (high - low) + low

def generate_gauss_miu0_sigma1(batch_size, z_dim, device='cpu', mean=0.0, std=1.0):
    """
    生成噪声输入

    Args:
        batch_size: 批大小
        z_dim: 噪声维度
        device: 设备
        mean: 均值 (默认 0.0)
        std: 标准差 (默认 1.0)

    Returns:
        噪声张量，正态分布 N(mean, std^2)
    """
    return torch.randn(batch_size, z_dim, device=device) * std + mean



def prepare_batch(batch_size, seq_len, z_dim, device='cpu',
                  data_type='bin', noise_type='gaussian', **kwargs):
    """
    准备一个训练批次的数据

    Args:
        batch_size: 批大小
        seq_len: 序列长度
        z_dim: 噪声维度
        device: 设备
        data_type: 数据类型 ('float', 'int', 'bytes')
        noise_type: 噪声类型 ('gaussian', 'uniform')
        **kwargs: 其他参数传递给数据生成函数

    Returns:
        real_data, noise, real_labels, fake_labels
    """
    # 生成真实数据
    if data_type == 'bin':
        real_data = generate_uniform_1to1(batch_size, seq_len, device, **kwargs)
    elif data_type == 'int8':
        real_data = generate_uniform_int8(batch_size, seq_len, device, **kwargs)
    else:
        raise ValueError(f"不支持的数据类型：{data_type}")

    # 生成噪声
    if noise_type == 'gaussian':
        noise = generate_gauss_miu0_sigma1(batch_size, z_dim, device, **{k: v for k, v in kwargs.items()
                                                              if k in ['mean', 'std']})
    elif noise_type == 'uniform':
        '''
        noise = generate_noise_uniform(batch_size, z_dim, device, **{k: v for k, v in kwargs.items()
                                                                      if k in ['low', 'high']})'''
        raise ValueError(f"不合适的噪声类型：{noise_type}")
    else:
        raise ValueError(f"不支持的噪声类型：{noise_type}")

    real_labels = torch.ones(batch_size, 1, device=device)
    fake_labels = torch.zeros(batch_size, 1, device=device)

    return real_data, noise, real_labels, fake_labels

#============

class DataGenerator:
    """数据生成器类"""
    
    def __init__(self, batch_size, seq_len, z_dim, device='cpu', 
                 data_type='bin', noise_type='gaussian', **kwargs):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.z_dim = z_dim
        self.device = device
        self.data_type = data_type
        self.noise_type = noise_type
        self.kwargs = kwargs
    
    def generate_real_data(self, **override_kwargs):
        """
        生成真实数据
        
        Args:
            **override_kwargs: 覆盖默认参数
            
        Returns:
            真实数据张量
        """
        params = {**self.kwargs, **override_kwargs}

        if self.data_type == 'bin':
            return generate_uniform_1to1(self.batch_size, self.seq_len, self.device, **params)
        elif self.data_type == 'int8':
            return generate_uniform_int8(self.batch_size, self.seq_len, self.device, **params)
        else:
            raise ValueError(f"不支持的数据类型：{self.data_type}")

    
    def generate_noise(self, **override_kwargs):
        """
        生成噪声
        
        Args:
            **override_kwargs: 覆盖默认参数
            
        Returns:
            噪声张量
        """
        params = {**self.kwargs, **override_kwargs}

        if self.noise_type == 'gaussian':
            return generate_gauss_miu0_sigma1(self.batch_size, self.z_dim, self.device, **params)
        elif self.noise_type == 'uniform':
            '''
            noise = generate_noise_uniform(batch_size, z_dim, device, **{k: v for k, v in kwargs.items()
                                                                          if k in ['low', 'high']})'''
            raise ValueError(f"不合适的噪声类型：{self.noise_type}")
        else:
            raise ValueError(f"不支持的噪声类型：{self.noise_type}")


    def prepare_batch(self, **override_kwargs):
        """
        准备训练批次
        
        Args:
            **override_kwargs: 覆盖默认参数
            
        Returns:
            real_data, noise, real_labels, fake_labels
        """
        params = {**self.kwargs, **override_kwargs}
        return prepare_batch(self.batch_size, self.seq_len, self.z_dim, self.device,
                            self.data_type, self.noise_type, **params)


def plot_data_histogram(data, title="数据分布直方图", bins=20, bin_width=0.02, save_path=None):
    """
    绘制数据统计直方图（按区间统计占比）
    
    Args:
        data: 数据（torch.Tensor 或 numpy.ndarray）
        title: 图表标题
        bins: 分组数量或自定义的边界列表
        bin_width: 默认区间宽度（当 bins 为整数时使用）
        save_path: 保存路径（如果为 None 则显示图表）
    
    Returns:
        bin_stats: 包含每个区间的统计信息字典
    """
    import matplotlib.pyplot as plt
    
    # 转换为 numpy 数组
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    
    data = data.flatten()
    
    # 计算数据范围
    data_min, data_max = data.min(), data.max()
    
    # 如果是整数数据，使用整数边界
    if np.issubdtype(data.dtype, np.integer):
        bins = np.arange(data_min, data_max + 2) - 0.5
        bin_width = 1
    else:
        # 浮点数数据，使用指定宽度
        if isinstance(bins, int):
            bins = np.arange(data_min, data_max + bin_width, bin_width)
    
    # 计算直方图
    counts, bin_edges = np.histogram(data, bins=bins)
    
    # 计算占比
    total = len(data)
    percentages = counts / total * 100
    
    # 创建区间标签
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        if np.issubdtype(data.dtype, np.integer):
            label = f"{int(bin_edges[i] + 0.5)}"
        else:
            label = f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
        bin_labels.append(label)
    
    # 创建统计字典
    bin_stats = {
        'bin_labels': bin_labels,
        'counts': counts,
        'percentages': percentages,
        'total': total,
        'mean': float(data.mean()),
        'std': float(data.std()),
        'min': float(data_min),
        'max': float(data_max)
    }
    
    # 绘制图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左侧：柱状图
    ax1.bar(bin_labels, percentages, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('数值区间', fontsize=12)
    ax1.set_ylabel('占比 (%)', fontsize=12)
    ax1.set_title(f'{title}\n(总样本数：{total:,})', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (label, pct) in enumerate(zip(bin_labels, percentages)):
        if pct > 0:
            ax1.text(i, pct, f'{pct:.2f}%', ha='center', va='bottom', fontsize=8, rotation=45)
    
    # 右侧：累积分布
    cumulative_pct = np.cumsum(percentages)
    ax2.plot(bin_labels, cumulative_pct, 'o-', linewidth=2, markersize=4, color='red')
    ax2.set_xlabel('数值区间', fontsize=12)
    ax2.set_ylabel('累积占比 (%)', fontsize=12)
    ax2.set_title('累积分布图', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax2.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90%')
    ax2.legend()
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"直方图已保存至：{save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return bin_stats


def print_statistics(bin_stats, data_name="数据"):
    """
    打印数据统计信息
    
    Args:
        bin_stats: 统计字典（来自 plot_data_histogram）
        data_name: 数据名称
    """
    print(f"\n{'='*60}")
    print(f"{data_name} 统计信息")
    print(f"{'='*60}")
    print(f"总样本数：{bin_stats['total']:,}")
    print(f"最小值：{bin_stats['min']:.4f}")
    print(f"最大值：{bin_stats['max']:.4f}")
    print(f"均值：{bin_stats['mean']:.4f}")
    print(f"标准差：{bin_stats['std']:.4f}")
    print(f"\n区间分布:")
    print(f"{'区间':<20} {'数量':>10} {'占比':>10}")
    print(f"{'-'*60}")
    
    for label, count, pct in zip(bin_stats['bin_labels'], 
                                  bin_stats['counts'], 
                                  bin_stats['percentages']):
        print(f"{label:<20} {count:>10,} {pct:>9.4f}%")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("--- 数据生成模块测试 ---\n")
    
    device = 'cpu'
    batch_size = 100  # 增加样本数以获得更好的统计
    seq_len = 1000
    z_dim = 32
    
    # ==================== 测试 1: 浮点型数据 ====================
    print("=" * 60)
    print("测试 1: 浮点型数据 (范围 [-1, 1])")
    print("=" * 60)
    
    real_data_float = generate_uniform_1to1(batch_size, seq_len, device, low=-1.0, high=1.0)
    print(f"\n数据形状：{real_data_float.shape}")
    print(f"范围：[{real_data_float.min():.4f}, {real_data_float.max():.4f}]")
    print(f"均值：{real_data_float.mean():.4f}, 标准差：{real_data_float.std():.4f}")
    
    # 绘制直方图
    stats_float = plot_data_histogram(
        real_data_float, 
        title="浮点型数据分布 (范围 [-1, 1])",
        bin_width=0.1
    )
    print_statistics(stats_float, "浮点型数据")
    
    # ==================== 测试 2: 整数型数据 ====================
    print("\n" + "=" * 60)
    print("测试 2: 整数型数据 (范围 [0, 256))")
    print("=" * 60)
    
    real_data_int = generate_uniform_int8(batch_size, seq_len, device, low=0, high=256)
    print(f"\n数据形状：{real_data_int.shape}")
    print(f"范围：[{real_data_int.min()}, {real_data_int.max()}]")
    print(f"均值：{real_data_int.float().mean():.4f}, 标准差：{real_data_int.float().std():.4f}")
    
    # 绘制直方图
    stats_int = plot_data_histogram(
        real_data_int, 
        title="整数型数据分布 (范围 [0, 256))",
        bin_width=2.5
    )
    print_statistics(stats_int, "整数型数据")
    
    # ==================== 测试 3: 高斯噪声 ====================
    print("\n" + "=" * 60)
    print("测试 3: 高斯噪声 (均值=0, 标准差=1)")
    print("=" * 60)
    
    noise_gaussian = generate_gauss_miu0_sigma1(batch_size, z_dim, device, mean=0.0, std=1.0)
    print(f"\n数据形状：{noise_gaussian.shape}")
    print(f"范围：[{noise_gaussian.min():.4f}, {noise_gaussian.max():.4f}]")
    print(f"均值：{noise_gaussian.mean():.4f}, 标准差：{noise_gaussian.std():.4f}")
    
    # 绘制直方图
    stats_noise = plot_data_histogram(
        noise_gaussian, 
        title="高斯噪声分布 (均值=0, 标准差=1)",
        bin_width=0.2
    )
    print_statistics(stats_noise, "高斯噪声")
    
    # ==================== 测试 4: 批次数据准备 ====================
    print("\n" + "=" * 60)
    print("测试 4: 批次数据准备 (不同类型)")
    print("=" * 60)
    
    # 浮点型 + 高斯噪声
    real_data, noise, real_labels, fake_labels = prepare_batch(
        batch_size=100, seq_len=256, z_dim=32, device=device,
        data_type='bin', noise_type='gaussian'
    )
    print(f"\n浮点型 + 高斯噪声:")
    print(f"  真实数据：{real_data.shape}, 范围 [{real_data.min():.2f}, {real_data.max():.2f}]")
    print(f"  噪声：{noise.shape}, 均值 {noise.mean():.2f}")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
