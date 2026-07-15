"""
GAN-PRNG 训练模块
包含训练循环、优化器设置和训练逻辑
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter

from data_loader import prepare_batch
from evaluator import AnalyzerRuns


# ==================== 约束性损失函数 ====================

# 生成器损失函数选项
LOSS_DEFAULT = 0           # 仅对抗损失
LOSS_SERIAL = 1            # + 串行均匀度损失 (serial_loss)
LOSS_DFT = 2               # + DFT 频谱损失 (dft_loss)
LOSS_RANDOM_EXCURSION = 3  # + 随机游走损失 (random_excursion_loss)
LOSS_LINEAR_COMPLEXITY = 4 # + 线性复杂度损失 (linear_complexity_loss)
LOSS_MEAN_CONSTRAINT = 5   # + 均值约束 (target=128)


def linear_complexity_loss(fake_data_g, L=16):
    x = fake_data_g.float()
    n = x.shape[0]
    x_centered = x - x.mean()
    x_pad = torch.nn.functional.pad(x_centered, (0, n - 1))
    X = torch.fft.rfft(x_pad)
    power = X.abs() ** 2
    autocorr_full = torch.fft.irfft(power)[:n]
    autocorr_norm = autocorr_full / (autocorr_full[0] + 1e-8)
    ac = autocorr_norm[1:L + 1]
    return torch.sum(torch.abs(ac))

# ① DFT 频谱
def dft_loss(fake_data_g, significance=0.05):
    """
    DFT 频谱损失：约束频谱无异常尖峰
    """
    x = fake_data_g.float()
    n = x.shape[0]
    # 去均值后 FFT
    S = torch.abs(torch.fft.fft(x - 128))
    # 只取前半部分正频率，忽略直流分量
    S_half = S[1:n // 2]  # shape: (n//2 - 1,)
    # 理论阈值
    threshold = torch.sqrt(
        torch.tensor(n * torch.log(torch.tensor(1.0 / significance)),
                     device=x.device, dtype=torch.float32)
    )
    # 方法：所有频率成分的平方和超出阈值的部分
    # 这样绝对返回标量，没有 dim 歧义
    excess = torch.relu(S_half - threshold)
    loss = torch.mean(excess ** 2)  # 或者 torch.sum，确保是标量
    return loss

# ③ 串行 (m=2 示例, 16 位直方图)
def serial_loss(fake_data_g, m=2):
    # 确保是1D
    x = fake_data_g.flatten().float()
    n = x.shape[0]
    n_bits = 4
    base = 2 ** n_bits
    x_scaled = ((x / 255.0) * (base - 1) + 0.5).long().clamp(0, base - 1)
    # 滑动窗口
    windows = x_scaled.unfold(0, m, 1)  # (num_windows, m)
    # 计算每个窗口的索引：∑ window[i] * base^(m-1-i)
    weights = base ** torch.arange(m-1, -1, -1, device=x.device)
    idx = (windows * weights).sum(dim=1)
    bins = base ** m
    counts = torch.bincount(idx, minlength=bins).float()
    p_soft = (counts + 1e-8) / counts.sum()
    uniform = 1.0 / bins
    return torch.sum((p_soft - uniform) ** 2 / uniform)

# ⑦ 随机游走
def random_excursion_loss(fake_data_g, levels=None):
    """
    随机游走损失：标准化累加和在各水平的访问次数应匹配理论值

    Args:
        fake_data_g: 生成器输出，形状 (n,)，int8 序列
        levels: 状态水平列表，默认 [-4,-3,-2,-1,1,2,3,4]（跳过 0）

    Returns:
        标量损失值
    """
    x = fake_data_g.float()
    # 标准化累加和: S_k = sum_{i=1}^k (x_i - 128) / std(x)
    centered = x - 128
    std = torch.std(centered)
    S = torch.cumsum(centered, dim=0) / (std + 1e-8)  # 避免除零
    # 默认水平（NIST 常用 ±1 到 ±4）
    if levels is None:
        levels = [-4, -3, -2, -1, 1, 2, 3, 4]
    # NIST 理论访问概率（简化，实际依赖序列长度）
    # 这里用标准正态稳态分布的近似值
    expected_visits = {
        -4: 0.003, -3: 0.009, -2: 0.027, -1: 0.065,
        1: 0.065, 2: 0.027, 3: 0.009, 4: 0.003
    }
    loss = 0.0
    for h in levels:
        # 软计数：S 落在 [h, h+1) 的累计"概率"
        visit = torch.sum(
            torch.sigmoid((S - h) * 10) * torch.sigmoid(((h + 1) - S) * 10)
        )
        # 归一化到概率
        visit_prob = visit / len(S)
        loss += (visit_prob - expected_visits[h]) ** 2

    return loss




class Trainer:
    """训练器类"""
    
    def __init__(self, generator, discriminator, device, lr=0.0002, betas=(0.5, 0.999),
                 z_dim=32, loss_type=0):
        """
        初始化训练器

        Args:
            generator: 生成器模型
            discriminator: 判别器模型
            device: 设备
            lr: 学习率
            betas: Adam 优化器参数
            z_dim: 噪声维度 (默认 32)
            loss_type: 生成器损失函数选项
                0 - LOSS_DEFAULT: 仅对抗损失
                1 - LOSS_SERIAL: + serial_loss (串行均匀度)
                2 - LOSS_DFT: + dft_loss (DFT频谱)
                3 - LOSS_RANDOM_EXCURSION: + random_excursion_loss (随机游走)
                4 - LOSS_LINEAR_COMPLEXITY: + linear_complexity_loss (线性复杂度)
                5 - LOSS_MEAN_CONSTRAINT: + 均值约束 (target=128)
        """
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.z_dim = z_dim
        self.loss_type = loss_type

        # 初始化优化器和损失函数
        self.optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=betas)
        self.optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
        self.criterion = nn.BCELoss()

        # 训练历史记录
        self.d_loss_history = []
        self.g_loss_history = []
        self.writer = None
    
    def setup_tensorboard(self, log_dir):
        """设置 TensorBoard 记录器"""
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard 日志目录：{log_dir}")
    
    def train_step(self, real_data, noise, real_labels, fake_labels, CNN_disc=0):
        """
        单步训练（包括 D 和 G 的训练）

        可用的附加损失函数 (通过 self.loss_type 选择):
            LOSS_DEFAULT (0): 仅对抗损失
            LOSS_SERIAL (1): + serial_loss(fake_data_g, m=2)
            LOSS_DFT (2): + dft_loss(fake_data_g, significance=0.05)
            LOSS_RANDOM_EXCURSION (3): + random_excursion_loss(fake_data_g)
            LOSS_LINEAR_COMPLEXITY (4): + linear_complexity_loss(fake_data_g, L=16)
            LOSS_MEAN_CONSTRAINT (5): + (torch.mean(fake_data_g) - 128) ** 2

        Args:
            real_data: 真实数据
            noise: 噪声
            real_labels: 真实标签
            fake_labels: 假标签
            CNN_disc: 是否使用 CNN 判别器 (0/1)

        Returns:
            d_loss, g_loss
        """
        batch_size = real_data.size(0)
        # 如果使用 CNN 判别器，需要调整数据维度
        if CNN_disc == 1:
            real_data = real_data.unsqueeze(1)  # [batch_size, 1, seq_len]

        # ==================== 训练判别器 D ====================
        self.optimizer_d.zero_grad()

        # 真实数据的损失
        out_real = self.discriminator(real_data)
        loss_d_real = self.criterion(out_real, real_labels)

        # 生成假数据并计算损失
        fake_data = self.generator(noise)
        # 如果使用 CNN 判别器，需要调整数据维度
        if CNN_disc == 1:
            fake_data = fake_data.unsqueeze(1)  # [batch_size, 1, seq_len]
        out_fake = self.discriminator(fake_data.detach())  # 切断梯度传回 G 的路径
        loss_d_fake = self.criterion(out_fake, fake_labels)

        # 总判别器损失
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        self.optimizer_d.step()

        # ==================== 训练生成器 G ====================
        self.optimizer_g.zero_grad()

        # 重新生成噪声用于训练 G
        noise_g = torch.randn(batch_size, self.z_dim, device=self.device)
        fake_data_g = self.generator(noise_g)
        # 如果使用 CNN 判别器，需要调整数据维度
        if CNN_disc == 1:
            fake_data_g = fake_data_g.unsqueeze(1)  # [batch_size, 1, seq_len]

        # 对抗损失
        loss_g = self.criterion(self.discriminator(fake_data_g), real_labels)

        # 根据 loss_type 添加约束损失
        if self.loss_type == 1:  # LOSS_SERIAL
            loss_g = loss_g + serial_loss(fake_data_g, m=2)
        elif self.loss_type == 2:  # LOSS_DFT
            loss_g = loss_g + dft_loss(fake_data_g, significance=0.05)
        elif self.loss_type == 3:  # LOSS_RANDOM_EXCURSION
            loss_g = loss_g + random_excursion_loss(fake_data_g)
        elif self.loss_type == 4:  # LOSS_LINEAR_COMPLEXITY
            loss_g = loss_g + linear_complexity_loss(fake_data_g, L=16)
        elif self.loss_type == 5:  # LOSS_MEAN_CONSTRAINT
            loss_g = loss_g + (torch.mean(fake_data_g) - 128) ** 2
        # loss_type == 0 (LOSS_DEFAULT): 仅对抗损失，不添加额外约束

        loss_g.backward()
        self.optimizer_g.step()

        return loss_d.item(), loss_g.item()
    
    def train_epoch(self, epoch, batch_size=128, seq_len=256, z_dim=32, CNN_disc=0, data_type='bin'):
        """
        单个训练轮次
        
        Args:
            epoch: 当前轮次
            batch_size: 批大小
            seq_len: 序列长度
            z_dim: 噪声维度
            CNN_disc: 模型类型
        
        Returns:
            d_loss, g_loss
        """
        # 同步噪声维度（模型可能用不同 z_dim 创建）
        self.z_dim = z_dim

        # 准备批次数据
        real_data, noise, real_labels, fake_labels = prepare_batch(
            batch_size, seq_len, z_dim, self.device, data_type
        )
        # 执行训练步骤
        d_loss, g_loss = self.train_step(real_data, noise, real_labels, fake_labels, CNN_disc)
        
        # 记录损失历史
        self.d_loss_history.append(d_loss)
        self.g_loss_history.append(g_loss)
        
        return d_loss, g_loss
    
    def evaluate_and_log(self, epoch, fake_data, bit_mode=0):
        """
        评估模型性能并记录日志
        
        Args:
            epoch: 当前轮次
            fake_data: 当前生成的假数据
            bit_mode: 比特模式
        
        Returns:
            metrics: 随机性指标
        """
        # 使用评估器获取随机性指标
        metrics = AnalyzerRuns.get_metrics(fake_data, bit_mode)
        first_10_vals = fake_data[0, :10].detach().cpu().numpy()  # 转为 numpy 便于格式化
        print(f"轮次 [{epoch}] | " 
              f"Data前10: {', '.join([f'{v:.4f}' if isinstance(v, float) else str(v) for v in first_10_vals])}\n"
              f"D 损失：{self.d_loss_history[-1]:.4f} | "
              f"G 损失：{self.g_loss_history[-1]:.4f} | "
              f"1 占比：{metrics['1 占比']:.4f} | "
              f"自相关性：{metrics['自相关性']:.4f} | "
              f"Runs_P 值：{metrics['Runs_P 值']:.6f}")
        
        # 记录到 TensorBoard
        if self.writer:
            self.writer.add_scalar("Loss/Discriminator", self.d_loss_history[-1], epoch)
            self.writer.add_scalar("Loss/Generator", self.g_loss_history[-1], epoch)
            self.writer.add_scalar("Metrics/1 占比", metrics['1 占比'], epoch)
            self.writer.add_scalar("Metrics/自相关性", metrics['自相关性'], epoch)
            self.writer.add_scalar("Metrics/Runs_P 值", metrics['Runs_P 值'], epoch)
        
        return metrics
    
    def save_checkpoint(self, epoch, save_dir):
        """
        保存检查点
        
        Args:
            epoch: 当前轮次
            save_dir: 保存目录
        
        Returns:
            ckpt_path: 检查点路径
        """
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, f"生成器_轮次_{epoch}.pth")
        torch.save(self.generator.state_dict(), ckpt_path)
        print(f"已保存检查点：{ckpt_path}")
        return ckpt_path
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint)
        print(f"已加载检查点：{checkpoint_path}")
    
    def close(self):
        """关闭 TensorBoard 记录器"""
        if self.writer:
            self.writer.close()
            print("TensorBoard 记录器已关闭。")


if __name__ == "__main__":
    from models import create_model
    
    print("--- 训练模块测试 ---")
    
    # 创建模型
    device = 'cpu'
    gen, disc = create_model(model_type=1, device=device)
    
    # 创建训练器
    trainer = Trainer(gen, disc, device)
    
    print(f"生成器参数量：{sum(p.numel() for p in gen.parameters()):,}")
    print(f"判别器参数量：{sum(p.numel() for p in disc.parameters()):,}")
    
    # 测试单个训练步骤
    real_data, noise, real_labels, fake_labels = prepare_batch(4, 256, 32, device,
                  data_type='int8', noise_type='gaussian')
    print(real_data.size(), noise.size())
    d_loss, g_loss = trainer.train_step(real_data, noise, real_labels, fake_labels, CNN_disc=1)
    
    print(f"\n测试训练步骤:")
    print(f"  D 损失：{d_loss:.4f}")
    print(f"  G 损失：{g_loss:.4f}")
    
    print("\n测试通过！")

    print("--- 增加测试 ---")

    from data_loader import generate_uniform_int8



    fake_data = generate_uniform_int8(128, 256, 'cpu')
    metrics = trainer.evaluate_and_log(4, fake_data, bit_mode=1)
