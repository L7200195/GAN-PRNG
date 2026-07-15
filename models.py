"""
GAN-PRNG 模型定义模块
包括生成器和判别器的多种架构实现
"""

import torch
import torch.nn as nn


class HardMod(nn.Module):
    """前向取模 [0, max_num)，反向梯度直传"""
    def __init__(self, max_num=256.0):
        super().__init__()
        self.register_buffer('max_num', torch.tensor(max_num))

    def forward(self, x):
        class _STE(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, m):
                return input - torch.floor(input / m) * m

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, None

        return _STE.apply(x, self.max_num)


# ----------          model 0      -------------
class Generator(nn.Module):
    """基础生成器模型"""
    def __init__(self, z_dim=32, seq_len=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, seq_len),
            HardMod(256.0)
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    """基础判别器模型"""
    def __init__(self, seq_len=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ----------          model 1      -------------
class GA_Gen_dual(nn.Module):
    """双生成器架构"""
    def __init__(self, z_dim=32, seq_len=256):
        super(GA_Gen_dual, self).__init__()
        self.gen1 = nn.Sequential(
            nn.Linear(z_dim, 200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 400),
            nn.LeakyReLU(0.2),
            nn.Linear(400, z_dim),
            HardMod(256.0),
        )
        self.gen2 = nn.Sequential(
            nn.Linear(z_dim, 200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 400),
            nn.LeakyReLU(0.2),
            nn.Linear(400, 800),
            nn.LeakyReLU(0.2),
            nn.Linear(800, seq_len),
            HardMod(256.0),
        )

    def forward(self, x):
        x1 = self.gen1(x)
        x = self.gen2(x1)
        return x


class GA_Disc(nn.Module):
    """CNN 判别器架构"""
    def __init__(self, z_dim=32, seq_len=256):
        super(GA_Disc, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv1d(1, 4, 2),
            nn.LeakyReLU(),
            nn.Conv1d(4, 4, 2),
            nn.LeakyReLU(),
            nn.Conv1d(4, 4, 2),
            nn.LeakyReLU(),
            nn.Conv1d(4, 4, 2),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 1),
            nn.Flatten(),
            nn.Linear(4 * (seq_len-5), 4),   #
            nn.LeakyReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dis(x)

# ----------          model 2      -------------
# 假设 HardMod 是你外部定义的自定义层
# class HardMod(nn.Module): ...

class GAN_g(nn.Module):
    """
    标准 5层 MLP 生成器 (方案 A: 平滑金字塔)
    """

    def __init__(self, z_dim=32, seq_len=256):
        super(GAN_g, self).__init__()
        self.gen1 = nn.Sequential(
            # Layer 1: 噪声输入 -> 256
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 256 -> 512
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            HardMod(256.0),

            # Layer 3: 512 -> 768 (新增过渡层)
            nn.Linear(512, 768),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 768 -> 1024
            nn.Linear(768, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 5 (Output): 1024 -> seq_len
            nn.Linear(1024, seq_len),
            HardMod(256.0)
        )

    def forward(self, x):
        return self.gen1(x)


class GAN_p(nn.Module):
    def __init__(self, z_dim=32, seq_len=256):  # 这里最好设为你实际使用的默认值
        super(GAN_p, self).__init__()

        # 经过 4 层 stride=2 的卷积，序列长度会除以 16 (2^4)
        # 注意：这里要求你的 seq_len 必须是 16 的倍数
        ds_len = seq_len // 16
        flatten_dim = 128 * ds_len

        self.dis = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 动态维度的全连接层
            nn.Flatten(),
            nn.Linear(flatten_dim, 128),  # <--- 这里不再写死 128*16，而是使用计算出来的 flatten_dim
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.dis(x)

# ----------          model 3      -------------
class GA_Gen_dual_8K(nn.Module):
    """大规模双生成器架构 - 8K序列版本"""
    def __init__(self, z_dim=64, seq_len=8192):
        super(GA_Gen_dual_8K, self).__init__()
        # 第一个生成器 - 输出增强
        self.gen1 = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, z_dim),
            HardMod(256.0),
        )
        # 第二个生成器 - 8K输出
        self.gen2 = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096, seq_len),
            HardMod(256.0),
        )

    def forward(self, x):
        x1 = self.gen1(x)
        x = self.gen2(x1)
        return x


class GA_Disc_8K(nn.Module):
    """大规模CNN判别器 - 8K序列版本"""
    def __init__(self):
        super(GA_Disc_8K, self).__init__()
        self.dis = nn.Sequential(
            # 第一层卷积
            nn.Conv1d(1, 8, 3, padding=1),
            nn.LeakyReLU(0.2),
            # 第二层卷积
            nn.Conv1d(8, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            # 第三层卷积
            nn.Conv1d(16, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            # 第四层卷积
            nn.Conv1d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            # 池化层
            nn.MaxPool1d(2),
            nn.MaxPool1d(2),
            # 展平
            nn.Flatten(),
            # 全连接层
            nn.Linear(64 * (8192 // 4), 1024),  # 8192/4=2048, 2048*64=131072
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.dis(x)




def create_model(model_type=0, device='cpu', z_dim=32, seq_len=256):
    """
    创建模型的工厂函数

    Args:
        model_type: 模型类型
            0 - 基础模型：简单全连接网络（256序列）
            1 - 双生成器模型：双路径生成器 + CNN判别器（256序列）
            2 - 大规模基础模型：增强MLP + CNN判别器
            3 - 大规模双生成器模型：8K序列增强版
        device: 设备
        z_dim: 噪声维度
        seq_len: 序列长度

    Returns:
        generator, discriminator
    """
    # 对于大规模模型，调整默认参数

    if model_type == 3 and seq_len == 256:
        seq_len = 8192
        print(f"警告：模型3 (GA_Disc_8K) 需要序列长度8192，已自动调整")

    if model_type == 0:
        # 基础模型
        gen = Generator(z_dim, seq_len).to(device)
        disc = Discriminator(seq_len).to(device)
    elif model_type == 1:
        # 双生成器模型
        gen = GA_Gen_dual(z_dim=z_dim, seq_len=seq_len).to(device)
        disc = GA_Disc(z_dim=z_dim, seq_len=seq_len).to(device)
    elif model_type == 2:
        # 大规模基础模型
        gen = GAN_g(z_dim=z_dim, seq_len=seq_len).to(device)
        disc = GAN_p(z_dim=z_dim, seq_len=seq_len).to(device)
    elif model_type == 3:
        # 大规模双生成器模型
        gen = GA_Gen_dual_8K(z_dim=z_dim, seq_len=seq_len).to(device)
        disc = GA_Disc_8K().to(device)
    else:
        raise ValueError(f"不支持的模型类型：{model_type}（仅支持 0-3）")

    return gen, disc


if __name__ == "__main__":
    print("--- 正在运行模型单元测试 ---")

    device = 'cpu'

    # 测试模型0：基础模型
    print("\n=== 测试模型0：基础模型 ===")
    gen0, disc0 = create_model(model_type=0, device=device)
    test_z = torch.randn(2, 32)
    output0 = gen0(test_z)
    print(f"基础生成器 - 输入形状：{test_z.shape}, 输出形状：{output0.shape}")

    test_data = torch.randn(3, 256)
    disc_output0 = disc0(test_data)
    print(f"基础判别器 - 输入形状：{test_data.shape}, 输出形状：{disc_output0.shape}")

    # 测试模型1：双生成器模型
    print("\n=== 测试模型1：双生成器模型 ===")
    gen1, disc1 = create_model(model_type=1, device=device)
    output1 = gen1(test_z)
    print(f"双生成器 - 输入形状：{test_z.shape}, 输出形状：{output1.shape}")

    test_data_3d = torch.randn(4, 1, 256)
    disc_output1 = disc1(test_data_3d)
    print(f"CNN判别器 - 输入形状：{test_data_3d.shape}, 输出形状：{disc_output1.shape}")

    # 测试模型2：大规模基础模型
    print("\n=== 测试模型2：大规模基础模型 ===")
    gen2, disc2 = create_model(model_type=2, device=device)
    test_z_large = torch.randn(2, 64)
    output2 = gen2(test_z_large)
    print(f"大规模基础生成器 - 输入形状：{test_z_large.shape}, 输出形状：{output2.shape}")

    test_data_large_2d = torch.randn(3, 1, 256)
    disc_output2 = disc2(test_data_large_2d)
    print(f"大规模基础判别器 - 输入形状：{test_data_large_2d.shape}, 输出形状：{disc_output2.shape}")

    # 测试模型3：大规模双生成器模型
    print("\n=== 测试模型3：大规模双生成器模型 ===")
    gen3, disc3 = create_model(model_type=3, device=device)
    output3 = gen3(test_z_large)
    print(f"大规模双生成器 - 输入形状：{test_z_large.shape}, 输出形状：{output3.shape}")

    test_data_large_3d = torch.randn(4, 1, 8192)
    disc_output3 = disc3(test_data_large_3d)
    print(f"大规模CNN判别器 - 输入形状：{test_data_large_3d.shape}, 输出形状：{disc_output3.shape}")

    # 测试无效模型类型
    print("\n=== 测试无效模型类型 ===")
    try:
        create_model(model_type=4, device=device)
        print("错误：应该抛出异常！")
    except ValueError as e:
        print(f"正确抛出异常：{e}")

    print("\n所有模型测试通过！")

    # 打印参数统计
    print("\n=== 模型参数统计 ===")
    for i in range(4):
        gen, disc = create_model(model_type=i, device=device)
        gen_params = sum(p.numel() for p in gen.parameters())
        disc_params = sum(p.numel() for p in disc.parameters())
        total_params = gen_params + disc_params
        print(f"模型{i}: 生成器 {gen_params:,}, 判别器 {disc_params:,}, 总计 {total_params:,}")

    print("\n=== 模型特性总结 ===")
    print("模型0: 基础全连接架构，256序列长度")
    print("模型1: 双生成器 + CNN判别器，256序列长度")
    print("模型2: 增强MLP生成器 + CNN判别器，推荐8K序列")
    print("模型3: 双生成器 + 深度CNN判别器，8K序列")
