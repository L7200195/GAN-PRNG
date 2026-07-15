"""
GAN-PRNG 评估模块
包含随机性质量指标的计算和评估
"""

import torch
import numpy as np
from scipy.special import erfc


class AnalyzerRuns:
    """随机性质量指标分析类 (高性能版)"""

    @staticmethod
    def get_metrics(samples, bit_mode=0):
        """
        计算随机性指标。

        参数:
            samples: 输入数据 (np.ndarray 或 torch.Tensor)
            bit_mode:
                0 - 传统模式：>0 -> 1, <=0 -> 0
                1 - 比特展开模式：将数据视为 uint8 (0-255), 向量化展开为 8 个比特位
        
        返回:
            dict: 包含 1 占比、自相关性、Runs_P 值等指标
        """
        # 1. 统一转换为 torch.Tensor (Float32)
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples).float()
        else:
            samples = samples.float()

        # 2. 核心：二值化处理
        flat_samples = None

        if bit_mode == 0:
            # 【模式 0】传统逻辑
            flat_samples = (samples.flatten() > 0).float()

        elif bit_mode == 1:
            # 【模式 1】高性能向量化比特展开
            # A. 预处理：限制范围 [0, 255] 并四舍五入转为 Long
            int_samples = samples.clamp(0, 255).to(torch.long)

            # B. 构造位移向量 (7, 6, 5, 4, 3, 2, 1, 0)
            shifts = torch.arange(7, -1, -1, device=int_samples.device, dtype=torch.long)
            shift_shape = [8] + [1] * int_samples.dim()
            shifts = shifts.view(shift_shape)

            # C. 向量化位提取
            bits_tensor = (int_samples >> shifts) & 1

            # D. 展平
            flat_samples = bits_tensor.flatten().float()

        else:
            raise ValueError(f"Unsupported bit_mode: {bit_mode}. Use 0 or 1.")

        n = len(flat_samples)

        if n == 0:
            return {"1 占比": 0, "自相关性": 0, "Runs_P 值": 0}

        # 3. 计算 1 的占比
        ones_count = torch.sum(flat_samples).item()
        ones_ratio = ones_count / n

        # 4. 计算滞后为 1 的自相关系数
        autocorr = 0.0
        if n > 1:
            y1, y2 = flat_samples[:-1], flat_samples[1:]
            combined = torch.stack([y1, y2])
            try:
                corr_matrix = torch.corrcoef(combined)
                val = corr_matrix[0, 1].item()
                autocorr = 0.0 if np.isnan(val) else val
            except RuntimeError:
                autocorr = 0.0

        # 5. NIST Runs Test
        tau = 2 / np.sqrt(n)
        runs_p_value = 0.0

        if abs(ones_ratio - 0.5) < tau:
            diff = torch.diff(flat_samples)
            v_n = torch.sum(torch.abs(diff)).item() + 1

            pi = ones_ratio
            numerator = abs(v_n - 2 * n * pi * (1 - pi))
            denominator = 2 * np.sqrt(2 * n) * pi * (1 - pi)

            if denominator > 1e-6:
                runs_p_value = erfc(numerator / denominator)

        return {
            "1 占比": round(ones_ratio, 4),
            "自相关性": round(autocorr, 4),
            "Runs_P 值": round(runs_p_value, 8)
        }


def evaluate_randomness(bits, bit_mode=0):
    """
    评估生成数据的随机性
    
    Args:
        bits: 比特数组
        bit_mode: 比特模式
    
    Returns:
        dict: 随机性指标
    """
    print("\n" + "="*50)
    print("随机性评估")
    print("="*50)
    
    ones_ratio = np.sum(bits) / len(bits)
    print(f"1 的比例：{ones_ratio:.4f} (理想值：0.5)")
    
    metrics = AnalyzerRuns.get_metrics(torch.tensor(bits, dtype=torch.float32), bit_mode=bit_mode)
    print(f"1 占比：{metrics['1 占比']}")
    print(f"自相关性：{metrics['自相关性']} (理想值：0)")
    print(f"Runs_P 值：{metrics['Runs_P 值']} (理想值：>0.01)")
    print("="*50 + "\n")
    
    return metrics


def evaluate_generator(generator, device, num_samples=10, bit_mode=0):
    """
    评估生成器的随机性质量
    
    Args:
        generator: 生成器模型
        device: 设备
        num_samples: 评估样本数量
        bit_mode: 比特模式
    
    Returns:
        dict: 平均指标
    """
    generator.eval()
    
    all_metrics = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            noise = torch.randn(1, 32, device=device)
            output = generator(noise)
            metrics = AnalyzerRuns.get_metrics(output, bit_mode)
            all_metrics.append(metrics)
    
    # 计算平均指标
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    return avg_metrics


if __name__ == "__main__":
    print("--- 评估模块测试 ---")
    
    # 测试 1: 随机数据
    random_data = torch.randn(32, 256)
    metrics1 = AnalyzerRuns.get_metrics(random_data, bit_mode=0)
    print(f"随机数据指标：{metrics1}")
    
    # 测试 2: 均匀分布数据
    uniform_data = torch.rand(32, 256) * 2 - 1
    metrics2 = AnalyzerRuns.get_metrics(uniform_data, bit_mode=0)
    print(f"均匀分布指标：{metrics2}")
    
    # 测试 3: 比特展开模式
    int_data = torch.randint(0, 256, (32, 256)).float()
    metrics3 = AnalyzerRuns.get_metrics(int_data, bit_mode=1)
    print(f"比特展开模式指标：{metrics3}")
    
    print("\n测试通过！")
