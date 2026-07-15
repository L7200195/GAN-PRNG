"""
GAN-PRNG 辅助工具模块
包含随机数生成、文件保存和模型加载等辅助函数
"""

import torch
import numpy as np
import os
import math

from models import create_model
from evaluator import AnalyzerRuns




def generate_speed(generator, total_bits, device='cpu', batch_size=128, z_dim=32, bit_mode=0):
    """
    测试纯推理速度：仅测量 G(z) forward pass 的时间

    不包含：
    - 噪声 z 的生成时间 (torch.randn)
    - G(z) 转换为比特的过程 (clamp, bit unpacking, cpu transfer)

    Args:
        generator: 生成器模型（建议先 .eval()，函数内部也会调用）
        total_bits: 目标生成比特数（用于计算等效吞吐量）
        device: 设备 ('cuda' 或 'cpu')
        batch_size: 批大小
        z_dim: 噪声维度
        bit_mode: 比特模式（0=每样本 seq_len bits, 1=每样本 seq_len*8 bits）

    Returns:
        dict: 测速结果
    """
    generator.eval()

    # 预分配 noise buffer（不参与计时）
    noise = torch.randn(batch_size, z_dim, device=device)

    # 获取 seq_len
    with torch.no_grad():
        sample_out = generator(noise)
    seq_len = sample_out.shape[-1]

    # 每个 forward pass 产出的比特数
    bits_per_forward = batch_size * seq_len * (8 if bit_mode == 1 else 1)
    forwards_needed = max(1, math.ceil(total_bits / bits_per_forward))

    # Warm-up：让 GPU 进入稳态（kernel 编译、频率拉升等）
    with torch.no_grad():
        for _ in range(10):
            _ = generator(noise)

    if device == 'cuda':
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)

        starter.record()
        with torch.no_grad():
            for _ in range(forwards_needed):
                _ = generator(noise)          # 只用预分配的 noise
        ender.record()
        torch.cuda.synchronize()

        total_ms = starter.elapsed_time(ender)
    else:
        import time
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(forwards_needed):
                _ = generator(noise)
            t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000

    avg_ms  = total_ms / forwards_needed
    sps     = batch_size * 1000 / avg_ms
    bps     = bits_per_forward * 1000 / avg_ms
    total_ms_for_bits = (total_bits / bits_per_forward) * avg_ms

    print(f"=== G(z) 纯推理速度 ===")
    print(f"设备: {device} | batch: {batch_size} | seq_len: {seq_len}")
    print(f"单次 forward:       {avg_ms:.4f} ms")
    print(f"吞吐量:             {sps:,.0f} samples/s | {bps / 1e6:.2f} Mbps | {bps / 1e9:.3f} Gbps")
    print(f"生成 {total_bits:,} bits 等效耗时: {total_ms_for_bits:.2f} ms")
    print(f"(不含噪声生成、不含比特转换、不含 CPU 传输)")

    return {
        'forward_ms':    round(avg_ms, 4),
        'samples_per_s': round(sps, 1),
        'bits_per_s':    round(bps, 1),
        'total_bits_ms': round(total_ms_for_bits, 2),
        'forwards':      forwards_needed,
    }

def generate_random_bits(generator, total_bits, device='cpu', batch_size=128, z_dim=32, bit_mode=0):
    """ 生成随机比特
    Args:
        generator: 生成器模型
        total_bits: 需要生成的总比特数
        device: 设备
        batch_size: 批大小
        z_dim: 噪声维度
        bit_mode: 比特生成模式
                 (0: [-1,1]浮点数，大于0为1，小于等于0为0;
                  1: [0,256)浮点数，截断为整数后转为8比特)
    Returns:
        numpy array: 生成的比特 (0/1)
    """
    generator.eval()
    all_bits = []
    generated = 0
    print(f"正在生成 {total_bits:,} 个随机比特...")

    with torch.no_grad():
        while generated < total_bits:
            # 生成噪声
            noise = torch.randn(batch_size, z_dim, device=device)
            # 生成随机数
            #output = generator(noise).cpu().numpy().flatten()
            output = generator(noise).flatten()
            if generated == 0:
                print(output.size())
                print(output[0])

            if bit_mode == 0:
                # 模式0: [-1,1]浮点数，大于0为1，小于等于0为0
                bits = (output > 0).cpu().numpy().astype(np.uint8)
            elif bit_mode == 1:
                output = output.flatten()
                int_samples = output.clamp(0, 255).to(torch.long)
                # ✅ 修复：shifts 是 [1, 8]，int_samples 是 [N, 1]
                shifts = torch.arange(7, -1, -1, device=output.device, dtype=torch.long).view(1, 8)
                bits_tensor = ((int_samples.unsqueeze(1) >> shifts) & 1).flatten()
                bits = bits_tensor.cpu().numpy().astype(np.uint8)
            else:
                raise ValueError(f"不支持的bit_mode: {bit_mode}")

            # 只取需要的数量
            remaining = total_bits - generated
            if len(bits) > remaining:
                bits = bits[:remaining]

            all_bits.append(bits)
            generated += len(bits)

            # 进度显示（避免除零错误）
            if total_bits > 10 and generated % (total_bits // 10) == 0:
                progress = (generated / total_bits) * 100
                print(f"进度：{progress:.1f}% ({generated:,}/{total_bits:,})")
        print(all_bits[0][0:8], all_bits.__len__())

    return np.concatenate(all_bits)


def bits_to_bytes(bits):
    """
    将比特数组转换为字节数组
    
    Args:
        bits: 比特数组 (0/1)
    
    Returns:
        bytes: 字节数组
    """
    n = len(bits)
    if n % 8 != 0:
        # 补零
        bits = np.pad(bits, (0, 8 - n % 8), mode='constant')
    
    # 每 8 个比特打包为 1 个字节
    bytes_array = np.packbits(bits)
    return bytes_array.tobytes()


def save_to_binary(data, output_path):
    """
    保存数据到二进制文件
    
    Args:
        data: 字节数据
        output_path: 输出文件路径
    """
    # 创建目录（如果需要）
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(data)
    
    file_size = os.path.getsize(output_path)
    print(f"已保存至：{output_path}")
    print(f"文件大小：{file_size / (1024*1024):.2f} MB")


def load_generator(checkpoint_path, model_type=0, z_dim=32, seq_len=256, device='cpu'):
    """
    加载生成器模型

    Args:
        checkpoint_path: 检查点文件路径
        model_type: 模型类型 (0-3)
        z_dim: 噪声维度
        seq_len: 序列长度
        device: 设备

    Returns:
        加载了权重的生成器模型
    """
    model_names = {0: '基础模型', 1: '双生成器模型', 2: '大规模基础模型', 3: '大规模双生成器模型'}
    gen, _ = create_model(model_type=model_type, device=device, z_dim=z_dim, seq_len=seq_len)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint)
    gen.to(device)
    gen.eval()

    print(f"已加载检查点：{checkpoint_path}")
    print(f"模型类型：{model_names.get(model_type, f'模型{model_type}')}")

    return gen


if __name__ == "__main__":
    print("--- 辅助工具模块测试 ---")
    
    # 测试模型加载
    from models import create_model
    
    device = 'cpu'
    gen, _ = create_model(model_type=0, device=device)
    
    # 保存测试检查点
    torch.save(gen.state_dict(), './test_checkpoint.pth')
    
    # 加载测试
    loaded_gen = load_generator('./test_checkpoint.pth', model_type=0, device=device)
    
    # 测试生成
    test_bits = generate_random_bits(loaded_gen, total_bits=10000, device=device, batch_size=100)
    print(f"\n生成比特数：{len(test_bits)}")
    print(f"1 的比例：{np.sum(test_bits) / len(test_bits):.4f}")
    
    # 测试保存
    byte_data = bits_to_bytes(test_bits)
    save_to_binary(byte_data, './test_output.bin')
    
    # 清理测试文件
    os.remove('./test_checkpoint.pth')
    os.remove('./test_output.bin')
    
    print("\n测试通过！")
