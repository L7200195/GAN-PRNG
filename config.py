"""
GAN-PRNG 配置文件
包含所有超参数和路径配置
"""

import os
from datetime import datetime
import torch


# ==================== 设备配置 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== 实验命名 ====================
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_NAME = f"GAN_训练_{TIMESTAMP}"


# ==================== 路径配置 ====================
CHECKPOINT_DIR = f"./checkpoints/{EXP_NAME}"
LOG_DIR = f"./logs/{EXP_NAME}"
OUTPUT_DIR = "./output"


# ==================== 模型架构参数 ====================
SEQ_LEN = 256          # 生成序列的长度
Z_DIM = 32             # 隐变量（噪声）维度


# ==================== 训练超参数 ====================
BATCH_SIZE = 128       # 批大小
LR = 0.0002            # 学习率
BETAS = (0.9, 0.999)   # Adam 优化器参数
EPOCHS = 50000         # 训练轮数
SAVE_INTERVAL = 1000   # 保存权重的间隔
EVAL_INTERVAL = 1000   # 评估间隔


# ==================== 数据生成参数 ====================
TARGET_BITS = 1048576 * 500  # 最终生成的比特数 (10MB)


# ==================== 评估参数 ====================
BIT_MODE = 1  # 0: 传统模式 (>0 -> 1, <=0 -> 0), 1: 比特展开模式 (uint8)


# ==================== 模型选择 ====================
# 0: Generator/Discriminator (基础模型)
# 1: GA_Gen_dual/GA_Disc (双生成器 + CNN 判别器)
# 2: GAN_g/GA_Disc (增强MLP生成器 + CNN判别器)
# 3: GA_Gen_dual_8K/GA_Disc_8K (大规模双生成器，8K序列)
MODEL_TYPE = 0


# ==================== 生成配置 ====================
TOTAL_BITS = 1048576   # 默认生成的总比特数 (1MB)
USE_CUDA = True        # 是否使用 CUDA


def create_directories():
    """创建必要的目录"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'已创建目录：{CHECKPOINT_DIR}')
    print(f'已创建目录：{LOG_DIR}')
    print(f'已创建目录：{OUTPUT_DIR}')


def print_config():
    """打印当前配置"""
    print("\n" + "="*50)
    print("GAN-PRNG 配置信息")
    print("="*50)
    print(f"设备：{DEVICE}")
    print(f"实验名称：{EXP_NAME}")
    print(f"序列长度 (SEQ_LEN): {SEQ_LEN}")
    print(f"噪声维度 (Z_DIM): {Z_DIM}")
    print(f"批大小 (BATCH_SIZE): {BATCH_SIZE}")
    print(f"学习率 (LR): {LR}")
    print(f"训练轮数 (EPOCHS): {EPOCHS}")
    print(f"保存间隔 (SAVE_INTERVAL): {SAVE_INTERVAL}")
    print(f"评估间隔 (EVAL_INTERVAL): {EVAL_INTERVAL}")
    print(f"目标比特数 (TARGET_BITS): {TARGET_BITS}")
    print(f"模型类型：{'基础模型' if MODEL_TYPE == 0 else '双生成器模型'}")
    print(f"比特模式：{'传统模式' if BIT_MODE == 0 else '比特展开模式'}")
    print("="*50 + "\n")


if __name__ == "__main__":
    # 测试配置
    print_config()
    create_directories()
