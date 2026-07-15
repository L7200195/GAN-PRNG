# GAN-PRNG 项目

使用生成对抗网络 (GAN) 生成高质量随机数的项目。

## 项目结构

```
gan_prng_project/
├── config.py           # 配置文件（所有超参数和路径）
├── models.py           # 模型定义（生成器、判别器）
├── data_loader.py      # 数据加载（生成训练数据、噪声）
├── evaluator.py        # 评估模块（随机性质量指标）
├── trainer.py          # 训练器（训练循环、优化器）
├── utils.py            # 辅助工具（随机数生成、文件 I/O、模型加载）
├── train.ipynb         # 训练 Notebook（可视化训练）
├── generate.ipynb      # 生成 Notebook（随机数生成）
└── README.md           # 使用说明
```

## 快速开始

### 1. 环境要求

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- Matplotlib
- Jupyter Notebook
- TensorBoard (可选)

安装依赖：
```bash
pip install torch numpy scipy matplotlib jupyter tensorboard
```

### 2. 训练模型

打开 Jupyter Notebook：
```bash
jupyter notebook
```

然后打开 `train.ipynb`，按照以下步骤操作：

1. **配置参数** - 在第二个单元格中修改训练参数：
   - `MODEL_TYPE`: 模型类型 (0=基础模型，1=双生成器模型)
   - `EPOCHS`: 训练轮数
   - `BATCH_SIZE`: 批大小
   - `LR`: 学习率
   - 等等...

2. **创建模型** - 运行模型创建单元格

3. **开始训练** - 运行训练单元格，会实时显示：
   - 损失曲线
   - 随机性指标（1 占比、自相关性、Runs P 值）
   - 定期保存检查点

4. **查看结果** - 训练完成后会显示：
   - 训练损失图
   - 随机性质量分析
   - 模型测试

### 3. 生成随机数

训练完成后，打开 `generate.ipynb`：

1. **配置检查点路径** - 设置 `CHECKPOINT_PATH` 为训练好的模型文件路径

2. **加载模型** - 运行加载单元格

3. **生成随机数** - 运行生成单元格

4. **评估质量** - 自动评估生成随机数的质量

5. **导出文件** - 保存为二进制文件

## 模块说明

### config.py - 配置模块
包含所有超参数和路径配置：
- 设备配置（CPU/CUDA）
- 模型参数（序列长度、噪声维度）
- 训练超参数（学习率、批大小、轮数）
- 路径配置（检查点、日志、输出目录）

### models.py - 模型定义
定义 GAN 网络架构：
- `Generator`: 基础生成器模型
- `Discriminator`: 基础判别器模型
- `GA_Gen_dual`: 双生成器架构
- `GA_Disc`: CNN 判别器架构
- `create_model()`: 模型创建工厂函数

### data_loader.py - 数据加载
生成训练数据：
- `generate_real_data_float()`: 生成真实数据
- `generate_noise()`: 生成噪声输入
- `prepare_batch()`: 准备训练批次
- `DataGenerator`: 数据生成器类

### evaluator.py - 评估模块
随机性质量评估：
- `AnalyzerRuns`: 随机性指标分析类
  - 1 占比
  - 自相关性
  - Runs P 值
- `evaluate_randomness()`: 评估函数
- `evaluate_generator()`: 生成器评估函数

### trainer.py - 训练器
训练循环和管理：
- `Trainer` 类：
  - 单步训练
  - 训练轮次
  - 评估和日志记录
  - 检查点保存/加载
  - TensorBoard 集成

### utils.py - 辅助工具
常用辅助函数：
- `generate_random_bits()`: 生成随机比特
- `bits_to_bytes()`: 比特转字节
- `save_to_binary()`: 保存二进制文件
- `load_generator()`: 加载生成器模型

## 配置说明

### 训练参数

```python
# config.py
MODEL_TYPE = 0          # 0=基础模型，1=双生成器模型
SEQ_LEN = 256           # 序列长度
Z_DIM = 32              # 噪声维度
BATCH_SIZE = 128        # 批大小
LR = 0.0002             # 学习率
EPOCHS = 50000          # 训练轮数
SAVE_INTERVAL = 1000    # 保存间隔
EVAL_INTERVAL = 1000    # 评估间隔
```

### 生成参数

```python
# config.py
CHECKPOINT_PATH = ''    # 检查点文件路径
TOTAL_BITS = 1048576    # 生成的总比特数 (1MB)
BATCH_SIZE = 128        # 批大小
BIT_MODE = 0            # 0=传统模式，1=比特展开模式
```

## 随机性指标

- **1 占比**: 理想值为 0.5（50%）
- **自相关性**: 理想值为 0（无相关性）
- **Runs P 值**: 理想值 > 0.01（通过 Runs 检验）

## 查看训练日志

使用 TensorBoard 查看详细训练日志：

```bash
tensorboard --logdir=./logs/实验名称
```

然后在浏览器中访问：http://localhost:6006

## 输出文件

训练和生成后会创建以下目录：

```
├── checkpoints/        # 模型检查点
│   └── GAN_训练_时间戳/
│       └── 生成器_轮次_X.pth
├── logs/              # TensorBoard 日志
│   └── GAN_训练_时间戳/
├── output/            # 生成的随机数文件
│   └── random_时间戳.bin
└── training_plot_时间戳.png  # 训练图表
```

## 优势

1. **模块化设计**: 清晰的功能分离，易于理解和维护
2. **直观的界面**: 使用 Jupyter Notebook，无需命令行参数
3. **实时可视化**: 训练过程中实时显示损失和指标
4. **灵活配置**: 在 Notebook 中直接修改参数，立即生效
5. **完整评估**: 自动生成随机性质量分析报告

## 常见问题

**Q: 训练速度慢？**
- 减小 `BATCH_SIZE` 或 `EPOCHS`
- 使用 CUDA 加速（设置 `USE_CUDA = True`）

**Q: 随机性质量不好？**
- 增加训练轮数
- 调整学习率
- 尝试不同的模型类型

**Q: 如何继续训练？**
- 在 `generate.ipynb` 中加载已有检查点
- 修改参数后继续训练

**Q: 如何修改模型架构？**
- 编辑 `models.py` 中的模型定义
- 添加新的层或修改网络结构

## 许可证

MIT License
