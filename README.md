# LLM_training - 从零开始构建大语言模型

> 一个完整的大模型训练学习项目，包含预训练、分类微调和指令微调三个完整流程
> 注意！此README暂时由GPT生成，可以临时了解一下，细节还有待优化……

## 项目简介

本项目是学习《从零开始训练大模型》一书的学习记录与实践代码。项目实现了从零开始训练大语言模型的完整流程，包括：

- **预训练 (Pretraining)**：从零开始训练自己的语言模型
- **分类微调 (Classification Fine-tuning)**：在预训练模型上进行下游分类任务微调
- **指令微调 (Instruction Fine-tuning)**：让模型学会遵循指令进行对话

项目特点：
- 代码结构清晰，注释详细，适合学习理解
- 三个任务的完整流程均已走通
- 支持多种模型架构（GPT、Qwen）
- 支持多种数据格式（txt、jsonl、bin）
- 已成功训练600M参数的预训练模型并完成微调

## 目录

- [环境配置](#环境配置)
- [项目结构](#项目结构)
- [数据准备](#数据准备)
- [任务一：预训练](#任务一预训练)
- [任务二：分类微调](#任务二分类微调)
- [任务三：指令微调](#任务三指令微调)
- [常见问题](#常见问题)
- [代码架构优化建议](#代码架构优化建议)

## 环境配置

### 1. 硬件要求

- **推荐配置**：至少一张显存 24GB 的 GPU（如 RTX 4090、A100 等）
- **最低配置**：12GB 显存可运行小模型（124M），但训练大模型需要更多显存
- **CPU**：多核处理器，用于数据预处理
- **内存**：建议 32GB 以上
- **存储**：至少 100GB 可用空间（用于存储数据和模型权重）

### 2. 软件依赖

```bash
# 创建虚拟环境
conda create -n llm python=3.11
conda activate llm

# 安装 PyTorch（根据你的 CUDA 版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install tiktoken
pip install datasets
pip install transformers
pip install accelerate
pip install gradio
pip install pandas
pip install numpy
pip install tqdm
```

或者直接安装项目依赖：
```bash
pip install -r requirements.txt
```

### 3. 验证安装

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
```

## 项目结构

```
LLM_training/
├── data/                           # 数据目录
│   ├── pretrain/                   # 预训练数据
│   │   └── CCI3/                   # CCI3 数据集
│   │       ├── data_bin/           # 预处理后的 bin 文件
│   │       │   ├── train/          # 训练集
│   │       │   └── val/            # 验证集
│   │       └── raw/                # 原始数据
│   ├── classification/             # 分类数据
│   │   └── sms_spam_collection/    # 垃圾短信分类数据集
│   │       ├── train.csv
│   │       ├── val.csv
│   │       └── test.csv
│   └── instruction/                # 指令微调数据
│       ├── BelleGroup/             # Belle 数据集
│       └── simple_instruction/     # 简单指令数据
│
├── src/llm/                        # 源代码目录
│   ├── model/                      # 模型定义
│   │   ├── gpt.py                  # GPT 模型
│   │   ├── qwen/                   # Qwen 模型
│   │   │   ├── qwen.py             # Qwen 模型实现
│   │   │   ├── attention.py        # 注意力机制
│   │   │   └── transformer.py      # Transformer 层
│   │   ├── attention.py            # GPT 注意力机制
│   │   └── transformer.py          # GPT Transformer 层
│   │
│   ├── config/                     # 配置文件
│   │   └── gpt_configs.py          # 模型配置（124M、355M、600M、774M、1.2B 等）
│   │
│   ├── tasks/                      # 任务代码
│   │   ├── pretrain/               # 预训练任务
│   │   │   ├── pretraining.py      # 单机预训练
│   │   │   ├── pretraining_DDP.py  # DDP 并行预训练
│   │   │   ├── pretraining_FSDP.py # FSDP 并行预训练
│   │   │   ├── datasets.py         # 数据加载器
│   │   │   └── loss.py             # 损失计算
│   │   │
│   │   ├── classification/         # 分类微调任务
│   │   │   ├── fine_tuning.py      # 微调脚本
│   │   │   ├── datasets.py         # 数据加载器
│   │   │   ├── loss.py             # 损失计算
│   │   │   └── eval.py             # 评估函数
│   │   │
│   │   └── instruction/            # 指令微调任务
│   │       ├── fine_tuning_huggingface.py  # 使用 HF 训练
│   │       ├── fine_tuning_self.py         # 自定义训练
│   │       ├── dataset.py          # 数据加载器
│   │       ├── loss.py             # 损失计算
│   │       ├── eval.py             # 评估函数
│   │       ├── lora.py             # LoRA 微调
│   │       └── gradio_demo.py      # Gradio 演示
│   │
│   ├── eval/                       # 推理评估
│   │   ├── tokenizer.py            # 分词器工具
│   │   └── generate.py             # 文本生成
│   │
│   └── utils/                      # 工具函数
│       ├── logger.py               # 日志工具
│       ├── loss_tracker.py         # 损失追踪
│       └── gpt_download.py         # GPT 权重下载
│
├── scripts/                        # 脚本工具
│   └── pretokenize_jsonl.py        # 数据预处理脚本
│
├── logs/                           # 训练日志和检查点
│   ├── pretraining/                # 预训练日志
│   ├── classification/             # 分类微调日志
│   └── instruction/                # 指令微调日志
│
├── notebook/                       # Jupyter 笔记本（学习记录）
│
├── gpt2/                           # GPT2 预训练权重
├── qwen2.5/                        # Qwen2.5 模型
├── qwen3/                          # Qwen3 模型
│
└── README.md                       # 本文件
```

## 数据准备

### 预训练数据

预训练需要大量的文本数据。本项目支持以下格式：

#### 方法一：使用 jsonl 文件

数据格式（每行一个 JSON 对象）：
```json
{"text": "这是一段训练文本..."}
{"text": "这是另一段训练文本..."}
```

预处理（转换为 bin 文件以加速训练）：
```bash
cd scripts
python pretokenize_jsonl.py
```

#### 方法二：使用 txt 文件

直接使用纯文本文件，会在训练时进行分词。

### 分类微调数据

使用 CSV 格式，包含文本和标签：

```csv
Text,Label
"You are a winner you have been specially selected to receive $1000 cash",spam
"Hi Mom, how are you?",ham
```

数据集划分：
- `train.csv`：训练集（通常占 80%）
- `val.csv`：验证集（通常占 10%）
- `test.csv`：测试集（通常占 10%）

### 指令微调数据

使用 jsonl 格式，每条数据包含 instruction、input 和 output：

```json
{"instruction": "请写一篇关于保护环境的文章", "input": "", "output": "保护环境是我们每个人的责任..."}
{"instruction": "解释什么是机器学习", "input": "", "output": "机器学习是人工智能的一个分支..."}
```

常用的指令微调数据集：
- BelleGroup/train_0.5M_CN：50万条中文指令数据
- Alpaca 数据集
- 自定义指令数据

## 任务一：预训练

### 1. 选择模型配置

在 `src/llm/config/gpt_configs.py` 中选择或创建模型配置：

```python
# 可选配置
GPT_CONFIG_124M    # 124M 参数
GPT_CONFIG_355M    # 355M 参数
QWEN_CONFIG_600M   # 600M 参数（Qwen 架构）
QWEN_CONFIG_1_2B   # 1.2B 参数（Qwen 架构）
```

### 2. 准备训练脚本

主要训练脚本位于 `src/llm/tasks/pretrain/pretraining.py`

关键参数说明：
```python
# 模型配置
MODEL_CONFIG = QWEN_CONFIG_600M  # 选择模型大小

# 分词器
tokenizer = tiktoken.get_encoding("cl100k_base")  # 中文推荐
# tokenizer = tiktoken.get_encoding("gpt2")       # 英文推荐

# 更新词表大小
MODEL_CONFIG["vocab_size"] = tokenizer.max_token_value

# 创建模型
model = QwenModel(MODEL_CONFIG)  # 或 GPTModel(MODEL_CONFIG)

# 数据加载器
train_loader = create_dataloader_bins(
    "/path/to/train/bin/dir",     # 训练数据路径
    batch_size=2,                  # 批次大小（根据显存调整）
    max_length=MODEL_CONFIG["context_length"],  # 序列长度
    stride=MODEL_CONFIG["context_length"],      # 滑窗步长
    drop_last=True,
    shuffle=True,
    num_workers=0                  # 数据加载进程数
)

# 设备和优化器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)

# 训练参数
num_epochs = 1                    # 训练轮数
eval_freq = 4000                  # 评估频率（每 N 个 step 评估一次）
eval_iter = 40                    # 评估时使用的批次数量
save_epoch = 1                    # 保存频率（每 N 个 epoch 保存一次）
save_step = 50000                 # 保存频率（每 N 个 step 保存一次）
```

### 3. 开始训练

```bash
cd src/llm/tasks/pretrain
python pretraining.py
```

### 4. 监控训练进度

训练日志会保存在 `logs/pretraining/` 目录下，包括：
- `train.log`：详细训练日志
- `checkpoints/`：模型检查点
- `samples/loss.png`：损失曲线图

### 5. 从检查点恢复训练

如果训练中断，可以从检查点恢复：

```python
# 在 pretraining.py 中取消注释并修改路径
checkpoint_model = torch.load("/path/to/checkpoint/model_epoch1_step50000.pt", map_location=device)
model.load_state_dict(checkpoint_model)
model.train()

# 可选：恢复优化器状态
# checkpoint_optimizer = torch.load("/path/to/checkpoint/optimizer_epoch1_step50000.pt", map_location=device)
# optimizer.load_state_dict(checkpoint_optimizer)

# 可选：重新设置学习率
# new_lr = 1e-5
# for param_group in optimizer.param_groups:
#     param_group["lr"] = new_lr
```

### 6. 预训练建议

- **批次大小**：尽可能大，但要在显存限制内
- **学习率**：1e-5 到 1e-4 之间，从小开始
- **显存优化**：使用 gradient checkpointing、混合精度训练
- **数据量**：预训练需要至少 10B tokens 的数据

## 任务二：分类微调

### 1. 准备数据

确保分类数据按以下格式组织：
```
data/classification/sms_spam_collection/
├── train.csv
├── val.csv
└── test.csv
```

### 2. 选择基础模型

可以使用：
- 自己预训练的模型
- GPT2 预训练权重

### 3. 修改训练脚本

主要脚本：`src/llm/tasks/classification/fine_tuning.py`

关键配置：
```python
# 模型配置
CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

# 分词器
tokenizer = tiktoken.get_encoding("gpt2")

# 加载数据
train_loader = create_dataloader_Spam(
    "/path/to/train.csv",
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eot_token,
    batch_size=8,
    max_length=None,
    shuffle=True,
    drop_last=True,
    num_workers=0
)

# 加载模型（使用 GPT2 预训练权重）
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.to(device)

# 冻结大部分参数，只微调最后一层
for param in model.parameters():
    param.requires_grad = False

# 设置分类头
num_classes = 2  # 二分类
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

# 解冻最后一层 Transformer 和分类头
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

# 训练参数
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 10
eval_freq = 50
eval_iter = 5
save_epoch = 5
```

### 4. 开始训练

```bash
cd src/llm/tasks/classification
python fine_tuning.py
```

### 5. 评估模型

训练完成后，会输出训练集、验证集和测试集的准确率。

## 任务三：指令微调

### 方法一：使用 HuggingFace 训练（推荐）

脚本：`src/llm/tasks/instruction/fine_tuning_huggingface.py`

```python
# 加载预训练模型
model_path = "/path/to/your/base/model"  # 可以是自己预训练的或 HF 模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(device)

# 设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载数据集
from datasets import load_dataset
ds = load_dataset("/path/to/instruction/data")

# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=10,
    eval_freq=1000,
    eval_iter=20,
    tokenizer=tokenizer,
    sample_prompt=sample_prompt,
    save_epoch=1,
    save_step=50000
)
```

### 方法二：自定义训练

脚本：`src/llm/tasks/instruction/fine_tuning_self.py`

支持使用自己实现的 Qwen 模型进行指令微调。

### 数据格式

指令微调使用 Alpaca 格式：
```json
{
    "instruction": "请写一篇关于保护环境的文章",
    "input": "",
    "output": "保护环境是我们每个人的责任..."
}
```

### 训练模板

模型使用以下格式进行训练：
```
### Instruction:
请写一篇关于保护环境的文章

### Input:


### Response:
保护环境是我们每个人的责任...
```

### 开始训练

```bash
cd src/llm/tasks/instruction
python fine_tuning_huggingface.py
```

### 模型推理

训练完成后，可以使用 `gradio_demo.py` 创建交互式演示：

```bash
python gradio_demo.py
```

## 常见问题

### Q1: 显存不足怎么办？

1. 减小 `batch_size`
2. 减小 `max_length`
3. 使用梯度累积
4. 使用混合精度训练（已默认启用）
5. 使用梯度检查点（gradient checkpointing）
6. 使用更小的模型

### Q2: 训练速度慢怎么办？

1. 增大 `batch_size`（在显存允许范围内）
2. 增加 `num_workers`（数据加载进程数）
3. 使用多 GPU 训练（DDP/FSDP）
4. 预处理数据为 bin 格式
5. 使用更快的存储（SSD）

### Q3: 损失不下降怎么办？

1. 检查学习率是否过大或过小
2. 检查数据是否正确加载
3. 检查模型是否正确初始化
4. 尝试调整权重衰减参数
5. 检查是否所有参数都正确解冻

### Q4: 如何选择合适的模型大小？

- **124M**：快速实验，单卡 12GB 可运行
- **355M**：中等规模，需要 16-24GB 显存
- **600M**：较大规模，需要 24GB+ 显存，推荐配置
- **1.2B+**：大规模，需要多卡或专业 GPU

### Q5: 如何评估模型效果？

- **预训练**：观察损失曲线、生成样本质量
- **分类微调**：测试集准确率
- **指令微调**：人工评估生成质量、使用评估数据集

## 代码架构优化建议

虽然当前代码结构可以正常工作，但为了更好的工程化，以下是一些优化建议：

### 1. 配置管理

**建议**：使用配置文件（如 YAML 或 JSON）替代硬编码参数

```yaml
# config/pretraining/qwen600m.yaml
model:
  name: qwen
  config: QWEN_CONFIG_600M

training:
  batch_size: 2
  num_epochs: 1
  learning_rate: 1e-5
  eval_freq: 4000
  save_step: 50000

data:
  train_path: "/data/pretrain/CCI3/data_bin/train"
  val_path: "/data/pretrain/CCI3/data_bin/val"
  max_length: 1024
```

### 2. 命令行参数

**建议**：使用 argparse 或 hydra 支持命令行参数

```bash
python pretraining.py --config config/qwen600m.yaml --device cuda:0 --resume checkpoint.pt
```

### 3. 代码模块化

**建议**：
- 将训练循环抽象为通用的 Trainer 类
- 统一数据加载接口
- 提取公共的评估和保存逻辑

```python
# src/llm/trainer.py
class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, config):
        ...

    def train(self):
        ...

    def evaluate(self):
        ...

    def save_checkpoint(self):
        ...
```

### 4. 路径管理

**建议**：使用相对路径和环境变量

```python
import os
from pathlib import Path

# 设置项目根目录
PROJECT_ROOT = Path(os.getenv("LLM_PROJECT_ROOT", "."))
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
```

### 5. 日志和监控

**建议**：
- 集成 wandb 或 tensorboard
- 统一日志格式
- 添加更多训练指标

### 6. 测试

**建议**：添加单元测试和集成测试

```
tests/
├── test_model.py
├── test_data.py
└── test_training.py
```

### 7. 文档

**建议**：
- 添加函数和类的 docstring
- 使用类型注解
- 添加示例代码

## 参考资源

- 《从零开始训练大模型》
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Qwen 模型](https://github.com/QwenLM/Qwen)

## 许可证

本项目仅供学习交流使用。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，欢迎提 Issue 讨论。
