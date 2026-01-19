import os
import json
from collections import Counter
import random
from pathlib import Path
from typing import Tuple

def format_input(entry):
    instruction_text = (         
        f"Below is an instruction that describes a task. "         
        f"Write a response that appropriately completes the request."         
        f"\n\n### Instruction:\n{entry['instruction']}"     
    )
    input_text = (         
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""     
    )     
    return instruction_text + input_text

def split_jsonl_dataset(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    shuffle: bool = True,
    ) -> Tuple[int, int, int]:
    """
    通用 jsonl 数据集划分函数（每行一个 JSON）

    Args:
        input_path: 原始 jsonl 文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        shuffle: 是否在切分前打乱

    Returns:
        (train_size, val_size, test_size)
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train/val/test ratio must sum to 1.0"

    random.seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 读取 jsonl
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    total = len(data)
    if total == 0:
        raise ValueError("Empty jsonl file")

    # 打乱
    if shuffle:
        random.shuffle(data)

    # 切分
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # 写回 jsonl
    def write_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for x in items:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    write_jsonl(Path(output_dir) / "train.jsonl", train_data)
    write_jsonl(Path(output_dir) / "val.jsonl", val_data)
    write_jsonl(Path(output_dir) / "test.jsonl", test_data)

    return len(train_data), len(val_data), len(test_data)

def sample_jsonl_dataset(
    input_path: str,
    output_path: str,
    sample_size: int,
    seed: int = 42,
    ):
    """
    从 jsonl 文件中随机采样 sample_size 条数据，写回新的 jsonl

    Args:
        input_path: 输入 jsonl 路径
        output_path: 输出 jsonl 路径
        sample_size: 采样条数
        seed: 随机种子
    """

    random.seed(seed)

    # 读取数据
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    total = len(data)
    if total == 0:
        raise ValueError("Empty jsonl file")

    # 处理 sample_size
    if sample_size >= total:
        sampled = data
        print(f"[INFO] sample_size >= total ({total}), use full dataset.")
    else:
        sampled = random.sample(data, sample_size)

    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 写回 jsonl
    with open(output_path, "w", encoding="utf-8") as f:
        for x in sampled:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(
        f"[DONE] Sampled {len(sampled)} / {total} "
        f"from {input_path} -> {output_path}"
    )

if __name__ == "__main__":
    # train_n, val_n, test_n = split_jsonl_dataset(
    #     input_path="/home/hjzd/lzz/LLM_training/data/instruction/belle_data/Belle_open_source_0.5M.json",
    #     output_dir="/home/hjzd/lzz/LLM_training/data/instruction/belle_data",
    #     train_ratio=0.90,
    #     val_ratio=0.05,
    #     test_ratio=0.05,
    #     seed=42,
    # )

    sample_jsonl_dataset(
        input_path="/home/hjzd/lzz/LLM_training/data/instruction/belle_data/train.jsonl",
        output_path="/home/hjzd/lzz/LLM_training/data/instruction/belle_data/train_150k.jsonl",
        sample_size=150_000,
        seed=42,
    )

# print(f"Train: {train_n}, Val: {val_n}, Test: {test_n}")
    
    


    