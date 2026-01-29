import os
import json
import re
from collections import Counter
import random
from pathlib import Path
from typing import Tuple
from typing import Union

## Phi-3 指令微调格式
def format_input_Phi(entry):
    user = entry[0]
    assistant = entry[1]

    user_content = user["content"]
    assistant_content = assistant["content"]

    user_text = (
        "<User>\n"
        f"{user_content}\n"
        "</User>\n\n"
        "<Assistant>\n"
        f"{assistant_content}\n"
        "</Assistant>"
    )
    return user_text


## Alpaca 指令微调格式
def format_input_Alpaca(entry):
    instruction_text = (         
        f"Below is an instruction that describes a task. "         
        f"Write a response that appropriately completes the request."         
        f"\n\n### Instruction:\n{entry['instruction']}"     
    )
    input_text = (         
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""     
    )     
    return instruction_text + input_text


## 切分jsonl，创建训练、验证、测试集
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


## 从jsonl中随机取出样本，用于构建小数据集
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


## 查看jsonl前n行的数据
def preview_jsonl(file_path, n=5):
    """
    查看 jsonl 文件前 n 行
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            data = json.loads(line)
            print(f"---- line {i+1} ----")
            print(data)


## 用于检查单个jsonl文件格式是否正确
def is_valid_jsonl_file(file_path: str) -> bool:
    """
    检查单个 jsonl 文件是否格式正确
    - 后缀为 .jsonl
    - 每一行都是合法 JSON
    """
    if not file_path.endswith(".jsonl"):
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # 允许空行
                json.loads(line)
        return True
    except Exception as e:
        print(f"[格式错误] 文件: {file_path}, 错误: {e}")
        return False


## 统计 jsonl 文件或文件夹中的数据条数
def count_jsonl_samples(path: Union[str, os.PathLike]) -> int:
    """
    统计 jsonl 文件或文件夹中的数据条数

    规则：
    - 输入是文件：必须是合法 jsonl，才统计
    - 输入是文件夹：
        - 文件夹下必须全部是合法 jsonl 文件
        - 否则直接返回 0
    """
    path = str(path)

    # 情况 1：单个文件
    if os.path.isfile(path):
        if not is_valid_jsonl_file(path):
            print("[终止] 文件不是合法的 jsonl")
            return 0

        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    # 情况 2：文件夹
    elif os.path.isdir(path):
        files = [
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if os.path.isfile(os.path.join(path, fname))
        ]

        if not files:
            print("[终止] 文件夹为空")
            return 0

        # 先检查所有文件是否为合法 jsonl
        for file_path in files:
            if not is_valid_jsonl_file(file_path):
                print("[终止] 文件夹中存在非法 jsonl 文件")
                return 0

        # 所有文件合法，开始统计
        total_count = 0
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                total_count += sum(1 for line in f if line.strip())

        return total_count

    else:
        print("[错误] 路径不存在")
        return 0

if __name__ == "__main__":
    # train_n, val_n, test_n = split_jsonl_dataset(
    #     input_path="/home/hjzd/lzz/LLM_training/data/instruction/belle_data/Belle_open_source_0.5M.json",
    #     output_dir="/home/hjzd/lzz/LLM_training/data/instruction/belle_data",
    #     train_ratio=0.90,
    #     val_ratio=0.05,
    #     test_ratio=0.05,
    #     seed=42,
    # )

    # print(count_jsonl_samples("/home/hjzd/lzz/LLM_training/data/instruction/minimind_dataset/sft_1024.jsonl"))
    # preview_jsonl("/home/hjzd/lzz/LLM_training/data/instruction/minimind_dataset/sft_1024.jsonl", 3)

    # filter_chinese_jsonl(
    #     input_path="/home/hjzd/lzz/LLM_training/data/instruction/minimind_dataset/sft_1024.jsonl",
    #     output_path="/home/hjzd/lzz/LLM_training/data/instruction/minimind_dataset/sft_1024_zh.jsonl",
    # )


    sample_jsonl_dataset(
        input_path="/home/hjzd/lzz/LLM_training/data/instruction/belle_data/test.jsonl",
        output_path="/home/hjzd/lzz/LLM_training/data/instruction/belle_data/test_2.jsonl",
        sample_size=2,
        seed=42,
    )


    # from datasets import load_dataset

    # ds = load_dataset("/home/hjzd/lzz/LLM_training/data/instruction/BelleGroup/train_0.5M_CN")

    # print(ds)
    # print(ds["train"][0])

    


    