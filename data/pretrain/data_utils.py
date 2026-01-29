import os
import numpy as np
from typing import Union
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tiktoken
import multiprocessing as mp

## 用于计算bin文件的token数量
def count_tokens_in_bin(
    path: Union[str, os.PathLike],
    dtype: str = "int32",
    suffix: str = ".bin",
    recursive: bool = True,
    ):
    """
    统计 bin 文件中的 token 数量

    参数:
        path: 单个 .bin 文件 或 包含 .bin 文件的目录
        dtype: token 的数据类型: "int32" | "int16" | "int64"
        suffix: bin 文件后缀
        recursive: 是否递归统计子目录

    返回:
        total_tokens (int)
    """
    dtype_map = {
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
    }

    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")

    np_dtype = dtype_map[dtype]
    total_tokens = 0
    file_count = 0

    def count_file(file_path):
        nonlocal total_tokens, file_count
        data = np.fromfile(file_path, dtype=np_dtype)
        total_tokens += data.size
        file_count += 1

    if os.path.isfile(path):
        count_file(path)

    elif os.path.isdir(path):
        if recursive:
            for root, _, files in os.walk(path):
                for fname in files:
                    if fname.endswith(suffix):
                        count_file(os.path.join(root, fname))
        else:
            for fname in os.listdir(path):
                if fname.endswith(suffix):
                    count_file(os.path.join(path, fname))
    else:
        raise FileNotFoundError(path)

    print(f"Scanned {file_count} bin files")
    print(f"Total tokens: {total_tokens:,}")
    print(f"≈ {total_tokens / 1e9:.3f} B tokens")

    return total_tokens

def process_one_jsonl(args):
    jsonl_path, out_dir, enc_name, eos_token_id = args

    tokenizer = tiktoken.get_encoding(enc_name)

    out_path = out_dir / (jsonl_path.stem + ".bin")
    if out_path.exists():
        print(f"[skip] {out_path}")
        return

    print(f"[processing] {jsonl_path.name}")
    all_tokens = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                text = json.loads(line).get("text", "")
                if not text:
                    continue
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)
                all_tokens.append(eos_token_id)
            except Exception:
                continue

    arr = np.array(all_tokens, dtype=np.uint32) ## 这里尤其需要注意dtype, 这个不会影响模型的效果, 是根据tokenizer的vocab_size来定的, 定小了会有溢出风险
    arr.tofile(out_path)

    print(f"[saved] {out_path} | tokens={len(arr):,}")


def pretokenize_jsonl_dir_mp(jsonl_dir, out_dir, enc_name, eos_token_id, num_workers):
    jsonl_dir = Path(jsonl_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))

    tasks = [
        (path, out_dir, enc_name, eos_token_id)
        for path in jsonl_files
    ]

    with mp.Pool(processes=num_workers) as pool:
        pool.map(process_one_jsonl, tasks)

if __name__ == "__main__":
    total_tokens = count_tokens_in_bin(
        "/home/hjzd/lzz/LLM_training/data/pretrain/CCI3/data_bin/train",
        dtype="int32"
    )

    # enc_name = "cl100k_base"
    # tokenizer = tiktoken.get_encoding(enc_name)

    # pretokenize_jsonl_dir_mp(
    #     jsonl_dir="/home/hjzd/lzz/LLM_training/data/pretrain/CCI3/original_data/train",
    #     out_dir="/home/hjzd/lzz/LLM_training/data/pretrain/CCI3/data_bin/train",
    #     enc_name=enc_name,
    #     eos_token_id=tokenizer.eot_token,
    #     num_workers=4,  # 👈 根据 CPU 核数调
    # )
