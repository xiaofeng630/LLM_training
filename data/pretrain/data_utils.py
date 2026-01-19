import os
import numpy as np
from typing import Union

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

if __name__ == "__main__":
    total_tokens = count_tokens_in_bin(
    "/home/hjzd/lzz/LLM_training/data/pretrain/CCI3/data_bin/train",
    dtype="int32"
)
