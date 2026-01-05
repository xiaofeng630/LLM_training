import torch
import tiktoken
import json
import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# 用于单个txt文本文件
class GPTDataset_txt(Dataset): 
    def __init__(self, txt_path, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids  = []
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# 用于单个jsonl文件
class GPTDataset_jsonl(Dataset): 
    def __init__(self, jsonl_path, tokenizer, max_length, stride):

        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.line_offsets = []
        self.max_length = max_length
        self.stride = stride
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            offset = 0
            for line in f:
                obj = json.loads(line)
                tokens = tokenizer.encode(obj.get("text"))
                if(len(tokens) > self.max_length): 
                    self.line_offsets.append(offset)
                offset += len(line.encode("utf-8"))
            

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, idx):
        # 1. 定位到对应行
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            f.seek(self.line_offsets[idx])
            line = f.readline()

        # 2. 解析 json
        obj = json.loads(line)
        text = obj["text"]

        # 3. tokenize
        token_ids = self.tokenizer.encode(text)

        # 4. 如果文本太短，直接跳过（或 pad）
        if len(token_ids) <= self.max_length + 1:
            input_ids = token_ids[:-1]
            target_ids = token_ids[1:]
        else:
            input_ids = token_ids[:self.max_length]
            target_ids = token_ids[1:self.max_length + 1]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )

# 用于一个文件夹下多个jsonl文件
class GPTDataset_jsonls(Dataset):
    def __init__(self, data_path, tokenizer, max_length, stride):
        """
        data_path: str
            - jsonl 文件路径
            - 或包含多个 jsonl 的目录
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        # 统一成文件列表
        if os.path.isdir(data_path):
            self.files = [
                os.path.join(data_path, f)
                for f in os.listdir(data_path)
                if f.endswith(".jsonl")
            ]
        else:
            self.files = [data_path]

        # 核心索引：(file_path, byte_offset)
        self.index = []

        for file_path in self.files:
            with open(file_path, "r", encoding="utf-8") as f:
                offset = 0
                for line in f:
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", "")
                        tokens = tokenizer.encode(text)
                        if len(tokens) > self.max_length:
                            self.index.append((file_path, offset))
                    except Exception:
                        pass
                    offset += len(line.encode("utf-8"))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_path, offset = self.index[idx]

        with open(file_path, "r", encoding="utf-8") as f:
            f.seek(offset)
            line = f.readline()

        obj = json.loads(line)
        text = obj["text"]

        token_ids = self.tokenizer.encode(text)

        # ====== 滑窗切分（真正利用 stride， 充分利用数据集） ======
        if len(token_ids) <= self.max_length + 1:
            input_ids = token_ids[:-1]
            target_ids = token_ids[1:]
        else:
            start = 0
            input_ids = token_ids[start:start + self.max_length]
            target_ids = token_ids[start + 1:start + self.max_length + 1]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )

# 用于单个bin文件
class GPTDataset_bin(Dataset):
    """
    用于单个 .bin token 文件
    bin 文件格式：
        - dtype: uint32
        - 一整条连续 token 流
    """

    def __init__(self, bin_path, max_length, stride):
        self.bin_path = Path(bin_path)
        self.max_length = max_length
        self.stride = stride

        # 使用 memmap，不把数据读入内存
        self.tokens = np.memmap(
            self.bin_path,
            dtype=np.uint32,
            mode="r"
        )

        self.num_tokens = len(self.tokens)

        # 可切分出的样本数
        self.num_samples = (
            self.num_tokens - (max_length + 1)
        ) // stride

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.max_length + 1

        chunk = self.tokens[start:end]

        input_ids = torch.from_numpy(
            chunk[:-1].astype(np.int64)
        )
        target_ids = torch.from_numpy(
            chunk[1:].astype(np.int64)
        )

        return input_ids, target_ids

# 用于一个文件夹下多个bin文件
class GPTDataset_bins(Dataset):
    """
    用于一个目录下的多个 .bin 文件
    """

    def __init__(self, bin_dir, max_length, stride):
        self.bin_dir = Path(bin_dir)
        self.max_length = max_length
        self.stride = stride

        self.bin_files = sorted(self.bin_dir.glob("*.bin"))
        assert len(self.bin_files) > 0, "No .bin files found"

        self.memmaps = []
        self.sample_counts = []

        for path in self.bin_files:
            tokens = np.memmap(path, dtype=np.uint32, mode="r")
            n_tokens = len(tokens)
            n_samples = (n_tokens - (max_length + 1)) // stride

            if n_samples <= 0:
                continue

            self.memmaps.append(tokens)
            self.sample_counts.append(n_samples)

        # 前缀和，用于全局 idx → 文件 idx
        self.cum_samples = np.cumsum(self.sample_counts)

    def __len__(self):
        return int(self.cum_samples[-1])

    def __getitem__(self, idx):
        # 找到属于哪个 bin
        file_idx = np.searchsorted(self.cum_samples, idx, side="right")
        prev_samples = 0 if file_idx == 0 else self.cum_samples[file_idx - 1]
        local_idx = idx - prev_samples

        tokens = self.memmaps[file_idx]

        start = local_idx * self.stride
        end = start + self.max_length + 1

        chunk = tokens[start:end]

        input_ids = torch.from_numpy(
            chunk[:-1].astype(np.int64)
        )
        target_ids = torch.from_numpy(
            chunk[1:].astype(np.int64)
        )

        return input_ids, target_ids


def create_dataloader_txt(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    dataset = GPTDataset_txt(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=shuffle, drop_last=drop_last, 
                            num_workers=num_workers)
    return dataloader

def create_dataloader_jsonl(jsonl_path, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    dataset = GPTDataset_jsonl(jsonl_path, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=shuffle, drop_last=drop_last, 
                            num_workers=num_workers)
    return dataloader

def create_dataloader_jsonls(jsonls_path, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    dataset = GPTDataset_jsonls(jsonls_path, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=shuffle, drop_last=drop_last, 
                            num_workers=num_workers)
    return dataloader

def create_dataloader_bin(bin_path, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDataset_bin(bin_path, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=shuffle, drop_last=drop_last, 
                            num_workers=num_workers)
    return dataloader

def create_dataloader_bins(bins_path, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDataset_bins(bins_path, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=shuffle, drop_last=drop_last, 
                            num_workers=num_workers)
    return dataloader