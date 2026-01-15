import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd 
import tiktoken

## 书中用于进行垃圾邮件分类的Datasets类
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, pad_token_id, max_length=None): 
        self.data = pd.read_csv(csv_file)  ## 这里读取所有的数据
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]] ## 将每条邮件的具体内容放到一个数组里面

        ## 进行裁切和填充，要保证每一条训练的数据长度都是一致的（也就是长度要对齐）。但是这里不对齐又会怎样呢？？？？
        if max_length is None:             
            self.max_length = self._longest_encoded_length()         
        else:             
            self.max_length = max_length
            self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
            
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))             
            for encoded_text in self.encoded_texts         
        ]
    
    ## 根据索引来获取分词后的训练数据，批量获取由dataloader来实现
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]         
        label = self.data.iloc[index]["Label"]         
        return (             
            torch.tensor(encoded, dtype=torch.long),             
            torch.tensor(label, dtype=torch.long)         
        )
    
    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:             
            encoded_length = len(encoded_text)             
            if encoded_length > max_length:                 
                max_length = encoded_length         
        return max_length

## 用于简单的情感二分类的Datasets类
## 数据集为modelscope上的AiNiklaus/ChnSentiCorp
class ChnSentiCorpDataset(Dataset):
    def __init__(self, parquet_file, tokenizer, pad_token_id, max_length=None):
        self.df = pd.read_parquet(parquet_file)

        assert "text" in self.df.columns
        assert "label" in self.df.columns

        # 1. tokenize 所有文本
        self.encoded_texts = [
            tokenizer.encode(text)
            for text in self.df["text"]
        ]

        # 2. 决定 max_length
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                enc[:self.max_length] for enc in self.encoded_texts
            ]

        # 3. padding 到 max_length
        self.encoded_texts = [
            enc + [pad_token_id] * (self.max_length - len(enc))
            for enc in self.encoded_texts
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_ids = torch.tensor(
            self.encoded_texts[idx], dtype=torch.long
        )
        label = torch.tensor(
            int(self.df.iloc[idx]["label"]), dtype=torch.long
        )

        return input_ids, label

    def _longest_encoded_length(self):
        max_len = 0
        for enc in self.encoded_texts:
            if len(enc) > max_len:
                max_len = len(enc)
        return max_len

def create_dataloader_Spam(csv_file_path, tokenizer, pad_token_id, batch_size, max_length=None, shuffle=True, drop_last=True, num_workers=0):
    dataset = SpamDataset(csv_file_path, tokenizer, pad_token_id, max_length)
    max_length = dataset.max_length
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=shuffle, drop_last=drop_last, 
                            num_workers=num_workers)
    return dataloader

def create_dataloader_ChnSentiCorp(parquet_file_path, tokenizer, pad_token_id, batch_size, max_length=None, shuffle=True, drop_last=True, num_workers=0):
    dataset = ChnSentiCorpDataset(parquet_file_path, tokenizer, pad_token_id, max_length)
    max_length = dataset.max_length
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=shuffle, drop_last=drop_last, 
                            num_workers=num_workers)
    return dataloader



