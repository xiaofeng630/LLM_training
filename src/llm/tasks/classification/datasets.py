import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd 
import tiktoken


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256): 
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        if max_length is None:             
            self.max_length = self._longest_encoded_length()         
        else:             
            self.max_length = max_length
            self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
        
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))             
            for encoded_text in self.encoded_texts         
        ]
    
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

def create_dataloader_Spam(csv_file_path, batch_size, max_length=None, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = SpamDataset(csv_file_path, tokenizer, max_length)
    max_length = dataset.max_length
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=shuffle, drop_last=drop_last, 
                            num_workers=num_workers)
    return dataloader





