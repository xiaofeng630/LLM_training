import torch 
import json
import tiktoken
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from functools import partial


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

def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):     
    batch_max_length = max(len(item)+1 for item in batch)     
    inputs_lst = []
    for item in batch:         
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        inputs = torch.tensor(padded[:-1])         
        inputs_lst.append(inputs)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor

def custom_collate_draft_2(batch, pad_token_id=50256, device="cpu" ):     
    batch_max_length = max(len(item)+1 for item in batch)     
    inputs_lst, targets_lst = [], []
    for item in batch:         
        new_item = item.copy()         
        new_item += [pad_token_id]         
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        inputs = torch.tensor(padded[:-1])         
        targets = torch.tensor(padded[1:])         
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor = torch.stack(inputs_lst).to(device)     
    targets_tensor = torch.stack(targets_lst).to(device)     
    return inputs_tensor, targets_tensor

def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)     
    inputs_lst, targets_lst = [], []
    for item in batch:         
        new_item = item.copy()         
        new_item += [pad_token_id]
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))          
        inputs = torch.tensor(padded[:-1])         
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id         
        indices = torch.nonzero(mask).squeeze() 
        if indices.numel() > 1:            
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:             
            inputs = inputs[:allowed_max_length]             
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)         
        targets_lst.append(targets)
    
    inputs_tensor = torch.stack(inputs_lst) 
    targets_tensor = torch.stack(targets_lst)  
    return inputs_tensor, targets_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
customized_collate_fn = partial(     
    custom_collate_fn,     
    device=device,     
    allowed_max_length=1024 
)

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):         
        self.data = data         
        self.encoded_texts = []        
        for entry in data:
            instruction_plus_input = format_input(entry)             
            response_text = f"\n\n### Response:\n{entry['output']}"             
            full_text = instruction_plus_input + response_text             
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):         
        return self.encoded_texts[index]

    def __len__(self):         
        return len(self.data)

class BelleDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer): 
        self.encoded_texts = []
        with open(jsonl_path, "r", encoding="utf-8") as lines:
            for line in lines:
                entry = json.loads(line)
                instruction_plus_input = format_input(entry)             
                response_text = f"\n\n### Response:\n{entry['output']}"             
                full_text = instruction_plus_input + response_text       
                self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):         
        return self.encoded_texts[index]

    def __len__(self):         
        return len(self.encoded_texts)


def create_dataloader_Instruction(arr_data, tokenizer, pad_token_id, batch_size, max_length=None, shuffle=True, drop_last=True, num_workers=0):
    dataset = InstructionDataset(arr_data, tokenizer)
    collate_fn = partial(
        custom_collate_fn,
        pad_token_id=pad_token_id,
        allowed_max_length=max_length,
        device=device,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                            shuffle=shuffle, drop_last=drop_last, 
                            num_workers=num_workers)
    return dataloader

def create_dataloader_Belle(arr_data, tokenizer, pad_token_id, batch_size, max_length=None, shuffle=True, drop_last=True, num_workers=0):
    dataset = BelleDataset(arr_data, tokenizer)
    collate_fn = partial(
        custom_collate_fn,
        pad_token_id=pad_token_id,
        allowed_max_length=max_length,
        device=device,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                            shuffle=shuffle, drop_last=drop_last, 
                            num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    # inputs_1 = [0, 1, 2, 3, 4] 
    # inputs_2 = [5, 6] 
    # inputs_3 = [7, 8, 9] 
    # batch = (inputs_1, inputs_2, inputs_3) 
    # # print(custom_collate_draft_2(batch))
    # inputs, targets = custom_collate_fn(batch) 
    # print(inputs) 
    # print(targets)

    tokenizer = tiktoken.get_encoding("gpt2")
    file_path = "/home/hjzd/lzz/LLM_training/data/instruction/simple_instruction/instruction-data.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_input = format_input(data[0]) 
    desired_response = f"\n\n### Response:\n{data[0]['output']}" 

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion
    train_data = data[:train_portion] 
    test_data = data[train_portion:train_portion + test_portion] 
    val_data = data[train_portion + test_portion:]
    # print(type(train_data[0]))
    # print("Training set length:", len(train_data)) 
    # print("Validation set length:", len(val_data)) 
    # print("Test set length:", len(test_data))



    num_workers = 0 
    batch_size = 8
    torch.manual_seed(123)

    train_dataset = BelleDataset("/home/hjzd/lzz/LLM_training/data/instruction/belle_data/val.jsonl", tokenizer) 
    train_loader = DataLoader(     
        train_dataset,     
        batch_size=batch_size,     
        collate_fn=customized_collate_fn,     
        shuffle=True,     
        drop_last=True,     
        num_workers=num_workers 
    )

    val_dataset = InstructionDataset(val_data, tokenizer) 
    val_loader = DataLoader(     
        val_dataset,     
        batch_size=batch_size,     
        collate_fn=customized_collate_fn,     
        shuffle=False,     
        drop_last=False,     
        num_workers=num_workers 
    )

    test_dataset = InstructionDataset(test_data, tokenizer) 
    test_loader = DataLoader(     
        test_dataset,     
        batch_size=batch_size,
        collate_fn=customized_collate_fn,     
        shuffle=False,     
        drop_last=False,     
        num_workers=num_workers 
    )
    
    print(train_loader[0])

    # print("Train loader:")
    # for inputs, targets in train_loader:
    #     print(inputs.shape, targets.shape)





