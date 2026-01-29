import torch 
import json
import tiktoken
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from functools import partial

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

def format_input_Alpaca(entry):
    instruction_text = (
        # f"Below is an instruction that describes a task. "         
        # f"Write a response that appropriately completes the request."         
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
        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
            )
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
    
    inputs_tensor = torch.stack(inputs_lst).to(device)     
    targets_tensor = torch.stack(targets_lst).to(device)     
    return inputs_tensor, targets_tensor




def collate_fn_manual(batch, pad_token_id=50256):
    """
    手写版 DataLoader collate_fn
    batch: List[{"input_ids": [...], "labels": [...]}]
    
    返回:
        input_ids_tensor: [batch_size, max_len]
        labels_tensor:    [batch_size, max_len]
    
    说明：
    - input_ids 用 pad_token_id 对齐
    - labels 用 -100 对齐
    """
    # 1️⃣ 计算 batch 中最长序列长度
    max_len = max(len(sample["input_ids"]) for sample in batch)

    # 2️⃣ 创建空列表存放对齐后的序列
    input_ids_batch = []
    labels_batch = []

    for sample in batch:
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # 3️⃣ 计算 padding 长度
        pad_len = max_len - len(input_ids)

        # 4️⃣ 手动 pad
        input_ids_padded = input_ids + [pad_token_id] * pad_len
        labels_padded    = labels + [-100] * pad_len

        # 5️⃣ 添加到 batch 列表
        input_ids_batch.append(input_ids_padded)
        labels_batch.append(labels_padded)

    # 6️⃣ 转为 tensor
    input_ids_tensor = torch.tensor(input_ids_batch, dtype=torch.long)
    labels_tensor    = torch.tensor(labels_batch, dtype=torch.long)

    return input_ids_tensor, labels_tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
collate_fn_manual = partial(
    collate_fn_manual,     
    device=device,     
    allowed_max_length=1024 
)

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):         
        self.data = data         
        self.encoded_texts = []        
        for entry in data:
            instruction_plus_input = format_input_Alpaca(entry)             
            response_text = f"\n\n### Response:\n{entry['output']}"             
            full_text = instruction_plus_input + response_text             
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):         
        return self.encoded_texts[index]

    def __len__(self):         
        return len(self.data)

class BelleDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=None): 
        self.encoded_texts = []
        self.max_length = max_length
        with open(jsonl_path, "r", encoding="utf-8") as lines:
            for line in lines:
                entry = json.loads(line)
                instruction_plus_input = format_input_Alpaca(entry)             
                response_text = f"\n\n### Response:\n{entry['output']}"             
                full_text = instruction_plus_input + response_text
                input_ids = tokenizer.encode(full_text)
                if self.max_length is not None:
                    input_ids = input_ids[-self.max_length:]
                self.encoded_texts.append(input_ids)

    def __getitem__(self, index):         
        return self.encoded_texts[index]

    def __len__(self):         
        return len(self.encoded_texts)

class BelleDatasetMasked(Dataset):
    """
    用于 Instruction Tuning 的 Dataset

    核心原则：
    - input_ids: 模型「能看到」的完整上下文（prompt + response）
    - labels:
        * prompt 部分设为 -100（不参与 loss）
        * response 部分是真实 token（参与 loss）
    """

    def __init__(self, jsonl_path, tokenizer, max_length=None):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                sample = self.build_sample(entry)
                self.samples.append(sample)

    def build_sample(self, entry):
        # ===== 1. 构造 prompt（instruction + input + response 前缀）=====
        instruction_plus_input = format_input_Alpaca(entry)
        response_prefix = "\n\n### Response:\n"
        prompt_text = instruction_plus_input + response_prefix

        # ===== 2. tokenize =====
        prompt_ids = self.tokenizer.encode(prompt_text)
        answer_ids = self.tokenizer.encode(entry["output"])

        # ===== 3. 构造模型输入（完整上下文）=====
        input_ids = prompt_ids + answer_ids

        # ===== 4. 构造 labels =====
        # prompt 部分不计算 loss，用 -100 mask
        # response 部分计算 loss
        labels = [-100] * len(prompt_ids) + answer_ids

        # ===== 5. 截断（从右往左，保留 response）=====
        if self.max_length is not None:
            input_ids = input_ids[-self.max_length:]
            labels = labels[-self.max_length:]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)




class MiniMindDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer): 
        self.encoded_texts = []
        with open(jsonl_path, "r", encoding="utf-8") as lines:
            for line in lines:
                entry = json.loads(line)["conversations"]
                full_text = format_input_Phi(entry)                
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
        device=device,
        allowed_max_length=max_length
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                            shuffle=shuffle, drop_last=drop_last, 
                            num_workers=num_workers)
    return dataloader

def create_dataloader_MiniMind(arr_data, tokenizer, pad_token_id, batch_size, max_length=None, shuffle=True, drop_last=True, num_workers=0):
    dataset = MiniMindDataset(arr_data, tokenizer)
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

    tokenizer = tiktoken.get_encoding("cl100k_base")
    # file_path = "/home/hjzd/lzz/LLM_training/data/instruction/simple_instruction/instruction-data.json"
    # with open(file_path, "r", encoding="utf-8") as f:
    #     data = json.load(f)

    # model_input = format_input_Alpaca(data[0]) 
    # desired_response = f"\n\n### Response:\n{data[0]['output']}" 

    # train_portion = int(len(data) * 0.85)
    # test_portion = int(len(data) * 0.1)
    # val_portion = len(data) - train_portion - test_portion
    # train_data = data[:train_portion] 
    # test_data = data[train_portion:train_portion + test_portion] 
    # val_data = data[train_portion + test_portion:]
    # print(type(train_data[0]))
    # print("Training set length:", len(train_data)) 
    # print("Validation set length:", len(val_data)) 
    # print("Test set length:", len(test_data))



    num_workers = 0 
    batch_size = 2
    torch.manual_seed(123)

    # train_dataset = BelleDatasetMasked("/home/hjzd/lzz/LLM_training/data/instruction/belle_data/train_150k.jsonl", tokenizer, max_length=512) 
    # train_loader = DataLoader(     
    #     train_dataset,     
    #     batch_size=batch_size,     
    #     collate_fn=collate_fn_manual,     
    #     shuffle=True,     
    #     drop_last=True,     
    #     num_workers=num_workers 
    # )

    train_loader = create_dataloader_Belle(
        "/home/hjzd/lzz/LLM_training/data/instruction/belle_data/train_150k.jsonl",
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eot_token,
        batch_size=2,
        max_length=1024,
        shuffle=True, 
        drop_last=True, 
        num_workers=0
    )

    # val_dataset = InstructionDataset(val_data, tokenizer) 
    # val_loader = DataLoader(     
    #     val_dataset,     
    #     batch_size=batch_size,     
    #     collate_fn=customized_collate_fn,     
    #     shuffle=False,     
    #     drop_last=False,     
    #     num_workers=num_workers 
    # )

    # test_dataset = InstructionDataset(test_data, tokenizer) 
    # test_loader = DataLoader(     
    #     test_dataset,     
    #     batch_size=batch_size,
    #     collate_fn=customized_collate_fn,     
    #     shuffle=False,     
    #     drop_last=False,     
    #     num_workers=num_workers 
    # )


    print("Train loader:")
    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)
        print("inputs: ", inputs)
        print("\n")
        print("targets: ", targets)





