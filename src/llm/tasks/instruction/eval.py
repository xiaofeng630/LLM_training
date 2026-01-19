import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
import pandas as pd
import torch
import tiktoken
import json
from torch._subclasses.fake_tensor import _device_handler
from src.llm.model.gpt import GPTModel
from src.llm.eval.generate import generate_text_token
from src.llm.eval.tokenizer import text_to_token_ids, token_ids_to_text
from src.llm.tasks.classification.datasets import SpamDataset, create_dataloader_Spam, create_dataloader_ChnSentiCorp

## 用于加载模型权重（注意要跟架构一致）
def load_weights(model, ckpt_path):
    ckpt_model = torch.load(ckpt_path)
    model.load_state_dict(ckpt_model)
    return model

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

if __name__ == "__main__":
    ckpt_path = "/home/hjzd/lzz/LLM_training/logs/instruction/2026-01-19_10-47-06_gpt355m/checkpoints/model_epoch3.pt"
    BASE_CONFIG = {
        "vocab_size": 50257, 
        "context_length": 1024,     
        "emb_dim": 1024,     
        "n_heads": 16,     
        "n_layers": 24, 
        "drop_rate": 0.0,
        "qkv_bias": True
    }
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTModel(BASE_CONFIG)
    model.eval()
    model.to(device)

    torch.manual_seed(123) 
    load_weights(model, ckpt_path)
    model.to(device)

    ## 加载数据集
    file_path = "/home/hjzd/lzz/LLM_training/data/instruction/simple_instruction/instruction-data.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_input = format_input(data[0]) 
    desired_response = f"\n\n### Response:\n{data[0]['output']}" 
    print("model_input: ", model_input)

    token_ids = generate_text_token(     
        model=model,     
        idx=text_to_token_ids(model_input, tokenizer).to(device),     
        max_new_tokens=23,     
        context_size=BASE_CONFIG["context_length"] 
    ) 
    print("ans: ", token_ids_to_text(token_ids, tokenizer))

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion
    train_data = data[:train_portion] 
    test_data = data[train_portion:train_portion + test_portion] 
    val_data = data[train_portion + test_portion:]

    for entry in test_data[:3]:     
        input_text = format_input(entry)     
        token_ids = generate_text_token(
            model=model,         
            idx=text_to_token_ids(input_text, tokenizer).to(device),         
            max_new_tokens=256,         
            context_size=BASE_CONFIG["context_length"],         
            eos_id=50256     
        )     
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (         
            generated_text[len(input_text):]         
            .replace("### Response:", "")         
            .strip()     
        )     
        print(input_text)     
        print(f"\nCorrect response:\n>> {entry['output']}")     
        print(f"\nModel response:\n>> {response_text.strip()}")     
        print("-------------------------------------")


    
