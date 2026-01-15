import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
import pandas as pd
import torch
import tiktoken
from torch._subclasses.fake_tensor import _device_handler
from src.llm.model.gpt import GPTModel
from src.llm.eval.generate import generate_text_token
from src.llm.eval.tokenizer import text_to_token_ids, token_ids_to_text
from src.llm.tasks.classification.datasets import SpamDataset, create_dataloader_Spam, create_dataloader_ChnSentiCorp

## 调用模型进行分类（根据padding后的最后一个token来计算结果）
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=0):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]
    input_ids = input_ids[:min(max_length, supported_context_length)]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
        print("logits: ", logits)
        predicted_label = torch.argmax(logits, dim=-1).item()
    return 1 if predicted_label == 1 else 0

## 用于加载模型权重（注意要跟架构一致）
def load_weights(model, ckpt_path):
    ckpt_model = torch.load(ckpt_path)
    model.load_state_dict(ckpt_model)
    return model


if __name__ == "__main__":
    ckpt_path = "/home/hjzd/lzz/LLM_training/logs/fine_tuning_classify/2026-01-14_17-44-08_gpt124m/checkpoints/model_epoch10.pt"
    BASE_CONFIG = {
        "vocab_size": 100276, 
        "context_length": 256,     
        "emb_dim": 768,     
        "n_heads": 12,     
        "n_layers": 12, 
        "drop_rate": 0.1,
        "qkv_bias": False 
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("cl100k_base")

    model = GPTModel(BASE_CONFIG)
    model.eval()
    model.to(device)

    torch.manual_seed(123) 
    num_classes = 2 
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
    load_weights(model, ckpt_path)
    model.to(device)

    # train_loader = create_dataloader_ChnSentiCorp(
    #     "/home/hjzd/lzz/LLM_training/data/classification/ChnSentiCorp/test.parquet",
    #     tokenizer=tokenizer,
    #     batch_size=1,
    #     max_length=256,
    #     shuffle=True, 
    #     drop_last=True, 
    #     num_workers=0
    # )

    # train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=None)
    # print(train_accuracy)



    df = pd.read_parquet(
        "/home/hjzd/lzz/LLM_training/data/classification/ChnSentiCorp/test.parquet"
    )

    correct = 0
    total = len(df)

    for _, row in df.iterrows():
        text = row["text"]
        label = int(row["label"])

        ans = classify_review(
            text,
            model,
            tokenizer,
            device,
            max_length=256,
            pad_token_id=tokenizer.eot_token
        )

        print(f"label: {label}, ans: {ans}")

        if ans == label:
            correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

    
