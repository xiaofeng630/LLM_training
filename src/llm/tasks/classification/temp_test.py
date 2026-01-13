import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from src.llm.tasks.classification.eval import classify_review
from src.llm.tasks.classification.datasets import SpamDataset, create_dataloader_Spam
from src.llm.utils.gpt_download import download_and_load_gpt2
from src.llm.model.gpt import GPTModel
import tiktoken
import torch
import numpy as np

def assign(left, right):
    if left.shape != right.shape:         
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "                           
        "Right: {right.shape}"         
        )     
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])     
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])):         
        q_w, k_w, v_w = np.split(             
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)         
        gpt.trf_blocks[b].att.W_query.weight = assign(             
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)         
        gpt.trf_blocks[b].att.W_key.weight = assign(             
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)         
        gpt.trf_blocks[b].att.W_value.weight = assign(             
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(             
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)         
        gpt.trf_blocks[b].att.W_query.bias = assign(             
            gpt.trf_blocks[b].att.W_query.bias, q_b)         
        gpt.trf_blocks[b].att.W_key.bias = assign(             
            gpt.trf_blocks[b].att.W_key.bias, k_b)         
        gpt.trf_blocks[b].att.W_value.bias = assign(             
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign( 
            gpt.trf_blocks[b].att.out_proj.weight,             
            params["blocks"][b]["attn"]["c_proj"]["w"].T)          
        gpt.trf_blocks[b].att.out_proj.bias = assign(             
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"]) 
        gpt.trf_blocks[b].ff.layers[0].weight = assign(             
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)

        gpt.trf_blocks[b].ff.layers[0].bias = assign(             
            gpt.trf_blocks[b].ff.layers[0].bias,             
            params["blocks"][b]["mlp"]["c_fc"]["b"])         
        gpt.trf_blocks[b].ff.layers[2].weight = assign(             
            gpt.trf_blocks[b].ff.layers[2].weight,             
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        
        gpt.trf_blocks[b].ff.layers[2].bias = assign(             
            gpt.trf_blocks[b].ff.layers[2].bias,             
            params["blocks"][b]["mlp"]["c_proj"]["b"])         
        gpt.trf_blocks[b].norm1.scale = assign(             
            gpt.trf_blocks[b].norm1.scale,             
            params["blocks"][b]["ln_1"]["g"])         
        gpt.trf_blocks[b].norm1.shift = assign(             
            gpt.trf_blocks[b].norm1.shift,             
            params["blocks"][b]["ln_1"]["b"])         
        gpt.trf_blocks[b].norm2.scale = assign(             
            gpt.trf_blocks[b].norm2.scale,             
            params["blocks"][b]["ln_2"]["g"])         
        gpt.trf_blocks[b].norm2.shift = assign(             
            gpt.trf_blocks[b].norm2.shift,             
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"]) 
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"]) 
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


if __name__ == "__main__":
    CHOOSE_MODEL = "gpt2-small (124M)" 
    BASE_CONFIG = {
        "vocab_size": 50257,     
        "context_length": 1024,
        "drop_rate": 0.0,     
        "qkv_bias": True 
    }
    model_configs = {     
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},     
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},     
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},     
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}, 
    } 
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    text_1 = ("You are a winner you have been specially"     
    " selected to receive $1000 cash or a $2000 award." 
    )
    train_loader = create_dataloader_Spam(
        "/home/hjzd/lzz/LLM_training/data/classification/sms_spam_collection/train.csv",
        batch_size=8,
        max_length=None,
        shuffle=True, 
        drop_last=True, 
        num_workers=0
    )
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="/home/hjzd/lzz/LLM_training/gpt2") 
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.to(device)

    torch.manual_seed(123) 
    num_classes = 2 
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
    model.to(device)

    model_state_dict = torch.load("/home/hjzd/lzz/LLM_training/logs/fine_tuning_classify/2026-01-12_10-33-41_gpt124m/checkpoints/model_epoch10.pt") 
    model.load_state_dict(model_state_dict)

    print(train_loader.next())
    print(classify_review(text_1, model, tokenizer, device, max_length=train_loader.dataset.max_length))
