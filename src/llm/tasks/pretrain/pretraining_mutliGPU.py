import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import torch
from src.llm.model.gpt import GPTModel
from src.llm.data.datasets import create_dataloader_jsonls, create_dataloader_bin, create_dataloader_bins
from src.llm.utils.logger import setup_logger, setup_run_dir
import tiktoken

# 多卡训练新增
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler



def text_to_token_ids(text, tokenizer):     
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})     
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) 
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):     
    flat = token_ids.squeeze(0)   
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch, model, device):     
    input_batch = input_batch.to(device)     
    target_batch = target_batch.to(device)
    logits = model(input_batch) 
    loss = torch.nn.functional.cross_entropy( 
        logits.flatten(0, 1), target_batch.flatten()     
    )
    return loss 

def calc_loss_loader(data_loader, model, device, num_batches=None):     
    total_loss = 0     
    if len(data_loader) == 0:         
        return float("nan")      
    elif num_batches is None: 
        num_batches = len(data_loader) 
    else:         
        num_batches = min(num_batches, len(data_loader))     
    for i, (input_batch, target_batch) in enumerate(data_loader):         
        if i < num_batches:             
            loss = calc_loss_batch(                 
                input_batch, target_batch, model, device             
            )              
            total_loss += loss.item()          
        else: 
            break 
    return total_loss / num_batches

def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.2):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        
        # 1. 应用重复惩罚
        if repetition_penalty != 1.0:
            for token_id in set(idx[0].tolist()):
                logits[0, token_id] /= repetition_penalty
        
        # 2. 应用温度
        logits = logits / temperature
        
        # 3. Top-k过滤
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')
        
        # 4. Top-p (nucleus) 过滤
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除累积概率超过top_p的token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('Inf')
        
        # 5. 采样而非贪婪选择
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probas, num_samples=1)
        
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

def generate_and_print_sample(model, tokenizer, device, start_context):     
    model.eval()     
    raw_model = model.module if isinstance(model, DDP) else model
    context_size = raw_model.pos_emb.weight.shape[0]     
    encoded = text_to_token_ids(start_context, tokenizer).to(device)     
    with torch.no_grad():         
        token_ids = generate_text_simple(             
            model=model, idx=encoded,             
            max_new_tokens=50, context_size=context_size         
        )     
    decoded_text = token_ids_to_text(token_ids, tokenizer) 
    decoded_text = decoded_text.replace("\n", " ")
    model.train()
    return decoded_text

def evaluate_model(model, train_loader, val_loader, device, eval_iter):     
    model.eval()     
    with torch.no_grad():         
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter         
        )         
        val_loss = calc_loss_loader(             
            val_loader, model, device, num_batches=eval_iter         
        )     
        model.train()     
        return train_loss, val_loss

def train_model_simple(model, train_loader, train_sampler, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer, save_epoch):
    train_losses, val_losses, track_tokens_seen = [], [], []     
    tokens_seen, global_step = 0, -1

    if is_main_process:
        logger.info("Start training")
        logger.info(f"num_epochs={num_epochs}, eval_freq={eval_freq}")

    for epoch in range(num_epochs): 
        train_sampler.set_epoch(epoch)
        model.train()
        
        for input_batch, target_batch in train_loader:      
            optimizer.zero_grad()             
            loss = calc_loss_batch(
                input_batch, target_batch, model, device 
            )             
            loss.backward()
            optimizer.step()             
            tokens_seen += input_batch.numel() 
            global_step += 1
            
            if global_step % eval_freq == 0: 
                train_loss, val_loss = evaluate_model(                     
                    model, train_loader, val_loader, device, eval_iter
                )                 
                train_losses.append(train_loss)                 
                val_losses.append(val_loss)                 
                track_tokens_seen.append(tokens_seen)  
                lr = optimizer.param_groups[0]["lr"]

                
                if is_main_process:
                    sample_text = generate_and_print_sample( 
                        model, tokenizer, device, start_context         
                    )

                    logger.info(
                        f"epoch={epoch+1} step={global_step:06d} | "
                        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                        f"tokens_seen={tokens_seen/1e6:.2f}M | lr={lr:.2e}\n"
                        f"{sample_text}"
                    )
                
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "                       
                        f"Train loss {train_loss:.3f}, "                       
                        f"Val loss {val_loss:.3f}\n"
                        f"{sample_text}"               
                        )
            
            if global_step % 10000 == 0 and is_main_process:
                ckpt_path = run_dir / "checkpoints" / f"model_epoch{epoch + 1}_step{global_step}.pt"
                optimizer_path = run_dir / "checkpoints" / f"optimizer_epoch{epoch + 1}_step{global_step}.pt"
                torch.save(model.state_dict(), ckpt_path)
                torch.save(optimizer.state_dict(), optimizer_path)
                logger.info(f"epoch{epoch + 1}_step{global_step}, Weigths saved successfully")
        
        if (epoch + 1) % save_epoch == 0 and is_main_process:
            ckpt_path = run_dir / "checkpoints" / f"model_epoch{epoch + 1}.pt"
            optimizer_path = run_dir / "checkpoints" / f"optimizer_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            logger.info(f"epoch{epoch + 1}, Weigths saved successfully")

        if is_main_process:
            sample_text = generate_and_print_sample( 
                model, tokenizer, device, start_context         
            )     
    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":

    def setup_ddp():
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = dist.get_rank() == 0

    if is_main_process:
        run_dir = setup_run_dir(experiment="pretraining", run_name="gpt124m")
        logger = setup_logger(name="train", log_file=run_dir / "train.log")

    GPT_CONFIG_124M = { "vocab_size": 100276, 
                        "context_length": 256,     
                        "emb_dim": 768,     
                        "n_heads": 12,     
                        "n_layers": 12, 
                        "drop_rate": 0.1,
                        "qkv_bias": False 
                    } 
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    ## 测试模型是否可以正常推理
    model.eval()
    start_context = "早上出门的时候我才发现忘记带钥匙，只好又回到家里。"
    tokenizer = tiktoken.get_encoding("cl100k_base")
    token_ids = generate_text_simple(
        model=model,     
        idx=text_to_token_ids(start_context, tokenizer),     
        max_new_tokens=20,     
        context_size=GPT_CONFIG_124M["context_length"] 
    ) 
    print("test model output:\n", token_ids_to_text(token_ids, tokenizer))

    ## 数据加载器
    train_loader = create_dataloader_bins(     
        "/home/hjzd/lzz/LLM_training/data/CCI3/data_bin/train",     
        batch_size=16,     
        max_length=GPT_CONFIG_124M["context_length"],     
        stride=GPT_CONFIG_124M["context_length"],     
        drop_last=True,     
        shuffle=True,     
        num_workers=0 
    ) 
    train_dataset = train_loader.dataset
    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,          # 每卡 batch
        sampler=train_sampler,
        num_workers=0,
        drop_last=True
    )

    val_loader = create_dataloader_bins(     
        "/home/hjzd/lzz/LLM_training/data/CCI3/data_bin/val",     
        batch_size=2,     
        max_length=GPT_CONFIG_124M["context_length"],     
        stride=GPT_CONFIG_124M["context_length"],     
        drop_last=False,     
        shuffle=False,     
        num_workers=0
    ) 


    ## 开始训练
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = GPTModel(GPT_CONFIG_124M).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW( 
        model.parameters(), 
        lr=0.0004, 
        weight_decay=0.1 
    )
    num_epochs = 100 
    train_losses, val_losses, tokens_seen = train_model_simple(     
        model, train_loader, train_sampler, val_loader, optimizer, device,     
        num_epochs=num_epochs, eval_freq=100, eval_iter=15,     
        start_context="早上出门的时候我才发现忘记带钥匙，只好又回到家里。", 
        tokenizer=tokenizer, save_epoch=1
    ) 
    dist.destroy_process_group()


    