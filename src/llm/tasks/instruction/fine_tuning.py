import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
import tiktoken
import torch
import json
import time
import numpy as np
from src.llm.config.gpt_configs import GPT_CONFIG_355M
from src.llm.eval.tokenizer import text_to_token_ids, token_ids_to_text
from src.llm.eval.generate import generate_text_token
from src.llm.utils.logger import setup_logger, setup_run_dir
from src.llm.model.gpt import GPTModel
from src.llm.utils.loss_tracker import LossTracker
from src.llm.utils.gpt_download import download_and_load_gpt2
from src.llm.tasks.instruction.datasets import InstructionDataset, create_dataloader_Belle, create_dataloader_Instruction
from src.llm.tasks.instruction.loss import calc_loss_batch, calc_loss_loader
from src.llm.eval.generate import generate_text_simple_old, generate_text_token, generate_and_print_sample

run_dir = setup_run_dir(experiment="instruction", run_name="gpt355m")
logger = setup_logger(name="train", log_file=run_dir / "train.log")
loss_tracker = LossTracker(run_dir / "samples")

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

def train_model_simple(model, train_loader, val_loader, eval_train_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer, save_epoch):
    train_losses, val_losses, track_tokens_seen = [], [], []     
    tokens_seen, global_step = 0, -1

    logger.info("Start training")
    logger.info(f"num_epochs={num_epochs}, eval_freq={eval_freq}")

    for epoch in range(num_epochs): 
        model.train()
        
        for input_batch, target_batch in train_loader:      
            optimizer.zero_grad()        
            with torch.autocast("cuda", dtype=torch.bfloat16):     
                loss = calc_loss_batch(
                    input_batch, target_batch, model, device 
                )
            loss.backward()
            optimizer.step()             
            tokens_seen += input_batch.numel() 
            global_step += 1
            
            if global_step % eval_freq == 0: 
                train_loss, val_loss = evaluate_model(                     
                    model, eval_train_loader, val_loader, device, eval_iter
                )
                loss_tracker.update_train(global_step, train_loss)
                loss_tracker.update_val(global_step, val_loss)
                train_losses.append(train_loss)                 
                val_losses.append(val_loss)                 
                track_tokens_seen.append(tokens_seen)  
                lr = optimizer.param_groups[0]["lr"]

                if global_step % (eval_freq * 10) == 0:
                    sample_text = generate_and_print_sample( 
                        model, tokenizer, device, start_context         
                    )
                    logger.info(f"sample_text: {sample_text}\n")
                    print(f"sample_text: {sample_text}\n")

                logger.info(
                    f"epoch={epoch+1} step={global_step:06d} | "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                    f"tokens_seen={tokens_seen/1e6:.2f}M | lr={lr:.2e}"
                )
                loss_tracker.plot("loss.png")

                
                print(f"Ep {epoch+1} (Step {global_step:06d}): "                       
                      f"Train loss {train_loss:.3f}, "                       
                      f"Val loss {val_loss:.3f}"            
                     )
            
            if global_step % 25000 == 0:
                ckpt_path = run_dir / "checkpoints" / f"model_epoch{epoch + 1}_step{global_step}.pt"
                optimizer_path = run_dir / "checkpoints" / f"optimizer_epoch{epoch + 1}_step{global_step}.pt"
                torch.save(model.state_dict(), ckpt_path)
                torch.save(optimizer.state_dict(), optimizer_path)
                logger.info(f"epoch{epoch + 1}_step{global_step}, Weigths saved successfully")
        
        if (epoch + 1) % save_epoch == 0:
            ckpt_path = run_dir / "checkpoints" / f"model_epoch{epoch + 1}.pt"
            optimizer_path = run_dir / "checkpoints" / f"optimizer_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            logger.info(f"epoch{epoch + 1}, Weigths saved successfully")

        sample_text = generate_and_print_sample( 
            model, tokenizer, device, start_context         
        )     
    return train_losses, val_losses, track_tokens_seen

if __name__ == "__main__":
    ## 配置模型参数
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    GPT_CONFIG_355M["vocab_size"] = tokenizer.max_token_value
    

    ## 初始化模型
    model = GPTModel(GPT_CONFIG_355M)
    model.eval()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ## 加载预训练模型权重
    ckpt_model = torch.load("/home/hjzd/lzz/LLM_training/logs/pretraining/2026-01-16_17-24-23_gpt355m/checkpoints/model_epoch2_step950000.pt", map_location=device)
    model.load_state_dict(ckpt_model)
    

    ## 测试是否正确加载了权重
    torch.manual_seed(123) 
    token_ids = generate_text_token(  
        model=model,     
        idx=text_to_token_ids("任何努力都将使你变的", tokenizer).to(device),     
        max_new_tokens=50,     
        context_size=GPT_CONFIG_355M["context_length"],     
        top_k=50,     
        temperature=1.0
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    ## 加载数据集
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

    val_loader = create_dataloader_Belle(
        "/home/hjzd/lzz/LLM_training/data/instruction/belle_data/val.jsonl",
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eot_token,
        batch_size=2,
        max_length=1024,
        shuffle=True, 
        drop_last=True, 
        num_workers=0
    )

    torch.manual_seed(123)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)     
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss) 


    file_path = "/home/hjzd/lzz/LLM_training/data/instruction/simple_instruction/instruction-data.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    model_input = format_input(data[0]) 


    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1) 
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(     
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        eval_train_loader=train_loader,
        optimizer=optimizer, 
        device=device,
        num_epochs=num_epochs, 
        eval_freq=1000, 
        eval_iter=15,
        tokenizer=tokenizer,
        start_context=model_input, 
        save_epoch=1
        )
    end_time = time.time() 
    execution_time_minutes = (end_time - start_time) / 60 
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
