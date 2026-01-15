import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import torch
from src.llm.model.gpt import GPTModel
from src.llm.config.gpt_configs import GPT_CONFIG_355M
from src.llm.data.datasets import create_dataloader_jsonls, create_dataloader_bin, create_dataloader_bins
from src.llm.utils.logger import setup_logger, setup_run_dir
from src.llm.eval.tokenizer import token_ids_to_text, text_to_token_ids
from src.llm.train.loss import calc_loss_batch, calc_loss_loader
from src.llm.eval.generate import generate_text_simple_old, generate_text_token, generate_and_print_sample
import tiktoken

run_dir = setup_run_dir(experiment="pretraining", run_name="gpt124m")
logger = setup_logger(name="train", log_file=run_dir / "train.log")

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

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer, save_epoch):
    train_losses, val_losses, track_tokens_seen = [], [], []     
    tokens_seen, global_step = 0, -1

    logger.info("Start training")
    logger.info(f"num_epochs={num_epochs}, eval_freq={eval_freq}")

    for epoch in range(num_epochs): 
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
            
            if global_step % 100000 == 0:
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
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_355M)

    ## 测试模型是否可以正常推理
    model.eval()
    start_context = "早上出门的时候我才发现忘记带钥匙，只好又回到家里。"
    tokenizer = tiktoken.get_encoding("cl100k_base")
    token_ids = generate_text_token(
        model=model,     
        idx=text_to_token_ids(start_context, tokenizer),     
        max_new_tokens=20,     
        context_size=GPT_CONFIG_355M["context_length"] 
    ) 
    print("test model output:\n", token_ids_to_text(token_ids, tokenizer))




    ## 数据加载器
    train_loader = create_dataloader_bins(     
        "/home/hjzd/lzz/LLM_training/data/CCI3/data_bin/train",     
        batch_size=16,     
        max_length=GPT_CONFIG_355M["context_length"],     
        stride=GPT_CONFIG_355M["context_length"],     
        drop_last=True,     
        shuffle=True,     
        num_workers=0 
    ) 

    val_loader = create_dataloader_bins(     
        "/home/hjzd/lzz/LLM_training/data/CCI3/data_bin/val",     
        batch_size=2,     
        max_length=GPT_CONFIG_355M["context_length"],     
        stride=GPT_CONFIG_355M["context_length"],     
        drop_last=False,     
        shuffle=False,     
        num_workers=0
    ) 


    ## 开始训练
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") 
    model = GPTModel(GPT_CONFIG_355M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)


    ## 这里根据需要加载已有的权重接着训练
    # checkpoint_model = torch.load("/home/hjzd/lzz/LLM_training/logs/pretraining/2026-01-06_17-22-12_gpt124m/checkpoints/model_epoch1_step300000.pt", map_location=device) 
    # checkpoint_optimizer = torch.load("/home/hjzd/lzz/LLM_training/logs/pretraining/2026-01-06_17-22-12_gpt124m/checkpoints/optimizer_epoch1_step300000.pt", map_location=device) 
    # model.load_state_dict(checkpoint_model) 
    # optimizer.load_state_dict(checkpoint_optimizer)


    
    num_epochs = 100 
    train_losses, val_losses, tokens_seen = train_model_simple(     
        model, train_loader, val_loader, optimizer, device,     
        num_epochs=num_epochs, eval_freq=10000, eval_iter=10,     
        start_context="早上出门的时候我才发现忘记带钥匙，只好又回到家里。", 
        tokenizer=tokenizer, save_epoch=1
    ) 


    