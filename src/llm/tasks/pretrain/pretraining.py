import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
import torch
from src.llm.model.gpt import GPTModel
from src.llm.utils.loss_tracker import LossTracker
from src.llm.config.gpt_configs import GPT_CONFIG_355M
from src.llm.tasks.pretrain.datasets import create_dataloader_jsonls, create_dataloader_bin, create_dataloader_bins
from src.llm.utils.logger import setup_logger, setup_run_dir
from src.llm.eval.tokenizer import token_ids_to_text, text_to_token_ids
from src.llm.tasks.pretrain.loss import calc_loss_batch, calc_loss_loader
from src.llm.eval.generate import generate_text_simple_old, generate_text_token, generate_and_print_sample
import tiktoken

## 设置日志模块并记录日志
run_dir = setup_run_dir(experiment="pretraining", run_name="gpt355m")
logger = setup_logger(name="train", log_file=run_dir / "train.log")

## loss记录模块，用于实时画loss曲线图
loss_tracker = LossTracker(run_dir / "samples")

## 评估函数，用于训练中计算train_loss和val_loss
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

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer, save_epoch, save_step):
    ## 初始化变量，两个loss数组。track_tokens_seen用于记录每次评估时模型都学到了多少的token（单位是M）
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
            
            ## 每隔eval_freq个step进行一次评估
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(                     
                    model, train_loader, val_loader, device, eval_iter
                )
                loss_tracker.update_train(global_step, train_loss)
                loss_tracker.update_val(global_step, val_loss)
                train_losses.append(train_loss)                 
                val_losses.append(val_loss)                 
                track_tokens_seen.append(tokens_seen)  
                lr = optimizer.param_groups[0]["lr"]
                
                ## 每隔eval_freq * 10个step就进行一次文本生成
                if global_step % (eval_freq * 10) == 0:
                    sample_text = generate_and_print_sample( 
                        model, tokenizer, device, start_context         
                    )
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
            

            ## 每隔save_step个step就保存一次模型和优化器权重
            if global_step % save_step == 0:
                ckpt_path = run_dir / "checkpoints" / f"model_epoch{epoch + 1}_step{global_step}.pt"
                optimizer_path = run_dir / "checkpoints" / f"optimizer_epoch{epoch + 1}_step{global_step}.pt"
                torch.save(model.state_dict(), ckpt_path)
                torch.save(optimizer.state_dict(), optimizer_path)
                logger.info(f"epoch{epoch + 1}_step{global_step}, Weigths saved successfully")
        
        ## 每个epoch保存一个权重
        if (epoch + 1) % save_epoch == 0:
            ckpt_path = run_dir / "checkpoints" / f"model_epoch{epoch + 1}.pt"
            optimizer_path = run_dir / "checkpoints" / f"optimizer_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            logger.info(f"epoch{epoch + 1}, Weigths saved successfully")
            
    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":
    ## 设置随机种子，方便复现
    torch.manual_seed(123)

    ## 加载分词器（中文用cl100k_base，英文用gpt2）
    tokenizer = tiktoken.get_encoding("cl100k_base")

    ## 更新配置参数，因为词表大小必须要与分词器的大小一致
    GPT_CONFIG_355M["vocab_size"] = tokenizer.max_token_value
    model = GPTModel(GPT_CONFIG_355M)

    ## 测试模型是否可以正常推理
    model.eval()
    start_context = "早上出门的时候我才发现忘记带钥匙，只好又回到家里。"
    token_ids = generate_text_token(
        model=model,     
        idx=text_to_token_ids(start_context, tokenizer),     
        max_new_tokens=20,     
        context_size=GPT_CONFIG_355M["context_length"] 
    ) 
    print("test model output:\n", token_ids_to_text(token_ids, tokenizer))



    ## 数据加载器
    train_loader = create_dataloader_bins(     
        "/home/hjzd/lzz/LLM_training/data/pretrain/CCI3/data_bin/train",     
        batch_size=4,     
        max_length=GPT_CONFIG_355M["context_length"],     
        stride=GPT_CONFIG_355M["context_length"],     
        drop_last=True,     
        shuffle=True,     
        num_workers=0 
    )

    val_loader = create_dataloader_bins(     
        "/home/hjzd/lzz/LLM_training/data/pretrain/CCI3/data_bin/val",     
        batch_size=2,     
        max_length=GPT_CONFIG_355M["context_length"],     
        stride=GPT_CONFIG_355M["context_length"],     
        drop_last=False,     
        shuffle=False,     
        num_workers=0
    ) 


    ## 指定训练的设备，将模型加载到设备上，加载优化器
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)


    ## 这里根据需要加载已有的权重接着训练，如果不需要直接注释掉
    # checkpoint_model = torch.load("/home/hjzd/lzz/LLM_training/logs/pretraining/2026-01-06_17-22-12_gpt124m/checkpoints/model_epoch1_step300000.pt", map_location=device) 
    # checkpoint_optimizer = torch.load("/home/hjzd/lzz/LLM_training/logs/pretraining/2026-01-06_17-22-12_gpt124m/checkpoints/optimizer_epoch1_step300000.pt", map_location=device) 
    # model.load_state_dict(checkpoint_model) 
    # optimizer.load_state_dict(checkpoint_optimizer)
    # # 这里可以在断点重新训练时重新设置学习率
    # new_lr = 1e-5
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = new_lr


    ## 开始训练
    ## 注意！ 权重保存是优化器和模型权重都保存，如果不需要保存优化器，请到代码里自行修改。
    num_epochs = 100 
    train_losses, val_losses, tokens_seen = train_model_simple(     
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        device=device,
        num_epochs=num_epochs, 
        eval_freq=2000,  ## 每eval_freq个step就计算一次train_loss和val_loss
        eval_iter=10,  ## 每次验证取eval_iter个批次进行验证，也就是计算loss。这里总样本数应该为batch_size * eval_iter
        start_context="早上出门的时候我才发现忘记带钥匙，只好又回到家里。", ## 训练过程中每隔eval_freq * 10个step就生成一次文本
        tokenizer=tokenizer, 
        save_epoch=1,  ## 每隔save_epoch轮保存一次
        save_step=50000  ## 每隔save_step轮保存一次, 由于大模型训练，数据量较大，训练完一轮时间可能比较久，而且模型有可能一轮不到就已经收敛并且有很好的效果，所以这里可以根据step来保存
    ) 


    