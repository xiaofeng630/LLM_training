import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
import tiktoken
import torch
import json
import time
import numpy as np
from src.llm.config.gpt_configs import GPT_CONFIG_355M, QWEN_CONFIG_600M
from src.llm.eval.tokenizer import text_to_token_ids, token_ids_to_text
from src.llm.eval.generate import generate_text_token
from src.llm.utils.logger import setup_logger, setup_run_dir
from src.llm.model.gpt import GPTModel
from src.llm.model.qwen.qwen import QwenModel
from src.llm.utils.loss_tracker import LossTracker
from src.llm.utils.gpt_download import download_and_load_gpt2
from src.llm.tasks.instruction.dataset import InstructionDataset, create_dataloader_Belle, create_dataloader_Instruction, format_input_Alpaca
from src.llm.tasks.instruction.loss import calc_loss_batch, calc_loss_loader
from src.llm.eval.generate import generate_text_simple_old, generate_text_token, generate_and_print_sample

## 设置日志模块并记录日志
run_dir = setup_run_dir(experiment="instruction", run_name="qwen600m")
logger = setup_logger(name="train", log_file=run_dir / "train.log")

## loss记录模块，用于实时画loss曲线图
loss_tracker = LossTracker(run_dir / "samples")

## 用于创建指令微调的数据格式（实际的格式规整应该放到具体的Dataset类中，这里主要是为了测试）

## 用于推理时输入的数据格式
def format_prompt(entry):
    instruction = entry["instruction"].strip()
    input_text = entry.get("input", "").strip()

    prompt = "### Instruction:\n"
    prompt += instruction

    if input_text:
        prompt += f"\n\n### Input:\n{input_text}"

    prompt += "\n\n### Response:\n"

    return prompt

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

## 训练入口函数
def train_model_simple(model, train_loader, val_loader, eval_train_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer, save_epoch, save_step):
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
                    model, eval_train_loader, val_loader, device, eval_iter
                )
                loss_tracker.update_train(global_step, train_loss)
                loss_tracker.update_val(global_step, val_loss)
                train_losses.append(train_loss)                 
                val_losses.append(val_loss)                 
                track_tokens_seen.append(tokens_seen)  
                lr = optimizer.param_groups[0]["lr"]

                if global_step % (eval_freq * 1) == 0:
                    sample_text = generate_and_print_sample( 
                        model, tokenizer, device, start_context, eos_id=tokenizer.eot_token
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
            
            ## 每隔save_step个step就保存一次模型和优化器权重
            if global_step % save_step == 0:
                ckpt_path = run_dir / "checkpoints" / f"model_epoch{epoch + 1}_step{global_step}.pt"
                optimizer_path = run_dir / "checkpoints" / f"optimizer_epoch{epoch + 1}_step{global_step}.pt"
                torch.save(model.state_dict(), ckpt_path)
                torch.save(optimizer.state_dict(), optimizer_path)
                logger.info(f"epoch{epoch + 1}_step{global_step}, Weigths saved successfully")
        
        ## 每个epoch保存一次权重
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

    BASE_CONFIG = QWEN_CONFIG_600M

    ## 加载分词器并更新配置参数的词表大小
    tokenizer = tiktoken.get_encoding("cl100k_base")
    BASE_CONFIG["vocab_size"] = tokenizer.max_token_value

    ## 初始化模型和设备
    model = QwenModel(BASE_CONFIG)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ## 加载预训练模型权重并且加载优化器
    ckpt_model = torch.load("/home/hjzd/lzz/LLM_training/logs/pretraining/qwen600m-exp1/checkpoints/model_epoch1.pt", map_location=device)
    model.load_state_dict(ckpt_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
    

    ## 测试是否正确加载了权重
    torch.manual_seed(123)
    token_ids = generate_text_token(
        model=model,
        idx=text_to_token_ids("任何努力都将使你变的", tokenizer).to(device),
        max_new_tokens=50,
        context_size=BASE_CONFIG["context_length"],
        top_k=50,
        temperature=1.0
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


    ## 加载数据集
    ## 注意！这里要根据实际的数据集进行加载，因为每个数据集的存储方式和格式都不同，需要根据特定的数据集去写Dataset类
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


    ## 用于在微调过程中查看推理是否正常（不是看效果，主要是看模型是否会生成正常回复而不是一堆乱码）
    entry = {"instruction": "请写一篇有关环境保护的文章", "input": ""}
    model_input = format_prompt(entry)


    
    ## 开始训练
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple( 
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        eval_train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=500,
        eval_iter=20,
        tokenizer=tokenizer,
        start_context=model_input,
        save_epoch=1,
        save_step=50000
    )
