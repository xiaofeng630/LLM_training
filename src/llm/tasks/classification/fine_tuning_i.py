import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
import tiktoken
import time
import torch
import numpy as np
from src.llm.utils.loss_tracker import LossTracker
from src.llm.model.gpt import GPTModel
from src.llm.eval.tokenizer import text_to_token_ids, token_ids_to_text
from src.llm.eval.generate import generate_text_token
from src.llm.utils.logger import setup_logger, setup_run_dir
from src.llm.tasks.classification.datasets import SpamDataset, create_dataloader_Spam, create_dataloader_ChnSentiCorp
from src.llm.utils.gpt_download import download_and_load_gpt2
from src.llm.tasks.classification.loss import calc_loss_batch, calc_loss_loader

run_dir = setup_run_dir(experiment="fine_tuning_classify", run_name="gpt124m")
logger = setup_logger(name="fine_tuning", log_file=run_dir / "fine_tuning.log")
loss_tracker = LossTracker(run_dir / "samples")


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()     
    correct_predictions, num_examples = 0, 0
    if num_batches is None:         
        num_batches = len(data_loader)     
    else:         
        num_batches = min(num_batches, len(data_loader))     
    for i, (input_batch, target_batch) in enumerate(data_loader):         
        if i < num_batches:             
            input_batch = input_batch.to(device)             
            target_batch = target_batch.to(device)

            with torch.no_grad():                 
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]             
            correct_predictions += (                 
                (predicted_labels == target_batch).sum().item()             
            )
        
        else:             
            break     
    return correct_predictions / num_examples

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

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, save_epoch):
    track_examples_seen = []
    train_losses, val_losses, train_accs, val_accs = [], [], [], [] 
    examples_seen, global_step = 0, -1

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
            examples_seen += input_batch.shape[0]   
            global_step += 1
            
            if global_step % eval_freq == 0: 
                train_loss, val_loss = evaluate_model(                     
                    model, train_loader, val_loader, device, eval_iter
                )
                loss_tracker.update_train(global_step, train_loss)
                loss_tracker.update_val(global_step, val_loss)
                train_losses.append(train_loss)                 
                val_losses.append(val_loss)                 
                track_examples_seen.append(examples_seen)  
                lr = optimizer.param_groups[0]["lr"]

                logger.info(
                    f"epoch={epoch+1} step={global_step:06d} | "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                    f"tokens_seen={examples_seen} | lr={lr:.2e}\n"
                )
                loss_tracker.plot("loss.png")

                print(f"Ep {epoch+1} (Step {global_step:06d}): "                       
                      f"Train loss {train_loss:.3f}, "                       
                      f"Val loss {val_loss:.3f}\n"            
                )
            
            if global_step % 100000 == 0:
                ckpt_path = run_dir / "checkpoints" / f"model_epoch{epoch + 1}_step{global_step}.pt"
                optimizer_path = run_dir / "checkpoints" / f"optimizer_epoch{epoch + 1}_step{global_step}.pt"
                torch.save(model.state_dict(), ckpt_path)
                torch.save(optimizer.state_dict(), optimizer_path)
                logger.info(f"epoch{epoch + 1}_step{global_step}, Weigths saved successfully")
        
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)         
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)

        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")         
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)         
        val_accs.append(val_accuracy)



        if (epoch + 1) % save_epoch == 0:
            ckpt_path = run_dir / "checkpoints" / f"model_epoch{epoch + 1}.pt"
            optimizer_path = run_dir / "checkpoints" / f"optimizer_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            logger.info(f"epoch{epoch + 1}, Weigths saved successfully")
            loss_tracker.plot("loss.png")


    return train_losses, val_losses, train_accs, val_accs, examples_seen


if __name__ == "__main__":
    ## 配置模型参数
    BASE_CONFIG = {
        "vocab_size": 100276, 
        "context_length": 256,     
        "emb_dim": 768,     
        "n_heads": 12,     
        "n_layers": 12, 
        "drop_rate": 0.1,
        "qkv_bias": False 
    }
    model_configs = {     
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},     
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},     
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},     
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}, 
    } 

    tokenizer = tiktoken.get_encoding("cl100k_base")

    ## 加载数据集
    train_loader = create_dataloader_ChnSentiCorp(
        "/home/hjzd/lzz/LLM_training/data/classification/ChnSentiCorp/train.parquet",
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eot_token,
        batch_size=16,
        max_length=256,
        shuffle=True, 
        drop_last=True, 
        num_workers=0
    )

    val_loader = create_dataloader_ChnSentiCorp(
        "/home/hjzd/lzz/LLM_training/data/classification/ChnSentiCorp/val.parquet",
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eot_token,
        batch_size=16,
        max_length=256,
        shuffle=True, 
        drop_last=True, 
        num_workers=0
    )

    test_loader = create_dataloader_ChnSentiCorp(
        "/home/hjzd/lzz/LLM_training/data/classification/ChnSentiCorp/test.parquet",
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eot_token,
        batch_size=8,
        max_length=256,
        shuffle=True, 
        drop_last=True, 
        num_workers=0
    )

    model = GPTModel(BASE_CONFIG)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ckpt_model = torch.load("/home/hjzd/lzz/LLM_training/logs/pretraining/2026-01-07_17-09-59_gpt124m/checkpoints/model_epoch4_step2200000.pt", map_location=device)
    model.load_state_dict(ckpt_model)
    

    ## 测试是否正确加载了权重
    torch.manual_seed(123) 
    token_ids = generate_text_token(     
        model=model,     
        idx=text_to_token_ids("任何努力都可以使你变得更好", tokenizer).to(device),     
        max_new_tokens=50,     
        context_size=BASE_CONFIG["context_length"],     
        top_k=50,     
        temperature=1.0
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    ## 测试原始权重的分类能力
    text_2 = ("Is the following text 'spam'? Answer with 'yes' or 'no':'You are a winner you have been specially selected to receive $1000 cash or a $2000 award." ) 
    token_ids = generate_text_token(     
        model=model,     
        idx=text_to_token_ids(text_2, tokenizer).to(device),     
        max_new_tokens=23,     
        context_size=BASE_CONFIG["context_length"] 
    ) 
    print(token_ids_to_text(token_ids, tokenizer))

    ## 查看模型架构
    # print(model)

    ## 先将所有的参数全部冻结
    for param in model.parameters():     
        param.requires_grad = False

    ## 按照分类数来设置分类头
    torch.manual_seed(123) 
    num_classes = 2 
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

    ## 解开最后一层transformers块和最后一层线性层的参数用于微调
    for param in model.trf_blocks[-1].parameters():     
        param.requires_grad = True 
    for param in model.final_norm.parameters():     
        param.requires_grad = True

    ## 测试原始权重在数据集上的分类效果（可有可无）
    # model.to(device)
    # train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10) 
    # val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10) 
    # test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)
    # print(f"Training accuracy: {train_accuracy*100:.2f}%") 
    # print(f"Validation accuracy: {val_accuracy*100:.2f}%") 
    # print(f"Test accuracy: {test_accuracy*100:.2f}%")


    ## 计算初始权重在数据集上的损失（可有可无）
    # with torch.no_grad():
    #     train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)     
    #     val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)     
    #     test_loss = calc_loss_loader(test_loader, model, device, num_batches=5) 
    # print(f"Training loss: {train_loss:.3f}") 
    # print(f"Validation loss: {val_loss:.3f}") 
    # print(f"Test loss: {test_loss:.3f}")


    start_time = time.time() 
    torch.manual_seed(123) 
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1) 
    num_epochs = 40
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(         
        model, train_loader, val_loader, optimizer, device,         
        num_epochs=num_epochs, eval_freq=100,         
        eval_iter=30, save_epoch=10
    )
    end_time = time.time() 
    execution_time_minutes = (end_time - start_time) / 60 
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")




