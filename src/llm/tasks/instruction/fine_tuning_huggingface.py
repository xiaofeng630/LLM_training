import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================================
# 你的工具
# =====================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.llm.utils.logger import setup_logger, setup_run_dir
from src.llm.utils.loss_tracker import LossTracker


# =====================================================
# run_dir & logger
# =====================================================
run_dir = setup_run_dir(
    experiment="instruction",
    run_name="qwen3-0.6b-base-sft"
)
logger = setup_logger("train", run_dir / "train.log")
loss_tracker = LossTracker(run_dir / "samples")


# =====================================================
# 指令模板（与你现有版本完全一致）
# =====================================================
def format_input(entry):
    instruction_text = (
        # "以下是一项任务描述。请撰写一份恰当的回复来完成该任务。"
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input:\n{entry['input']}"
        if entry.get("input", "")
        else ""
    )
    return instruction_text + input_text


# =====================================================
# HF Dataset → tokenize
# =====================================================
def preprocess_function(example, tokenizer, max_length):
    prompt = format_input(example)
    answer = example["output"]

    full_text = prompt + "\n\n### Response:\n" + answer

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False
    )

    input_ids = tokenized["input_ids"]
    labels = input_ids.copy()

    return {
        "input_ids": input_ids,
        "labels": labels
    }


# =====================================================
# collate_fn（pad + mask loss）
# =====================================================
def collate_fn(batch, pad_token_id):
    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=pad_token_id
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=pad_token_id
    )

    labels[labels == pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "labels": labels
    }


# =====================================================
# sample 生成
# =====================================================
@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt):
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=" * 40)
    print(text)
    print("=" * 40)

    model.train()
    return text


# =====================================================
# eval
# =====================================================
@torch.no_grad()
def evaluate_model(model, dataloader, device, eval_iter):
    model.eval()
    losses = []

    for step, batch in enumerate(dataloader):
        if step >= eval_iter:
            break

        batch = {
            "input_ids": batch["input_ids"].to(device),
            "labels": batch["labels"].to(device)
        }

        outputs = model(**batch)
        losses.append(outputs.loss.item())

    model.train()
    return sum(losses) / len(losses)


# =====================================================
# train loop（与你原结构高度一致）
# =====================================================
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    tokenizer,
    sample_prompt,
    save_epoch,
    save_step
    ):
    global_step = 0
    tokens_seen = 0

    logger.info("Start instruction fine-tuning (HF Dataset)")
    logger.info(f"epochs={num_epochs}, eval_freq={eval_freq}")

    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()

            batch = {
                "input_ids": batch["input_ids"].to(device),
                "labels": batch["labels"].to(device)
            }

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss

            loss.backward()
            optimizer.step()

            global_step += 1
            tokens_seen += batch["input_ids"].numel()

            if global_step % eval_freq == 0:
                train_loss = loss.item()
                val_loss = evaluate_model(
                    model, val_loader, device, eval_iter
                )

                loss_tracker.update_train(global_step, train_loss)
                loss_tracker.update_val(global_step, val_loss)
                loss_tracker.plot("loss.png")

                lr = optimizer.param_groups[0]["lr"]

                logger.info(
                    f"epoch={epoch+1} step={global_step:06d} | "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                    f"tokens={tokens_seen/1e6:.2f}M lr={lr:.2e}"
                )

                sample_text = generate_sample(
                    model, tokenizer, device, sample_prompt
                )
                logger.info(f"sample:\n{sample_text}")

            if global_step % save_step == 0:
                ckpt_dir = run_dir / "checkpoints" / f"step{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

        if (epoch + 1) % save_epoch == 0:
            ckpt_dir = run_dir / "checkpoints" / f"epoch{epoch+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)


# =====================================================
# main
# =====================================================
if __name__ == "__main__":
    torch.manual_seed(123)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    model_path = "/home/hjzd/lzz/LLM_training/qwen3/Qwen3-0.6B-Base"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.1
    )

    # ======================
    # HuggingFace Dataset
    # ======================
    ds = load_dataset(
        "/home/hjzd/lzz/LLM_training/data/instruction/BelleGroup/train_0.5M_CN"
    )

    tokenized_ds = ds.map(
        lambda x: preprocess_function(x, tokenizer, 1024),
        remove_columns=ds["train"].column_names,
        num_proc=4
    )

    train_loader = DataLoader(
        tokenized_ds["train"],
        batch_size=4,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id)
    )

    val_loader = DataLoader(
        tokenized_ds["train"].select(range(2000)),
        batch_size=2,
        shuffle=False,
        drop_last=True,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id)
    )

    # ======================
    # sample prompt
    # ======================
    sample_entry = {
        "instruction": "请写一篇关于保护环境的文章",
        "input": ""
    }
    sample_prompt = format_input(sample_entry)

    # ======================
    # train
    # ======================
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
        eval_freq=1000,
        eval_iter=20,
        tokenizer=tokenizer,
        sample_prompt=sample_prompt,
        save_epoch=1,
        save_step=50000
    )
