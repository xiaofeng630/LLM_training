import sys
import os
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            )
        )
    )
)

import torch
import tiktoken
import gradio as gr

from src.llm.model.gpt import GPTModel
from src.llm.config.gpt_configs import GPT_CONFIG_355M
from src.llm.eval.generate import generate_text_token
from src.llm.eval.tokenizer import text_to_token_ids, token_ids_to_text


# ======================
# 1. 模型 & Tokenizer 加载
# ======================

CKPT_PATH = "/home/hjzd/lzz/LLM_training/logs/instruction/2026-01-19_14-30-49_gpt355m/checkpoints/model_epoch7.pt"

BASE_CONFIG = GPT_CONFIG_355M

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

print("Loading tokenizer...")
tokenizer = tiktoken.get_encoding("cl100k_base")
GPT_CONFIG_355M["vocab_size"] = tokenizer.max_token_value

print("Loading model...")
model = GPTModel(BASE_CONFIG)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully!")


# ======================
# 2. Prompt 构造
# ======================

def format_input(instruction: str):
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        "\n\n### Instruction:\n"
        f"{instruction}\n\n### Response:\n"
    )


# ======================
# 3. 推理函数（Gradio 会调用）
# ======================

@torch.no_grad()
def predict(user_input):
    if not user_input.strip():
        return "Please enter an instruction."

    prompt = format_input(user_input)

    input_ids = text_to_token_ids(prompt, tokenizer).to(device)

    output_ids = generate_text_token(
        model=model,
        idx=input_ids,
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=tokenizer.eot_token
    )

    output_text = token_ids_to_text(output_ids, tokenizer)

    # 只截取 Response 部分
    response = output_text[len(prompt):].strip()
    return response


# ======================
# 4. Gradio UI
# ======================

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=6,
        label="Instruction",
        placeholder="Enter your instruction here..."
    ),
    outputs=gr.Textbox(
        lines=10,
        label="Model Response"
    ),
    title="GPT-355M Instruction Demo",
    description="A simple Gradio demo for your instruction-tuned GPT model.",
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=9007,
        share=False
    )
