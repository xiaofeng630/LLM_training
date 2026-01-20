import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================
# 1. 模型加载
# ======================
MODEL_PATH = "/home/hjzd/lzz/LLM_training/qwen2.5/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
model.eval()


# ======================
# 2. 推理函数
# ======================
def chat(user_input, history):
    """
    user_input: str
    history: list of {"role": ..., "content": ...}
    """

    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        }
    ]

    # 把历史直接拼进去（已经是 messages 格式）
    messages.extend(history)

    # 当前用户输入
    messages.append({"role": "user", "content": user_input})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )

    output_ids = generated_ids[0][model_inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    # ✅ 更新 history（messages 格式）
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})

    return history, history


# ======================
# 3. Gradio UI
# ======================
with gr.Blocks(title="Qwen2.5-0.5B Chat Demo") as demo:
    gr.Markdown("# 🤖 Qwen2.5-0.5B-Instruct Demo")

    chatbot = gr.Chatbot(height=500)
    state = gr.State([])

    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="请输入你的问题，按 Enter 发送"
        )

    with gr.Row():
        clear = gr.Button("清空对话")

    txt.submit(chat, [txt, state], [chatbot, state])
    clear.click(lambda: ([], []), None, [chatbot, state])


# ======================
# 4. 启动
# ======================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=9007)
