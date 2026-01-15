import token
import torch
import tiktoken

## 文本转token_id
## 返回的shape为 [1, token_ids], 因为模型推理需要按照批次为第一维度, 所以这里加了批次的维度
def text_to_token_ids(text, tokenizer):     
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})     
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) 
    return encoded_tensor

## token_ids转文本
## 这里也是同理，输入的shape为: [1, token_ids], 输出为str类型的文本
def token_ids_to_text(token_ids, tokenizer):     
    flat = token_ids.squeeze(0)   
    return tokenizer.decode(flat.tolist())

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("cl100k_base")
    text = "你好，我喜欢吃肯德基。"
    tokens = text_to_token_ids(text, tokenizer)
    print(f"tokens: {tokens}")
    text1 = token_ids_to_text(tokens, tokenizer)
    print(type(text1))