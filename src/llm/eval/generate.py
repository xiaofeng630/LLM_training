import torch
from src.llm.eval.tokenizer import token_ids_to_text, text_to_token_ids

## 使用自回归方式，从给定的 token 序列中逐 token 生成文本。
## idx.shape: (B, T_cur), 也就是[batch_size, max_length] 
@torch.no_grad()
def generate_text_token(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=50, eos_id=None):
    ## 主循环：每一次循环生成 1 个新 token
    for _ in range(max_new_tokens):

        ## 1. 截断上下文，只取最近的 context_size 个 token，因为只能喂给模型max_length长度（这里解释，自己理解）
        idx_cond = idx[:, -context_size:]

        ## 2. 前向推理，得到 logits，shape：[B, max_length, vocab_size]
        with torch.no_grad():
            logits = model(idx_cond)

        ## 3. 只取最后一个位置的 logits，因为是自回归生成，只关心“下一个 token”的分布
        logits = logits[:, -1, :]

        ## 4. Top-k 过滤（logits 层面）
        if top_k is not None:
            ## 取 logits 中最大的 k 个值
            top_logits, _ = torch.topk(logits, top_k)

            ## 每个 batch 中，第 k 大的 logit
            min_val = top_logits[:, -1]

            ## 小于 min_val 的位置全部置为 -inf，这样 softmax 后概率为 0
            logits = torch.where(
                logits < min_val.unsqueeze(-1),   # (B, V)
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        ## 5. 根据 temperature 进行采样 / 贪心。简单说就是“是不是要严格服从模型的偏好”
        ## temperature越大，各个token之间的概率差距会被拉小，会具有更多的可能性
        ## temperature越小，各个token之间的概率差距会被拉大，每次几乎就会选择top-1的token导致重复生成文本
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            # multinomial 按概率采样一个 token
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # temperature == 0 等价于 greedy decoding，选概率最大的 token
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        ## 6. EOS 判断（是否提前结束）
        ## idx_next 的 shape 是 (B, 1)
        ## 如果 B > 1，这里逻辑并不严格
        if eos_id is not None:
            ## 当 batch_size=1 时，这个判断是合理的
            if idx_next.item() == eos_id:
                break

        ## 7. 将新 token 拼接回序列
        ## idx:      (B, T_cur)
        ## idx_next: (B, 1)
        ## 新 idx:   (B, T_cur + 1)
        idx = torch.cat((idx, idx_next), dim=1)

    ## 返回完整生成序列（包含输入 + 新生成的 token）
    return idx

## 此版本已弃用
def generate_text_simple_old(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.2):
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

## 输入文本和模型，直接返回模型生成的文本，训练时测试用
def generate_and_print_sample(model, tokenizer, device, start_context, eos_id):     
    model.eval()     
    context_size = model.pos_emb.weight.shape[0]     
    encoded = text_to_token_ids(start_context, tokenizer).to(device)     
    with torch.no_grad():         
        token_ids = generate_text_token(             
            model=model, idx=encoded,             
            max_new_tokens=1024, context_size=context_size, eos_id=eos_id       
        )     
    decoded_text = token_ids_to_text(token_ids, tokenizer) 
    decoded_text = decoded_text.replace("\n", " ")
    model.train()
    return decoded_text