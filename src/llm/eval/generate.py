import torch
from src.llm.eval.tokenizer import token_ids_to_text, text_to_token_ids

@torch.no_grad()
def generate_text_token(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=50, eos_id=None):
    for _ in range(max_new_tokens):         
        idx_cond = idx[:, -context_size:]  
        with torch.no_grad():        
            logits = model(idx_cond) 
        logits = logits[:, -1, :]         
        if top_k is not None: 
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]             
            logits = torch.where(                 
                logits < min_val,                 
                torch.tensor(float('-inf')).to(logits.device),                 
                logits 
            ) 
        
        if temperature > 0.0: 
            logits = logits / temperature            
            probs = torch.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1)         
        else:             
            idx_next = torch.argmax(logits, dim=-1, keepdim=True) 
        
        if idx_next == eos_id:             
            break          
        idx = torch.cat((idx, idx_next), dim=1)
    return idx 

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

def generate_and_print_sample(model, tokenizer, device, start_context):     
    model.eval()     
    context_size = model.pos_emb.weight.shape[0]     
    encoded = text_to_token_ids(start_context, tokenizer).to(device)     
    with torch.no_grad():         
        token_ids = generate_text_token(             
            model=model, idx=encoded,             
            max_new_tokens=70, context_size=context_size         
        )     
    decoded_text = token_ids_to_text(token_ids, tokenizer) 
    decoded_text = decoded_text.replace("\n", " ")
    model.train()
    return decoded_text