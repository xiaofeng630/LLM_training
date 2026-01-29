#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn 
import torch.nn.functional as F

## 最简单、最基本的注意力机制实现
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out): 
        super().__init__()
        ## 三个可学习的线性映射：
        ## 把每个 token 的输入向量 (d_in) 投影到新的空间 (d_out)
        ## 也就是embedding的维度，本来token被emb到了128个维度，然后这里转换后也是同样的维度
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))         
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))         
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        ## x: (seq_len, d_in)
        ## 每一行是一个 token 的表示
        keys = x @ self.W_key    ## keys: (seq_len, d_out) —— 用来“被匹配”
        queries = x @ self.W_query    ## queries:(seq_len, d_out) —— 用来“发起匹配”
        values = x @ self.W_value    ## values: (seq_len, d_out) —— 最终被加权汇聚的信息

        ## attn_scores: (seq_len, seq_len)  第 i 行第 j 列：第 i 个 token 对第 j 个 token 的关注程度（未归一化）
        attn_scores = queries @ keys.T

        ## 缩放 + softmax：
        ## 1) 除以 sqrt(d_out) 防止点积过大导致 softmax 饱和
        ## 2) softmax 后，每一行变成对所有 token 的注意力分布（和为 1）
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        ## context_vec: (seq_len, d_out)
        ## 每个 token 用自己的注意力权重，对所有 value 做加权求和
        ## 最终得到“融合了上下文信息”的新表示
        context_vec = attn_weights @ values
        return context_vec


## 完全等价v1，只是使用了线性变换，更加清晰直观
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):         
        super().__init__()         
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)         
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)         
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):         
        keys = self.W_key(x)         
        queries = self.W_query(x)         
        values = self.W_value(x)         
        attn_scores = queries @ keys.T         
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)         
        context_vec = attn_weights @ values         
        return context_vec


## 因果注意力机制，加入了掩码，让当前token只能看到之前的信息，而不能看到未来的信息
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out

        ## Q / K / V 线性投影：(d_in → d_out)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)      
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)       
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        ## 用在注意力权重上的 dropout（正则化）
        self.dropout = nn.Dropout(dropout)

        ## 上三角 mask（不含对角线）  mask[i, j] = 1 表示：位置 i 不能看位置 j（未来 token）
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        ## x: (batch_size, num_tokens, d_in)
        b, num_tokens, d_in = x.shape

        ## (b, n, d_in) → (b, n, d_out)   
        keys = self.W_key(x)    
        queries = self.W_query(x)        
        values = self.W_value(x)

        ## (b, n, d_out) @ (b, d_out, n) → (b, n, n)
        attn_scores = queries @ keys.transpose(1, 2)

        ## 对“未来位置”打 mask，强制其注意力为 0，当前token只能看到以前的信息，看不到未来的信息
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)    

        ## 缩放 + softmax，得到注意力分布     
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) 

        ## 对注意力权重做 dropout（防止过度依赖某些 token）
        attn_weights = self.dropout(attn_weights)

        # (b, n, n) @ (b, n, d_out) → (b, n, d_out)
        context_vec = attn_weights @ values 
        return context_vec


## 多头注意力机制，也就是多个头，这里就是线性的拼上了多个因果注意力机制
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length,                  
                 dropout, num_heads, qkv_bias=False):         
        super().__init__()         
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):         
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,                  
                 context_length, dropout, num_heads, qkv_bias=False):         
        super().__init__()

        ## 多头要求：d_out 必须能均分给每个 head
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out       
        self.num_heads = num_heads 
        self.head_dim = d_out // num_heads  ## 每个 head 的维度

        ## 一次性投影到 d_out，之后再拆成 num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        ## 多头拼接后的线性映射（head mixing） 
        self.out_proj = nn.Linear(d_out, d_out)

        ## 注意力权重上的 dropout   
        self.dropout = nn.Dropout(dropout)

        ## causal mask（防止看未来） 
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)         
        )

    def forward(self, x):
        ## x: (b, L, d_in) [批次，长度，emb维度]
        b, num_tokens, d_in = x.shape

        ## 线性变换 (b, L, d_in) → (b, L, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        ## 拆成多头：(b, L, num_heads, head_dim)  d_out = num_heads * head_dim
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim) 
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim) 

        ## 调整维度，方便并行算注意力
        ## (b, num_heads, L, head_dim)
        keys = keys.transpose(1, 2) 
        queries = queries.transpose(1, 2) 
        values = values.transpose(1, 2)


        # --------------------------------------------------
        # 方案 1：手写 attention（直观，但慢）——你已经写得很标准
        # --------------------------------------------------
        # ## (b, num_heads, num_token, head_dim) @ (b, num_heads, head_dim, num_token) -> (b, num_heads, num_token, num_token)
        # ## 每个 head 内：当前 token 对所有 token 的打分
        # attn_scores = queries @ keys.transpose(2, 3)

        # ## 取出与当前序列长度匹配的 causal mask。True 表示“不能看（未来 token）”
        # mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # ## 对未来位置打 -inf，softmax 后权重变为 0
        # attn_scores.masked_fill_(mask_bool, -torch.inf)

        # ## 缩放 + softmax。每一行变成：对所有 token 的注意力分布
        # attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # ## 对注意力权重做 dropout（正则化关注关系）
        # attn_weights = self.dropout(attn_weights)

        # ## (b, h, L, L) @ (b, h, L, d) → (b, h, L, d) → (b, L, h, d)
        # ## 每个 token 在每个 head 上加权汇聚 value
        # context_vec = (attn_weights @ values).transpose(1, 2)

        # ## (b, L, h, d) → (b, L, d_out)
        # ## 拼接所有 head
        # context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # ## 最后一层线性映射，混合各个 head 的信息
        # context_vec = self.out_proj(context_vec) # (b, num_tokens, n_heads, head_dim)
        # return context_vec



        # --------------------------------------------------
        # 方案 2：FlashAttention（PyTorch 内置高效实现）不用完整计算注意力分数矩阵
        # --------------------------------------------------
        attn_output = F.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=None,        # causal 用 is_causal
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True         # GPT-style causal mask
        )

        ## 合并多头：(b, h, L, d) → (b, L, d_out)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(b, num_tokens, self.d_out)
        
        # 最后一层线性映射，混合各个 head 的信息
        return self.out_proj(attn_output)