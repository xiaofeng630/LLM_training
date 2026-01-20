#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn 
import torch.nn.functional as F

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):         
        super().__init__()         
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))         
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))         
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):         
        keys = x @ self.W_key         
        queries = x @ self.W_query         
        values = x @ self.W_value         
        attn_scores = queries @ keys.T # omega         
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)         
        context_vec = attn_weights @ values         
        return context_vec

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

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,                 
                 dropout, qkv_bias=False):         
        super().__init__()         
        self.d_out = d_out        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)      
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)       
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias) 
        self.dropout = nn.Dropout(dropout) 
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):         
        b, num_tokens, d_in = x.shape         
        keys = self.W_key(x)    
        queries = self.W_query(x)        
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)         
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)         
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) 
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values 
        return context_vec

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
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out       
        self.num_heads = num_heads 
        self.head_dim = d_out // num_heads         
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)         
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)         
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)         
        self.out_proj = nn.Linear(d_out, d_out)         
        self.dropout = nn.Dropout(dropout)        
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)         
        )

    def forward(self, x): 
        b, num_tokens, d_in = x.shape

        ## (batch, num_token, d_out) 
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # print(values.shape)

        ## (batch, num_token, num_heads, head_dim)
        ## d_out = num_heads * head_dim
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim) 
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim) 
        # print(keys.shape)

        ## (b, num_heads, num_token, head_dim)
        keys = keys.transpose(1, 2) 
        queries = queries.transpose(1, 2) 
        values = values.transpose(1, 2)
        # print(keys)

        ## 1、下面这段为原始的计算方式，更好理解过程但是训练会变慢
        # ## (b, num_heads, num_token, head_dim) @ (b, num_heads, head_dim, num_token) -> (b, num_heads, num_token, num_token)
        # attn_scores = queries @ keys.transpose(2, 3)
        # mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # attn_scores.masked_fill_(mask_bool, -torch.inf) 
        # attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) 
        # attn_weights = self.dropout(attn_weights)
        # # print(attn_weights.shape)
        # # print(values.shape)

        # ## (b, num_token, num_heads, head_dim)
        # context_vec = (attn_weights @ values).transpose(1, 2)
        # # print(context_vec.shape)
        # context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # context_vec = self.out_proj(context_vec) # (b, num_tokens, n_heads, head_dim)
        # return context_vec



        ## 2、下面是FlashAttention的实现方式，可以加快训练速度，具体的不解释，请自行了解
        ## FlashAttention入口
        attn_output = F.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=None,        # causal 用 is_causal
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True         # GPT-style causal mask
        )

        ## (b, h, L, d) → (b, L, d_out)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(b, num_tokens, self.d_out)
        
        return self.out_proj(attn_output)