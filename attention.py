#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tiktoken
import torch
torch.manual_seed(123)
raw_text = "Your journey starts with one step"

tokenizer = tiktoken.get_encoding("gpt2")
inputs = torch.tensor(tokenizer.encode(raw_text))

vocab_size = 50257 
output_dim = 3
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim) 
token_embeddings = token_embedding_layer(inputs)

context_length = 6
# Each location has its own position vector
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim) 
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

input_embeddings = token_embeddings + pos_embeddings 
print("pos_embeddings: ", pos_embeddings.shape)

print(input_embeddings)


# In[3]:


####### Simple attention mechanism #######


# In[4]:


### ------Calculate the attention score and the attention weight for a single word------ ###
import torch 
inputs = torch.tensor([[0.43, 0.15, 0.89], # Your     (x^1)    
                       [0.55, 0.87, 0.66], # journey  (x^2)    
                       [0.57, 0.85, 0.64], # starts   (x^3)    
                       [0.22, 0.58, 0.33], # with     (x^4)    
                       [0.77, 0.25, 0.10], # one      (x^5)    
                       [0.05, 0.80, 0.55]] # step     (x^6)
) 
query = inputs[1]
## This variable "atten_scores_2" is used to store the context vector of the word "journey"
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

attn_weights_2 = torch.softmax(attn_scores_2, dim=0) 
print("Attention weights:", attn_weights_2) 
print("Sum:", attn_weights_2.sum())


# In[5]:


### ------Calculate the context vector for a single word------ ###

qurey = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)


# In[6]:


### ------Calculate the context weight for all words------ ###
attn_scores = torch.empty(6, 6) 
# for i, x_i in enumerate(inputs):     
#     for j, x_j in enumerate(inputs):         
#         attn_scores[i, j] = torch.dot(x_i, x_j)
# print(attn_scores)

attn_scores = inputs @ inputs.T # 必为实对称矩阵！
attn_weights = torch.softmax(attn_scores, dim=-1) 
print(attn_weights)


# In[7]:


### ------ Calculate the context vector for all words------ ###
all_context_vecs = attn_weights @ inputs
print(all_context_vecs) 


# In[8]:


####### Attention mechanism with training weights ######


# In[35]:


x_2 = inputs[1]  
d_in = inputs.shape[1] 
d_out = 2

torch.manual_seed(123) 
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) 
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) 

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value 
print(query_2)

### ---Calculate keys and values for all words--- ###
keys = inputs @ W_key 
values = inputs @ W_value 
print("keys.shape:", keys.shape)
print("values.shape:", values.shape) 


# In[10]:


keys_2 = keys[1] 
attn_score_22 = query_2.dot(keys_2) 
print(attn_score_22)

attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

d_k = keys.shape[-1] 
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1) 
print(attn_weights_2)



# In[11]:


context_vec_2 = attn_weights_2 @ values 
print(context_vec_2)


# In[12]:


import torch.nn as nn 
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


# In[13]:


torch.manual_seed(123) 
sa_v1 = SelfAttention_v1(d_in, d_out) 
print(sa_v1(inputs))


# In[14]:


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


# In[15]:


torch.manual_seed(789) 
sa_v2 = SelfAttention_v2(d_in, d_out) 
print(sa_v2(inputs))


# In[16]:


####### Attention mechanism with mask ######


# In[17]:


queries = sa_v2.W_query(inputs) 
keys = sa_v2.W_key(inputs) 
attn_scores = queries @ keys.T 
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) 
print(attn_weights)

## Use tril to build a mask matrix
context_length = attn_scores.shape[0] 
mask_simple = torch.tril(torch.ones(context_length, context_length)) 
print(mask_simple)
masked_simple = attn_weights*mask_simple
print(masked_simple) 

## Normalization again
row_sums = masked_simple.sum(dim=-1, keepdim=True) 
masked_simple_norm = masked_simple / row_sums 
print(masked_simple_norm)


# In[18]:


## More efficient normalization
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1) 
masked = attn_scores.masked_fill(mask.bool(), -torch.inf) 
print(masked)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1) 
print(attn_weights)


# In[19]:


## start dropout
torch.manual_seed(123) 
dropout = torch.nn.Dropout(0.5) 
example = torch.ones(6, 6) 
print(dropout(example))

torch.manual_seed(123) 
print(dropout(attn_weights))


# In[20]:


batch = torch.stack((inputs, inputs), dim=0) 
print(batch)


# In[79]:


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



# In[80]:


torch.manual_seed(123) 
context_length = batch.shape[1] 
ca = CausalAttention(d_in, d_out, context_length, 0.0) 
context_vecs = ca(batch) 
# print("context_vecs.shape:", context_vecs)


# In[81]:


### ------Mutil-head attention mechanism------ ###


# In[82]:


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



# In[83]:


torch.manual_seed(123) 
context_length = batch.shape[1] # 这是词元的数量
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(     
    d_in, d_out, context_length, 0.0, num_heads=2 
) 
context_vecs = mha(batch)
print(context_vecs) 
print("context_vecs.shape:", context_vecs.shape)


# In[75]:


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

        # (batch, num_token, d_out) 
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # print(values.shape)

        # (batch, num_token, num_heads, head_dim)
        # d_out = num_heads * head_dim
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim) 
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim) 
        # print(keys.shape)

        # (b, num_heads, num_token, head_dim)
        keys = keys.transpose(1, 2) 
        queries = queries.transpose(1, 2) 
        values = values.transpose(1, 2)
        # print(keys)

        # (b, num_heads, num_token, head_dim) @ (b, num_heads, head_dim, num_token) -> (b, num_heads, num_token, num_token)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf) 
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) 
        attn_weights = self.dropout(attn_weights)
        # print(attn_weights.shape)
        # print(values.shape)

        # (b, num_token, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # print(context_vec.shape)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # (b, num_tokens, n_heads, head_dim)
        return context_vec





# In[76]:


torch.manual_seed(123) 
context_length = batch.shape[1] # 这是词元的数量
d_in, d_out = 3, 4
mha = MultiHeadAttention(     
    d_in, d_out, context_length, 0.0, num_heads=2
)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


# In[ ]:




