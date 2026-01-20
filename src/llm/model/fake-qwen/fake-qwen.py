import torch
import torch.nn as nn
from .transformer import TransformerBlock
from .transformer import LayerNorm, RMSNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ## Token embedding：把词 ID 映射成向量
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        ## Position embedding：提供位置信息（GPT 使用可学习位置编码）
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])    

        ## Embedding 后的 dropout，防止过拟合
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        ## 多层 Transformer Block（自注意力 + FFN）
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg)
              for _ in range(cfg["n_layers"])]
        )

        ## 最后一层 LayerNorm，稳定输出分布（Pre-LN 架构的收尾）
        self.final_norm = RMSNorm(cfg["emb_dim"])

        ## 输出头：将隐藏状态映射到词表维度，得到 logits 
        self.out_head = nn.Linear(             
            cfg["emb_dim"], cfg["vocab_size"], bias=False     
        )

        self.out_head.weight = self.tok_emb.weight

    def forward(self, in_idx):
        ## in_idx: [batch_size, seq_len]
        batch_size, seq_len = in_idx.shape

        ## 词嵌入：[B, T] -> [B, T, emb_dim]   T = sequence length（每条序列有多少个 token）
        tok_embeds = self.tok_emb(in_idx)

        ## 位置嵌入：[T] -> [T, emb_dim]
        # pos_embeds = self.pos_emb(          
        #     torch.arange(seq_len, device=in_idx.device)         
        # )

        ## token embedding + position embedding
        x = tok_embeds

        ## dropout，提高泛化能力   
        x = self.drop_emb(x)

        ## 通过多层 Transformer，建模上下文依赖  
        x = self.trf_blocks(x)

        ## 最终归一化，保证数值稳定   
        x = self.final_norm(x)

        # 映射到词表维度，得到每个位置的预测 logits 
        logits = self.out_head(x)    
        return logits