GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

GPT_CONFIG_355M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": False
}

GPT_CONFIG_774M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1280,
    "n_heads": 20,
    "n_layers": 36,
    "drop_rate": 0.1,
    "qkv_bias": False
}

GPT_CONFIG_1558M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1600,
    "n_heads": 25,
    "n_layers": 48,
    "drop_rate": 0.1,
    "qkv_bias": False
}

QWEN_CONFIG_600M = {
    "vocab_size": 50257,
    "context_length": 1024,

    # ====== 核心规模 ======
    "emb_dim": 1152,        # 介于 1024 和 1280
    "n_heads": 18,          # 1152 / 18 = 64（完美 head_dim）
    "n_layers": 30,         # 介于 24 和 36

    # ====== 架构选择 ======
    "norm_type": "rmsnorm",
    "ffn_type": "swiglu",
    "use_rope": True,
    "rope_theta": 10000.0,
    "tie_embeddings": True,

    # ====== Attention ======
    "qkv_bias": False,
    "attn_dropout": 0.0,

    # ====== FFN ======
    "ffn_mult": None,       # auto: 2 * d * 4 / 3

    # ====== 训练稳定性 ======
    "drop_rate": 0.1,
    "resid_dropout": 0.1,
    "norm_eps": 1e-6,
}