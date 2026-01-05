from transformers import CamembertConfig
import os

os.makedirs("config", exist_ok=True)

config = CamembertConfig(
    vocab_size=32000,
    max_position_embeddings=514,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    layer_norm_eps=1e-5,
    initializer_range=0.02
)

config.save_pretrained("config")
