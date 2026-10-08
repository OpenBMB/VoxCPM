import torch

from voxcpm.modules.minicpm4.config import MiniCPM4Config, RopeScalingConfig
from voxcpm.modules.minicpm4.model import MiniCPMAttention

BATCH_SIZE = 1
HIDDEN_SIZE = 8
INTERMEDIATE_SIZE = 16
MAX_CACHE_LENGTH = 8
NUM_ATTENTION_HEADS = 2
NUM_HIDDEN_LAYERS = 1
NUM_KEY_VALUE_HEADS = 1
POSITION_ID = 2
ACTIVE_CACHE_LENGTH = POSITION_ID + 1
RMS_NORM_EPS = 1e-6
ROPE_FACTOR = 1.0
ROPE_THETA = 10_000.0
VOCAB_SIZE = 16


def test_forward_step_only_attends_to_populated_cache(monkeypatch):
    config = MiniCPM4Config(
        bos_token_id=0,
        eos_token_id=1,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=MAX_CACHE_LENGTH,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        rms_norm_eps=RMS_NORM_EPS,
        rope_scaling=RopeScalingConfig(
            type="longrope",
            long_factor=[ROPE_FACTOR] * (HIDDEN_SIZE // NUM_ATTENTION_HEADS // 2),
            short_factor=[ROPE_FACTOR] * (HIDDEN_SIZE // NUM_ATTENTION_HEADS // 2),
            original_max_position_embeddings=MAX_CACHE_LENGTH,
        ),
        vocab_size=VOCAB_SIZE,
        scale_emb=ROPE_FACTOR,
        dim_model_base=HIDDEN_SIZE,
        scale_depth=ROPE_FACTOR,
        rope_theta=ROPE_THETA,
    )
    attention = MiniCPMAttention(config, layer_idx=0)
    hidden_states = torch.randn(BATCH_SIZE, HIDDEN_SIZE)
    head_dim = HIDDEN_SIZE // NUM_ATTENTION_HEADS
    key_cache = torch.zeros(BATCH_SIZE, NUM_KEY_VALUE_HEADS, MAX_CACHE_LENGTH, head_dim)
    value_cache = torch.zeros_like(key_cache)
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    observed = {}

    def record_cache_length(query, key, value, **kwargs):
        observed["key_length"] = key.size(-2)
        observed["value_length"] = value.size(-2)
        observed["has_mask"] = "attn_mask" in kwargs
        return original_sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", record_cache_length)

    attention.forward_step(hidden_states, None, POSITION_ID, (key_cache, value_cache))

    assert observed == {
        "key_length": ACTIVE_CACHE_LENGTH,
        "value_length": ACTIVE_CACHE_LENGTH,
        "has_mask": False,
    }
