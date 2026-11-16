from pydantic import BaseModel
from typing import List, Optional


class RopeScalingConfig(BaseModel):
    type: str
    long_factor: List[float]
    short_factor: List[float]
    original_max_position_embeddings: int


class MiniCPM4Config(BaseModel):
    bos_token_id: int
    eos_token_id: int
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_scaling: RopeScalingConfig
    vocab_size: int
    use_mup: bool = True
    scale_emb: float
    dim_model_base: int
    scale_depth: float
    rope_theta: float
    kv_channels: Optional[int] = None
    no_rope: bool = False
