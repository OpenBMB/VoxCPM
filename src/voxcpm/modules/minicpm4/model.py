from .config import MiniCPM4Config
import torch
import torch.nn as nn
from typing import List, Tuple
import math
from .cache import StaticKVCache


def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class MiniCPMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MiniCPMRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Args:
        q: Tensor(batch_size, num_heads, seq_len, head_dim)
        k: Tensor(batch_size, num_key_value_heads, seq_len, head_dim)
        cos: Tensor(seq_len, head_dim)
        sin: Tensor(seq_len, head_dim)
    Returns:
        Tensor(batch_size, num_heads, seq_len, head_dim), Tensor(batch_size, num_key_value_heads, seq_len, head_dim)
    """
    orig_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class MiniCPMLongRoPE(nn.Module):
    """MiniCPMRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, config: MiniCPM4Config):
        super().__init__()
        self.config = config
        self.dim = config.kv_channels if config.kv_channels else config.hidden_size // config.num_attention_heads
        self.base = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings

        self.short_factor = config.rope_scaling.short_factor
        self.long_factor = config.rope_scaling.long_factor
        self.original_max_position_embeddings = config.rope_scaling.original_max_position_embeddings

        scale = self.max_position_embeddings / self.original_max_position_embeddings
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.max_seq_len_cached = 0

        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

        self._set_cos_sin_cache(seq_len=self.max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """设置cos和sin缓存"""
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=device)

        freqs = torch.mul(
            torch.outer(t, 1.0 / ext_factors).to(device=device), self.inv_freq.to(device=device).to(dtype)
        )

        # 创建embeddings
        emb = torch.cat((freqs, freqs), dim=-1)

        self.cos_cached = emb.cos().to(dtype) * self.scaling_factor
        self.sin_cached = emb.sin().to(dtype) * self.scaling_factor

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            position_ids: Tensor(seq_len) 或 Tensor(batch_size, seq_len)
        Returns:
            Tensor(seq_len, head_dim), Tensor(seq_len, head_dim)
        """
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]

        return cos, sin


class MiniCPMAttention(nn.Module):
    def __init__(self, config: MiniCPM4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = 10000.0

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_emb: Tuple[torch.Tensor, torch.Tensor],
        is_causal: bool,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_emb is not None:
            cos, sin = position_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ref: https://github.com/pytorch/pytorch/issues/163597
        # there is a bug in MPS for non-contiguous tensors, so we need to make them contiguous
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=is_causal,
            enable_gqa=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        past_key_value = (key_states, value_states)
        return attn_output, past_key_value

    def forward_step(
        self,
        hidden_states: torch.Tensor,
        position_emb: Tuple[torch.Tensor, torch.Tensor],
        position_id: int,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_emb is not None:
            cos, sin = position_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_cache, value_cache = kv_cache

        key_cache[:, :, position_id, :] = key_states
        value_cache[:, :, position_id, :] = value_states

        # Slice KV cache to actual sequence length instead of attending over the
        # full static cache (max_length=8192). This avoids wasting compute on
        # empty positions and allows SDPA to select the flash attention kernel
        # (which does not support arbitrary attn_mask).
        seq_end = position_id + 1
        k_slice = key_cache[:, :, :seq_end, :].contiguous()
        v_slice = value_cache[:, :, :seq_end, :].contiguous()

        # ref: https://github.com/pytorch/pytorch/issues/163597
        # there is a bug in MPS for non-contiguous tensors, so we need to make them contiguous
        query_states = query_states.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            k_slice,
            v_slice,
            is_causal=False,  # q_len=1, no causal mask needed
            enable_gqa=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class MiniCPMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MiniCPMDecoderLayer(nn.Module):
    def __init__(self, config: MiniCPM4Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MiniCPMAttention(config=config, layer_idx=layer_idx)

        self.mlp = MiniCPMMLP(config)
        self.input_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers
        self.use_mup = config.use_mup

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_emb: Tuple[torch.Tensor, torch.Tensor],
        is_causal: bool,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            position_ids (`torch.LongTensor`): position ids of shape `(batch_size, seq_len)`
            is_causal (`bool`): whether the attention mask is causal
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_emb=position_emb,
            is_causal=is_causal,
        )

        if self.use_mup:
            hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        else:
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        if self.use_mup:
            hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        else:
            hidden_states = residual + hidden_states

        return hidden_states, present_key_value

    def forward_step(
        self,
        hidden_states: torch.Tensor,
        position_emb: Tuple[torch.Tensor, torch.Tensor],
        position_id: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn.forward_step(
            hidden_states=hidden_states,
            position_emb=position_emb,
            position_id=position_id,
            kv_cache=kv_cache,
        )

        if self.use_mup:
            hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        else:
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        if self.use_mup:
            hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        else:
            hidden_states = residual + hidden_states

        return hidden_states


class MiniCPMModel(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MiniCPMDecoderLayer`]

    Args:
        config: MiniCPMConfig
    """

    def __init__(self, config: MiniCPM4Config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.config = config

        if config.vocab_size > 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = nn.Identity()

        self.layers = nn.ModuleList(
            [MiniCPMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.no_rope:
            self.rope_emb = None
        else:
            self.rope_emb = MiniCPMLongRoPE(config)

        self.kv_cache = None

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            inputs_embeds: Tensor(batch_size, seq_length, hidden_size)
            is_causal: bool, whether the attention mask is causal
        Returns:
            hidden_states: Tensor(batch_size, seq_length, hidden_size)
            next_decoder_cache: List[(batch_size, num_heads, seq_length, head_dim), (batch_size, num_heads, seq_length, head_dim)]
        """
        if self.rope_emb is not None:
            position_ids = torch.arange(0, inputs_embeds.size(1), dtype=torch.long, device=inputs_embeds.device)
            position_emb = self.rope_emb(position_ids)
        else:
            position_emb = None
        hidden_states = inputs_embeds

        next_decoder_cache = []

        for decoder_layer in self.layers:

            hidden_states, this_cache = decoder_layer(
                hidden_states,
                position_emb,
                is_causal,
            )
            next_decoder_cache.append(this_cache)
        hidden_states = self.norm(hidden_states)
        return hidden_states, next_decoder_cache

    def forward_step(
        self,
        inputs_embeds: torch.Tensor,
        position_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs_embeds: Tensor(batch_size, hidden_size)
        Returns:
            hidden_states: Tensor(batch_size, hidden_size)
        """
        assert self.kv_cache is not None, "KV cache is not setup"

        if self.rope_emb is not None:
            position_emb = self.rope_emb(position_id)
        else:
            position_emb = None
        hidden_states = inputs_embeds

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer.forward_step(
                hidden_states,
                position_emb,
                position_id,
                self.kv_cache.get_layer_cache(i),
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def setup_cache(self, batch_size: int, max_length: int, device, dtype: torch.dtype):
        self.kv_cache = StaticKVCache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            dim_kv_head=(
                self.config.hidden_size // self.config.num_attention_heads
                if self.config.kv_channels is None
                else self.config.kv_channels
            ),
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            max_length=max_length,
        )
