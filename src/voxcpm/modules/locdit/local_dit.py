import torch
from typing import Optional
from ..minicpm4 import MiniCPMModel, MiniCPM4Config
import torch.nn as nn
import math


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=x.dtype, device=device) * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: Optional[int] = None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim

        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, bias=True)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class VoxCPMLocDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        config: MiniCPM4Config,
        in_channels: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.config = config

        self.in_proj = nn.Linear(in_channels, config.hidden_size, bias=True)
        self.cond_proj = nn.Linear(in_channels, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, self.out_channels, bias=True)

        self.time_embeddings = SinusoidalPosEmb(config.hidden_size)
        self.time_mlp = TimestepEmbedding(
            in_channels=config.hidden_size,
            time_embed_dim=config.hidden_size,
        )
        self.delta_time_mlp = TimestepEmbedding(
            in_channels=config.hidden_size,
            time_embed_dim=config.hidden_size,
        )

        assert config.vocab_size == 0, "vocab_size must be 0 for local DiT"
        self.decoder = MiniCPMModel(config)

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        dt: torch.Tensor,
    ):
        """
        Forward pass of DiT.
        x: (N, C, T) tensor of inputs
        mu: (N, C) tensor of hidden embedding
        t: (N,) tensor of diffusion timesteps
        cond: (N, C, T') tensor of prefix conditions
        dt: (N,) used for mean velocity (may be supported in the future...)
        """
        x = self.in_proj(x.transpose(1, 2).contiguous())

        cond = self.cond_proj(cond.transpose(1, 2).contiguous())
        prefix = cond.size(1)

        t = self.time_embeddings(t).to(x.dtype)
        t = self.time_mlp(t)
        dt = self.time_embeddings(dt).to(x.dtype)
        dt = self.delta_time_mlp(dt)
        t = t + dt

        x = torch.cat([(mu + t).unsqueeze(1), cond, x], dim=1)
        hidden, _ = self.decoder(x, is_causal=False)
        hidden = hidden[:, prefix + 1 :, :]
        hidden = self.out_proj(hidden)

        return hidden.transpose(1, 2).contiguous()
