import torch
from einops import rearrange
from torch import nn

from ..minicpm4 import MiniCPM4Config, MiniCPMModel


class VoxCPMLocEnc(nn.Module):
    def __init__(self, config: MiniCPM4Config, input_dim: int = 64):
        super().__init__()
        self.config = config
        self.special_token = nn.Parameter(torch.randn(1, 1, 1, config.hidden_size))
        self.in_proj = nn.Linear(input_dim, config.hidden_size, bias=True)

        assert config.vocab_size == 0, "vocab_size must be 0 for local encoder"
        self.encoder = MiniCPMModel(config)

    def forward(self, x):
        """
        x: [B, T, P, D]
        """
        B, T, P, D = x.shape

        x = self.in_proj(x)
        special_tokens = self.special_token.expand(B, T, 1, -1)
        x = torch.cat([special_tokens, x], dim=2)
        x = rearrange(x, "b t p c -> (b t) p c")
        outputs, _ = self.encoder(x, is_causal=False)
        cls_output = outputs[:, 0, :]

        return rearrange(cls_output, "(b t) c -> b t c", b=B)
