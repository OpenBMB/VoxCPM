import torch
import torch.nn as nn


def require_quant_scale(scale: int) -> int:
    """Reject the scale that divides the rounded latent by zero."""
    if scale == 0:
        raise ValueError("ScalarQuantizationLayer scale must be non-zero")
    return scale


class ScalarQuantizationLayer(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim: int = 64, scale: int = 9):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.scale = require_quant_scale(scale)

        self.in_proj = nn.Linear(in_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, out_dim)

    def forward(self, hidden):
        hidden = self.in_proj(hidden)
        hidden = torch.tanh(hidden)

        if self.training:
            quantized = torch.round(hidden * self.scale) / self.scale
            hidden = hidden + (quantized - hidden).detach()
        else:
            hidden = torch.round(hidden * self.scale) / self.scale

        return self.out_proj(hidden)
