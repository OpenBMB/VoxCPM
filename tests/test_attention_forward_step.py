from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = ROOT / "src" / "voxcpm"

for name, path in [
    ("voxcpm", PKG_DIR),
    ("voxcpm.modules", PKG_DIR / "modules"),
    ("voxcpm.modules.minicpm4", PKG_DIR / "modules" / "minicpm4"),
]:
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.__path__ = [str(path)]
        sys.modules[name] = stub

model = importlib.import_module("voxcpm.modules.minicpm4.model")
MiniCPMAttention = model.MiniCPMAttention

MAX_LEN = 256


def _make_config():
    cfg = types.SimpleNamespace(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=2,
        kv_channels=16,
        max_position_embeddings=32768,
    )
    return cfg


def _reference_full_cache(attn, hidden_states, position_id, key_cache, value_cache):
    """Old semantics: SDPA over the full cache with an explicit causal mask."""
    bsz, _ = hidden_states.size()
    query_states = attn.q_proj(hidden_states)
    key_states = attn.k_proj(hidden_states)
    value_states = attn.v_proj(hidden_states)
    query_states = query_states.view(bsz, 1, attn.num_heads, attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, 1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, 1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    key_cache[:, :, position_id, :] = key_states
    value_cache[:, :, position_id, :] = value_states
    attn_mask = (torch.arange(key_cache.size(2)) <= int(position_id)).view(1, 1, 1, -1)
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        attn_mask=attn_mask,
        enable_gqa=True,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, attn.num_heads * attn.head_dim)
    return attn.o_proj(attn_output)


@pytest.mark.parametrize("position_id", [torch.tensor([7])])
def test_forward_step_matches_full_cache_masked_attention(position_id):
    torch.manual_seed(0)
    attn = MiniCPMAttention(_make_config(), layer_idx=0).float().eval()
    bsz = 1
    hidden_states = torch.randn(bsz, 128)
    key_cache = torch.zeros(bsz, 2, MAX_LEN, attn.head_dim)
    value_cache = torch.zeros_like(key_cache)
    # fill a short prefix so the cache is mostly empty, as during decoding
    for p in range(position_id if isinstance(position_id, int) else 8):
        key_cache[:, :, p, :] = attn.k_proj(hidden_states).view(bsz, 2, attn.head_dim)
        value_cache[:, :, p, :] = attn.v_proj(hidden_states).view(bsz, 2, attn.head_dim)

    expected = _reference_full_cache(attn, hidden_states, position_id, key_cache, value_cache)

    key_cache_step = key_cache.clone()
    value_cache_step = value_cache.clone()
    with torch.no_grad():
        actual = attn.forward_step(hidden_states, None, position_id, (key_cache_step, value_cache_step))

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)
    # the prefix beyond position_id must stay untouched
    assert torch.equal(key_cache_step[:, :, position_id + 1 :, :], torch.zeros_like(key_cache_step[:, :, position_id + 1 :, :]))


def test_forward_step_attends_only_to_the_populated_prefix():
    """Keys written after the query position must not influence the output."""
    torch.manual_seed(0)
    attn = MiniCPMAttention(_make_config(), layer_idx=0).float().eval()
    hidden_states = torch.randn(1, 128)
    position_id = torch.tensor([5])

    key_cache = torch.zeros(1, 2, MAX_LEN, attn.head_dim)
    value_cache = torch.zeros_like(key_cache)
    with torch.no_grad():
        out = attn.forward_step(hidden_states, None, position_id, (key_cache, value_cache))

    # garbage in the far tail (as if a previous session left data there)
    key_cache[:, :, 100:, :] = torch.randn_like(key_cache[:, :, 100:, :])
    value_cache[:, :, 100:, :] = torch.randn_like(value_cache[:, :, 100:, :])
    with torch.no_grad():
        out_tail = attn.forward_step(hidden_states, None, position_id, (key_cache, value_cache))

    assert torch.allclose(out, out_tail, rtol=1e-5, atol=1e-6)
