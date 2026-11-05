"""
Regression test for concurrent batch processing in VoxCPM.

This test verifies that the KV cache properly supports batch_size > 1,
which is required for concurrent TTS request processing (Issue #374).
"""

import torch

from voxcpm.modules.minicpm4.cache import StaticKVCache
from voxcpm.modules.minicpm4.model import MiniCPMModel, MiniCPM4Config
from voxcpm.modules.minicpm4.config import RopeScalingConfig


def _create_test_config() -> MiniCPM4Config:
    """Create a minimal config for testing."""
    rope_scaling = RopeScalingConfig(
        type="linear",
        long_factor=[1.0] * 16,
        short_factor=[1.0] * 16,
        original_max_position_embeddings=512,
    )
    return MiniCPM4Config(
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=256,
        intermediate_size=1024,
        max_position_embeddings=512,
        num_attention_heads=8,
        num_hidden_layers=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_scaling=rope_scaling,
        vocab_size=32000,
        use_mup=True,
        scale_emb=1.0,
        dim_model_base=256,
        scale_depth=1.0,
        rope_theta=10000.0,
        kv_channels=32,
        no_rope=False,
    )


def test_static_kv_cache_batch_support():
    """Test that StaticKVCache supports batch_size > 1 with per-batch positions."""
    cache = StaticKVCache(
        num_layers=2,
        num_kv_heads=4,
        dim_kv_head=64,
        batch_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_length=128,
    )

    # Fill with prefill data for all batches
    kv_caches = [
        (torch.randn(4, 4, 10, 64), torch.randn(4, 4, 10, 64)),
        (torch.randn(4, 4, 10, 64), torch.randn(4, 4, 10, 64)),
    ]
    cache.fill_caches(kv_caches)

    # Verify all batches have the same prefill length
    assert torch.all(cache.current_length == 10).item(), "All batches should have prefill length 10"

    # Step all batches
    positions = cache.step()
    assert torch.all(positions == 10).item(), "All batches should start at position 10"
    assert torch.all(cache.current_length == 11).item(), "All batches should advance to 11"

    # Step individual batch
    pos = cache.step(batch_idx=0)
    assert pos == 11, "Batch 0 should be at position 11"
    assert cache.current_length[0] == 12, "Batch 0 should advance to 12"
    assert torch.all(cache.current_length[1:] == 11).item(), "Other batches should stay at 11"

    # Reset individual batch
    cache.reset(batch_idx=0)
    assert cache.current_length[0] == 0, "Batch 0 should reset to 0"

    # Reset all
    cache.reset()
    assert torch.all(cache.current_length == 0).item(), "All batches should reset to 0"


def test_minicpm_model_batch_forward_step():
    """Test that MiniCPMModel.forward_step works with batch_size > 1."""
    config = _create_test_config()
    model = MiniCPMModel(config)
    model.setup_cache(batch_size=4, max_length=128, device="cpu", dtype=torch.float32)

    inputs_embeds = torch.randn(4, 256)

    # Multiple generation steps using kv_cache.step()
    for step in range(5):
        position_id = model.kv_cache.step()
        with torch.no_grad():
            output = model.forward_step(inputs_embeds, position_id)
        assert output.shape == (4, 256), f"Output shape mismatch at step {step}"
        assert torch.all(model.kv_cache.current_length == step + 1).item(), \
            f"KV cache position mismatch at step {step}"


def test_minicpm_model_legacy_scalar_position():
    """Test that MiniCPMModel.forward_step still works with scalar position_id (legacy).

    In the legacy pattern, caller must call kv_cache.step() first to get position
    and increment the counter, then pass that position to forward_step.
    """
    config = _create_test_config()
    model = MiniCPMModel(config)
    model.setup_cache(batch_size=4, max_length=128, device="cpu", dtype=torch.float32)

    inputs_embeds = torch.randn(4, 256)

    # Legacy usage: call step() to get position, then pass to forward_step
    position_id = model.kv_cache.step()  # This increments current_length
    assert torch.all(position_id == 0).item(), "First step should return position 0"
    assert torch.all(model.kv_cache.current_length == 1).item(), "Counter should advance to 1"

    with torch.no_grad():
        output = model.forward_step(inputs_embeds, position_id)
    assert output.shape == (4, 256), "Output shape mismatch with scalar position"

    # Second step
    position_id = model.kv_cache.step()
    assert torch.all(position_id == 1).item(), "Second step should return position 1"
    assert torch.all(model.kv_cache.current_length == 2).item(), "Counter should advance to 2"

    with torch.no_grad():
        output = model.forward_step(inputs_embeds, position_id)
    assert output.shape == (4, 256), "Output shape mismatch on second step"


def test_per_batch_different_positions():
    """Test that different batches can be at different positions (continuous batching)."""
    config = _create_test_config()
    model = MiniCPMModel(config)
    model.setup_cache(batch_size=4, max_length=128, device="cpu", dtype=torch.float32)

    inputs_embeds = torch.randn(4, 256)

    # Simulate continuous batching: each batch at different position
    # Batch 0: position 5, Batch 1: position 10, Batch 2: position 3, Batch 3: position 8
    position_id = torch.tensor([5, 10, 3, 8])
    with torch.no_grad():
        output = model.forward_step(inputs_embeds, position_id)
    assert output.shape == (4, 256), "Output shape mismatch with per-batch positions"

    # Verify KV cache was written at correct positions for each batch
    key_cache, _ = model.kv_cache.get_layer_cache(0)
    # Check that data was written (non-zero) at the specified positions
    for b, pos in enumerate([5, 10, 3, 8]):
        cache_slice = key_cache[b, :, pos, :]
        assert not torch.allclose(cache_slice, torch.zeros_like(cache_slice)), \
            f"Batch {b} should have data written at position {pos}"