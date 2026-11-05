"""Behavioral coverage for fixed-capacity, configurable decode attention."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def runtime(monkeypatch):
    # Load the real lightweight modules without importing the TTS pipeline or
    # sharing sys.modules stubs with the CLI/checkpoint-loading tests.
    root = Path(__file__).resolve().parents[1] / "src" / "voxcpm"
    name = "_voxcpm_runtime_test"
    package = types.ModuleType(name)
    package.__path__ = [str(root / "modules" / "minicpm4")]
    monkeypatch.setitem(sys.modules, name, package)

    def load(module_name, path):
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
        return module

    config = load(f"{name}.config", root / "modules" / "minicpm4" / "config.py")
    cache = load(f"{name}.cache", root / "modules" / "minicpm4" / "cache.py")
    model = load(f"{name}.model", root / "modules" / "minicpm4" / "model.py")
    utils = load(f"{name}.utils", root / "model" / "utils.py")
    return types.SimpleNamespace(config=config, cache=cache, model=model, utils=utils)


def tiny_config(runtime):
    return runtime.config.MiniCPM4Config(
        bos_token_id=0,
        eos_token_id=1,
        hidden_size=32,
        intermediate_size=64,
        max_position_embeddings=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_scaling=runtime.config.RopeScalingConfig(
            type="longrope",
            long_factor=[1.0] * 4,
            short_factor=[1.0] * 4,
            original_max_position_embeddings=64,
        ),
        vocab_size=64,
        scale_emb=1.0,
        dim_model_base=32,
        scale_depth=1.0,
        rope_theta=10000.0,
    )


@pytest.mark.parametrize("backend", ["auto", "math"])
def test_decode_matches_full_prefix_with_different_cache_capacities(runtime, backend):
    torch.manual_seed(41)
    model = runtime.model.MiniCPMModel(tiny_config(runtime), attention_backend=backend).eval()
    inputs = torch.randn(1, 7, 32)
    with torch.inference_mode():
        expected, _ = model(inputs)
        outputs = []
        for capacity in (7, 32):
            model.setup_cache(1, capacity, "cpu", torch.float32)
            _, prefix = model(inputs[:, :3])
            model.kv_cache.fill_caches(prefix)
            # Unused positions must remain masked even if they contain old data.
            model.kv_cache.kv_cache[:, :, :, :, 3:].fill_(100.0)
            for position in range(3, inputs.size(1)):
                actual = model.forward_step(inputs[:, position], torch.tensor([model.kv_cache.step()]))
                reference, _ = model(inputs[:, : position + 1])
                torch.testing.assert_close(actual, reference[:, -1], rtol=1e-5, atol=1e-6)
            outputs.append(actual)
        torch.testing.assert_close(outputs[0], expected[:, -1], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)


def test_decode_cache_reuse_discards_previous_request(runtime):
    torch.manual_seed(17)
    model = runtime.model.MiniCPMModel(tiny_config(runtime), attention_backend="math").eval()
    model.setup_cache(1, 12, "cpu", torch.float32)
    with torch.inference_mode():
        _, first = model(torch.randn(1, 9, 32))
        model.kv_cache.fill_caches(first)
        second = torch.randn(1, 4, 32)
        _, prefix = model(second[:, :3])
        model.kv_cache.fill_caches(prefix)
        actual = model.forward_step(second[:, -1], torch.tensor([model.kv_cache.step()]))
        expected, _ = model(second)
    torch.testing.assert_close(actual, expected[:, -1], rtol=1e-5, atol=1e-6)


def test_prefix_overflow_does_not_destroy_previous_cache(runtime):
    cache = runtime.cache.StaticKVCache(1, 2, 4, 1, "cpu", torch.float32, max_length=3)
    valid = [(torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4))]
    cache.fill_caches(valid)
    previous = cache.kv_cache.clone()
    oversized = [(torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4))]
    with pytest.raises(ValueError, match="max_cache_length"):
        cache.fill_caches(oversized)
    assert cache.current_length == 2
    torch.testing.assert_close(cache.kv_cache, previous, rtol=0, atol=0)
    assert cache.step() == 2
    with pytest.raises(ValueError, match="max_cache_length"):
        cache.step()
    assert cache.current_length == 3


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5, "2048"])
def test_invalid_cache_capacity_is_rejected(runtime, capacity):
    with pytest.raises(ValueError, match="max_cache_length"):
        runtime.utils.validate_inference_options("auto", capacity)


@pytest.mark.parametrize("backend,device", [("unknown", "cpu"), ("cudnn", "cpu"), ("cudnn", "mps")])
def test_unavailable_backend_is_not_silently_replaced(runtime, backend, device):
    with pytest.raises(ValueError, match="attention_backend|CUDA"):
        runtime.utils.validate_inference_options(backend, None, device=device)


@pytest.mark.parametrize("backend", ["cudnn", "math"])
def test_explicit_compiled_backend_rejects_unsupported_torch(runtime, monkeypatch, backend):
    monkeypatch.setattr(torch, "__version__", "2.5.1+cu124")
    with pytest.raises(ValueError, match="2.6"):
        runtime.utils.validate_inference_options(backend, None, device="cuda:0", optimize=True)


def test_explicit_math_attention_traces_with_tensor_positions(runtime):
    if tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 6):
        pytest.skip("Compiled explicit SDPA selection requires torch >= 2.6")
    torch.manual_seed(7)
    model = runtime.model.MiniCPMModel(tiny_config(runtime), attention_backend="math").eval()
    model.setup_cache(1, 12, "cpu", torch.float32)
    inputs = torch.randn(1, 5, 32)
    with torch.inference_mode():
        _, prefix = model(inputs[:, :2])
        model.kv_cache.fill_caches(prefix)
        compiled = torch.compile(model.forward_step, backend="eager", fullgraph=True)
        for position in range(2, 5):
            actual = compiled(inputs[:, position], torch.tensor([model.kv_cache.step()]))
            expected, _ = model(inputs[:, : position + 1])
            torch.testing.assert_close(actual, expected[:, -1], rtol=1e-5, atol=1e-6)
