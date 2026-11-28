from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = ROOT / "src" / "voxcpm" / "model" / "utils.py"

transformers_stub = types.ModuleType("transformers")
transformers_stub.PreTrainedTokenizer = object
sys.modules.setdefault("transformers", transformers_stub)

spec = importlib.util.spec_from_file_location("voxcpm.model.utils", UTILS_PATH)
utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(utils)


def test_resolve_runtime_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(utils, "_has_mps", lambda: False)
    monkeypatch.setattr(utils, "_has_xpu", lambda: False)

    assert utils.resolve_runtime_device(None, "cuda") == "cpu"


def test_resolve_runtime_device_auto_uses_mps_when_available(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(utils, "_has_mps", lambda: True)
    monkeypatch.setattr(utils, "_has_xpu", lambda: False)

    assert utils.resolve_runtime_device("auto", "cuda") == "mps"


def test_resolve_runtime_device_respects_explicit_cpu(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(utils, "_has_mps", lambda: True)
    monkeypatch.setattr(utils, "_has_xpu", lambda: False)

    assert utils.resolve_runtime_device("cpu", "cuda") == "cpu"


def test_resolve_runtime_device_rejects_unavailable_explicit_cuda(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(utils, "_has_mps", lambda: True)
    monkeypatch.setattr(utils, "_has_xpu", lambda: False)

    with pytest.raises(ValueError, match="CUDA is not available"):
        utils.resolve_runtime_device("cuda:0", "cuda")


def test_materialize_generation_seed_preserves_explicit_seed():
    assert utils.materialize_generation_seed(42) == 42


def test_materialize_generation_seed_creates_concrete_seed_for_none(monkeypatch):
    monkeypatch.setattr(utils.torch, "seed", lambda: 0x123456789)

    assert utils.materialize_generation_seed(None) == 0x23456789


def test_apply_generation_seed_sets_cpu_and_cuda_rng(monkeypatch):
    calls = []

    monkeypatch.setattr(utils.torch, "manual_seed", lambda seed: calls.append(("cpu", seed)))
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(utils.torch.cuda, "manual_seed_all", lambda seed: calls.append(("cuda", seed)))
    monkeypatch.setattr(utils, "_has_xpu", lambda: False)

    utils.apply_generation_seed(123)

    assert calls == [("cpu", 123), ("cuda", 123)]


def test_resolve_runtime_device_auto_uses_xpu_when_available(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(utils, "_has_xpu", lambda: True)
    monkeypatch.setattr(utils, "_has_mps", lambda: True)

    assert utils.resolve_runtime_device("auto", "cuda") == "xpu"


def test_resolve_runtime_device_auto_prefers_cuda_over_xpu(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(utils, "_has_xpu", lambda: True)
    monkeypatch.setattr(utils, "_has_mps", lambda: False)

    assert utils.resolve_runtime_device(None, "cuda") == "cuda"


def test_resolve_runtime_device_accepts_explicit_xpu(monkeypatch):
    monkeypatch.setattr(utils, "_has_xpu", lambda: True)

    assert utils.resolve_runtime_device("xpu", "cuda") == "xpu"
    assert utils.resolve_runtime_device("xpu:1", "cuda") == "xpu:1"


def test_resolve_runtime_device_rejects_unavailable_explicit_xpu(monkeypatch):
    monkeypatch.setattr(utils, "_has_xpu", lambda: False)

    with pytest.raises(ValueError, match="XPU is not available"):
        utils.resolve_runtime_device("xpu", "cuda")
    with pytest.raises(ValueError, match="XPU is not available"):
        utils.resolve_runtime_device("xpu:0", "cuda")


def test_has_xpu_checks_torch_xpu_availability(monkeypatch):
    monkeypatch.setattr(utils.torch.xpu, "is_available", lambda: True, raising=False)
    assert utils._has_xpu() is True

    monkeypatch.setattr(utils.torch.xpu, "is_available", lambda: False, raising=False)
    assert utils._has_xpu() is False


def test_apply_generation_seed_sets_xpu_rng(monkeypatch):
    calls = []

    monkeypatch.setattr(utils.torch, "manual_seed", lambda seed: calls.append(("cpu", seed)))
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(utils, "_has_xpu", lambda: True)
    monkeypatch.setattr(utils.torch.xpu, "manual_seed_all", lambda seed: calls.append(("xpu", seed)), raising=False)

    utils.apply_generation_seed(7)

    assert calls == [("cpu", 7), ("xpu", 7)]
