from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
torch = types.ModuleType("torch")
torch.Tensor = object
torch.device = object
torch.dtype = object
sys.modules.setdefault("torch", torch)

spec = importlib.util.spec_from_file_location(
    "voxcpm.modules.minicpm4.cache",
    ROOT / "src" / "voxcpm" / "modules" / "minicpm4" / "cache.py",
)
cache = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cache)


class _Tensor:
    def __init__(self, seq_len: int):
        self._seq_len = seq_len

    def size(self, dim: int) -> int:
        assert dim == 2
        return self._seq_len


def test_prompt_longer_than_the_buffer_is_rejected():
    caches = [(_Tensor(9000), _Tensor(9000))]
    try:
        cache._prompt_length(caches, max_length=8192)
    except ValueError as exc:
        assert "9000" in str(exc)
        assert "8192" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_empty_fill_is_rejected():
    try:
        cache._prompt_length([], max_length=8192)
    except ValueError as exc:
        assert "at least one layer" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_fitting_prompt_returns_its_length():
    assert cache._prompt_length([(_Tensor(12), _Tensor(12))], max_length=8192) == 12
