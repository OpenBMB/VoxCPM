from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

torch = types.ModuleType("torch")
torch.Tensor = object

class _Module:
    def __init__(self, *args, **kwargs):
        pass

nn = types.ModuleType("torch.nn")
nn.Module = _Module
nn.Linear = _Module
nn.Parameter = lambda value: value
functional = types.ModuleType("torch.nn.functional")
torch.nn = nn
sys.modules["torch"] = torch
sys.modules["torch.nn"] = nn
sys.modules["torch.nn.functional"] = functional

spec = importlib.util.spec_from_file_location(
    "lora_layer",
    ROOT / "src" / "voxcpm" / "modules" / "layers" / "lora.py",
)
lora = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(lora)


def test_negative_rank_is_rejected():
    try:
        lora.require_lora_rank(-1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_zero_rank_stays_a_disabled_adapter():
    assert lora.require_lora_rank(0) == 0
