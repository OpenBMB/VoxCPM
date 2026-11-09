from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

torch = types.ModuleType("torch")

class _Module:
    def __init__(self, *args, **kwargs):
        pass

nn = types.ModuleType("torch.nn")
nn.Module = _Module
nn.Linear = _Module
torch.nn = nn
sys.modules["torch"] = torch
sys.modules["torch.nn"] = nn

spec = importlib.util.spec_from_file_location(
    "scalar_quantization_layer",
    ROOT / "src" / "voxcpm" / "modules" / "layers" / "scalar_quantization_layer.py",
)
layer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(layer)


def test_zero_scale_is_rejected():
    try:
        layer.require_quant_scale(0)
    except ValueError as exc:
        assert "non-zero" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_default_scale_is_accepted():
    assert layer.require_quant_scale(9) == 9
