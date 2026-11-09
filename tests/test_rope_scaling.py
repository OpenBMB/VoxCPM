from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.float32 = object
    torch.dtype = object

    class Module:
        def __init__(self, *args, **kwargs):
            pass

    nn = types.ModuleType("torch.nn")
    nn.Module = Module
    nn.Linear = Module
    nn.Parameter = lambda value: value
    functional = types.ModuleType("torch.nn.functional")
    torch.nn = nn
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = functional

    package = types.ModuleType("voxcpm")
    package.__path__ = [str(ROOT / "src" / "voxcpm")]
    package.__package__ = "voxcpm"
    modules = types.ModuleType("voxcpm.modules")
    modules.__path__ = [str(ROOT / "src" / "voxcpm" / "modules")]
    modules.__package__ = "voxcpm.modules"
    minicpm4 = types.ModuleType("voxcpm.modules.minicpm4")
    minicpm4.__path__ = [str(ROOT / "src" / "voxcpm" / "modules" / "minicpm4")]
    minicpm4.__package__ = "voxcpm.modules.minicpm4"
    config = types.ModuleType("voxcpm.modules.minicpm4.config")
    config.MiniCPM4Config = object
    cache = types.ModuleType("voxcpm.modules.minicpm4.cache")
    cache.StaticKVCache = object
    sys.modules["voxcpm"] = package
    sys.modules["voxcpm.modules"] = modules
    sys.modules["voxcpm.modules.minicpm4"] = minicpm4
    sys.modules["voxcpm.modules.minicpm4.config"] = config
    sys.modules["voxcpm.modules.minicpm4.cache"] = cache

    spec = importlib.util.spec_from_file_location(
        "voxcpm.modules.minicpm4.model",
        ROOT / "src" / "voxcpm" / "modules" / "minicpm4" / "model.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["voxcpm.modules.minicpm4.model"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_original_context_of_one_is_rejected():
    model = _load()
    try:
        model.rope_scaling_factor(32768, 1)
    except ValueError as exc:
        assert "greater than 1" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_matching_context_lengths_scale_to_one():
    model = _load()
    assert model.rope_scaling_factor(32768, 32768) == 1.0
