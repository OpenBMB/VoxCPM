from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(module_name: str, relative: str):
    torch = types.ModuleType("torch")

    class Module:
        def __init__(self, *args, **kwargs):
            pass

    nn = types.ModuleType("torch.nn")
    nn.Module = Module
    nn.Linear = Module
    torch.nn = nn
    torch.Tensor = object
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn

    package = types.ModuleType("voxcpm")
    package.__path__ = [str(ROOT / "src" / "voxcpm")]
    package.__package__ = "voxcpm"
    modules = types.ModuleType("voxcpm.modules")
    modules.__path__ = [str(ROOT / "src" / "voxcpm" / "modules")]
    modules.__package__ = "voxcpm.modules"
    locdit = types.ModuleType("voxcpm.modules.locdit")
    locdit.__path__ = [str(ROOT / "src" / "voxcpm" / "modules" / "locdit")]
    locdit.__package__ = "voxcpm.modules.locdit"
    minicpm4 = types.ModuleType("voxcpm.modules.minicpm4")
    minicpm4.MiniCPMModel = object
    minicpm4.MiniCPM4Config = object
    sys.modules["voxcpm"] = package
    sys.modules["voxcpm.modules"] = modules
    sys.modules["voxcpm.modules.locdit"] = locdit
    sys.modules["voxcpm.modules.minicpm4"] = minicpm4

    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_even_width_of_two_is_rejected():
    dit = _load("voxcpm.modules.locdit.local_dit", "src/voxcpm/modules/locdit/local_dit.py")
    dit_v2 = _load("voxcpm.modules.locdit.local_dit_v2", "src/voxcpm/modules/locdit/local_dit_v2.py")
    for module in (dit, dit_v2):
        try:
            module.require_sinusoidal_dim(2)
        except ValueError as exc:
            assert ">= 4" in str(exc)
        else:
            raise AssertionError("expected ValueError")


def test_width_of_four_is_accepted():
    dit = _load("voxcpm.modules.locdit.local_dit_ok", "src/voxcpm/modules/locdit/local_dit.py")
    assert dit.require_sinusoidal_dim(4) == 4
    try:
        dit.require_sinusoidal_dim(3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
