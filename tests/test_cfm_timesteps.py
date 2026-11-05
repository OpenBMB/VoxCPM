from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    torch = types.ModuleType("torch")
    torch.Tensor = object

    class Module:
        def __init__(self, *args, **kwargs):
            pass

    nn = types.ModuleType("torch.nn")
    nn.Module = Module
    torch.nn = nn

    def inference_mode(*args, **kwargs):
        def decorate(fn):
            return fn

        return decorate

    torch.inference_mode = inference_mode
    functional = types.ModuleType("torch.nn.functional")
    func = types.ModuleType("torch.func")
    func.jvp = lambda *args, **kwargs: None
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = functional
    sys.modules["torch.func"] = func

    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        pass

    pydantic.BaseModel = BaseModel
    sys.modules["pydantic"] = pydantic

    package = types.ModuleType("voxcpm")
    package.__path__ = [str(ROOT / "src" / "voxcpm")]
    package.__package__ = "voxcpm"
    modules = types.ModuleType("voxcpm.modules")
    modules.__path__ = [str(ROOT / "src" / "voxcpm" / "modules")]
    modules.__package__ = "voxcpm.modules"
    locdit = types.ModuleType("voxcpm.modules.locdit")
    locdit.__path__ = [str(ROOT / "src" / "voxcpm" / "modules" / "locdit")]
    locdit.__package__ = "voxcpm.modules.locdit"
    local_dit = types.ModuleType("voxcpm.modules.locdit.local_dit")
    local_dit.VoxCPMLocDiT = object
    sys.modules["voxcpm"] = package
    sys.modules["voxcpm.modules"] = modules
    sys.modules["voxcpm.modules.locdit"] = locdit
    sys.modules["voxcpm.modules.locdit.local_dit"] = local_dit

    spec = importlib.util.spec_from_file_location(
        "voxcpm.modules.locdit.unified_cfm",
        ROOT / "src" / "voxcpm" / "modules" / "locdit" / "unified_cfm.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["voxcpm.modules.locdit.unified_cfm"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_zero_timesteps_are_rejected():
    cfm = _load()
    try:
        cfm.require_positive_timesteps(0)
    except ValueError as exc:
        assert "at least 1" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_one_timestep_is_accepted():
    cfm = _load()
    assert cfm.require_positive_timesteps(1) == 1
