from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(module_name: str, relative: str):
    torch = types.ModuleType("torch")
    torch.Tensor = object

    class Module:
        def __init__(self, *args, **kwargs):
            pass

    class Conv1d(Module):
        pass

    class ConvTranspose1d(Conv1d):
        pass

    nn = types.ModuleType("torch.nn")
    nn.Module = Module
    nn.Conv1d = Conv1d
    nn.ConvTranspose1d = ConvTranspose1d
    nn.Sequential = Module
    nn.Parameter = lambda value: value
    nn.Linear = Module
    nn.Identity = Module
    functional = types.ModuleType("torch.nn.functional")
    functional.pad = lambda tensor, pad: tensor
    nn_utils = types.ModuleType("torch.nn.utils")
    nn_utils.weight_norm = lambda module, *args, **kwargs: module
    torch.nn = nn
    torch.jit = types.SimpleNamespace(script=lambda fn: fn)
    torch.randn = lambda *args, **kwargs: None
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = functional
    sys.modules["torch.nn.utils"] = nn_utils

    numpy = types.ModuleType("numpy")
    numpy.prod = lambda values: 1
    sys.modules["numpy"] = numpy

    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        pass

    pydantic.BaseModel = BaseModel
    sys.modules["pydantic"] = pydantic

    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_zero_trim_is_not_a_full_slice():
    vae = _load("audio_vae_trim", "src/voxcpm/modules/audiovae/audio_vae.py")
    vae_v2 = _load("audio_vae_v2_trim", "src/voxcpm/modules/audiovae/audio_vae_v2.py")

    assert vae.causal_trim_length(0, 0) == 0
    assert vae_v2.causal_trim_length(0, 0) == 0
    assert vae.causal_trim_length(1, 1) == 1


def test_negative_trim_is_rejected():
    vae_v2 = _load("audio_vae_v2_trim_negative", "src/voxcpm/modules/audiovae/audio_vae_v2.py")
    try:
        vae_v2.causal_trim_length(0, 1)
    except ValueError as exc:
        assert "output_padding" in str(exc)
    else:
        raise AssertionError("expected ValueError")
