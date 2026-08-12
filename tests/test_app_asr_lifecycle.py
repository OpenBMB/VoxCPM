from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"


class _I18n:
    def __init__(self, **translations):
        self.translations = translations

    def __call__(self, key):
        return key


@pytest.fixture
def app_module(monkeypatch):
    gradio_stub = types.ModuleType("gradio")
    gradio_stub.I18n = _I18n
    gradio_stub.themes = types.SimpleNamespace(
        Soft=lambda **kwargs: kwargs,
        GoogleFont=lambda name: name,
    )
    monkeypatch.setitem(sys.modules, "gradio", gradio_stub)

    funasr_stub = types.ModuleType("funasr")
    funasr_stub.AutoModel = object
    monkeypatch.setitem(sys.modules, "funasr", funasr_stub)

    voxcpm_stub = types.ModuleType("voxcpm")
    voxcpm_stub.VoxCPM = object
    monkeypatch.setitem(sys.modules, "voxcpm", voxcpm_stub)
    model_stub = types.ModuleType("voxcpm.model")
    monkeypatch.setitem(sys.modules, "voxcpm.model", model_stub)
    utils_stub = types.ModuleType("voxcpm.model.utils")
    utils_stub.resolve_runtime_device = lambda requested, default: "cuda:0"
    monkeypatch.setitem(sys.modules, "voxcpm.model.utils", utils_stub)

    spec = importlib.util.spec_from_file_location("voxcpm_demo_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("generate_error", [None, RuntimeError("ASR failed")])
def test_prompt_recognition_releases_cuda_asr_model(app_module, monkeypatch, generate_error):
    empty_cache_calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))

    class _FakeAutoModel:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            if generate_error is not None:
                raise generate_error
            return [{"text": "<|zh|>测试文本"}]

    monkeypatch.setattr(app_module, "AutoModel", _FakeAutoModel)
    demo = app_module.VoxCPMDemo()

    if generate_error is None:
        assert demo.prompt_wav_recognition("reference.wav") == "测试文本"
    else:
        with pytest.raises(RuntimeError, match="ASR failed"):
            demo.prompt_wav_recognition("reference.wav")

    assert demo.asr_model is None
    assert empty_cache_calls == [True]
