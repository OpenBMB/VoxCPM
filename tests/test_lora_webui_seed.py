from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def webui(monkeypatch):
    for name in ("gradio", "funasr", "voxcpm.core", "voxcpm.model.voxcpm"):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("lora_ft_webui_under_test", ROOT / "lora_ft_webui.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = MagicMock()
    model.tts_model.sample_rate = 16000
    monkeypatch.setattr(module, "current_model", model)
    return module, model


def _run(module, seed):
    return module.run_inference("hello", None, None, "None", 2.0, 10, seed)


def test_fixed_seed_is_passed_to_generate(webui):
    module, model = webui
    _, status = _run(module, 1234)
    assert status == "Generation Success"
    assert model.generate.call_args.kwargs["seed"] == 1234


@pytest.mark.parametrize("seed", [-1, None, ""])
def test_random_seed_is_not_forced(webui, seed):
    module, model = webui
    _run(module, seed)
    assert model.generate.call_args.kwargs["seed"] is None
