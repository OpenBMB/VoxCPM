from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "src" / "voxcpm" / "core.py"
PKG_NAME = "voxcpm_core_long_form_under_test"


pkg = types.ModuleType(PKG_NAME)
pkg.__path__ = [str(ROOT / "src" / "voxcpm")]
sys.modules[PKG_NAME] = pkg

model_pkg = types.ModuleType(f"{PKG_NAME}.model")
model_pkg.__path__ = [str(ROOT / "src" / "voxcpm" / "model")]
sys.modules[f"{PKG_NAME}.model"] = model_pkg

hf_stub = types.ModuleType("huggingface_hub")
hf_stub.snapshot_download = lambda *args, **kwargs: "/tmp/fake"
sys.modules.setdefault("huggingface_hub", hf_stub)

model_stub = types.ModuleType(f"{PKG_NAME}.model.voxcpm")
model_stub.LoRAConfig = object
model_stub.VoxCPMModel = type("VoxCPMModel", (), {})
sys.modules[f"{PKG_NAME}.model.voxcpm"] = model_stub

model2_stub = types.ModuleType(f"{PKG_NAME}.model.voxcpm2")
model2_stub.VoxCPM2Model = type("VoxCPM2Model", (), {})
sys.modules[f"{PKG_NAME}.model.voxcpm2"] = model2_stub

utils_stub = types.ModuleType(f"{PKG_NAME}.model.utils")
utils_stub.next_and_close = lambda gen: next(gen)
sys.modules[f"{PKG_NAME}.model.utils"] = utils_stub

spec = importlib.util.spec_from_file_location(f"{PKG_NAME}.core", CORE_PATH)
core = importlib.util.module_from_spec(spec)
sys.modules[f"{PKG_NAME}.core"] = core
assert spec.loader is not None
spec.loader.exec_module(core)


class _DummyTTSModel(core.VoxCPM2Model):
    sample_rate = 10


class _FakeVoxCPM(core.VoxCPM):
    def __init__(self):
        self.tts_model = _DummyTTSModel()
        self.calls = []

    def generate(self, text, **kwargs):
        self.calls.append((text, kwargs.copy()))
        return np.full(10, len(self.calls), dtype=np.float32)


def test_generate_long_form_reuses_seed_prompt_with_control_context():
    model = _FakeVoxCPM()

    wav = model.generate_long_form(
        text="First! Second! Third!",
        control="steady narrator",
        max_chars=7,
        silence_ms=100,
        cfg_value=1.5,
        inference_timesteps=4,
    )

    assert len(model.calls) == 3
    assert model.calls[0][0] == "(steady narrator)First!"
    assert model.calls[0][1]["prompt_wav_path"] is None
    assert model.calls[0][1]["reference_wav_path"] is None

    assert model.calls[1][0] == "Second!"
    assert model.calls[1][1]["prompt_text"] == "(steady narrator)First!"
    assert model.calls[1][1]["prompt_wav_path"]
    assert model.calls[1][1]["reference_wav_path"]

    assert model.calls[2][0] == "Third!"
    assert model.calls[2][1]["prompt_text"] == "(steady narrator)First!"
    assert model.calls[2][1]["prompt_wav_path"] == model.calls[1][1]["prompt_wav_path"]
    assert model.calls[2][1]["reference_wav_path"] == model.calls[1][1]["reference_wav_path"]

    assert wav.dtype == np.float32
    assert wav.shape == (32,)


def test_generate_long_form_delegates_single_segment_once():
    model = _FakeVoxCPM()

    wav = model.generate_long_form(
        text="Short!",
        control="steady narrator",
        max_chars=20,
    )

    assert len(model.calls) == 1
    assert model.calls[0][0] == "(steady narrator)Short!"
    assert wav.shape == (10,)
