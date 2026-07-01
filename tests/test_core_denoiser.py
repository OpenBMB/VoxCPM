from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "src" / "voxcpm" / "core.py"


class FakeTensor:
    def squeeze(self, dim):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array([0.0], dtype=np.float32)


class DummyVoxCPM2Model:
    sample_rate = 16000

    @classmethod
    def from_local(cls, *args, **kwargs):
        return cls()

    def __init__(self):
        self.prompt_cache_calls = []
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)

    def build_prompt_cache(self, **kwargs):
        self.prompt_cache_calls.append(kwargs)
        return {"prompt_cache": True}

    def _generate_with_prompt_cache(self, **kwargs):
        yield FakeTensor(), None, None


class DummyVoxCPMModel(DummyVoxCPM2Model):
    pass


class DummyLoRAConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _next_and_close(generator):
    try:
        return next(generator)
    finally:
        close = getattr(generator, "close", None)
        if close is not None:
            close()


def load_core_with_stubs(monkeypatch):
    for module_name in [
        "voxcpm",
        "voxcpm.core",
        "voxcpm.model",
        "voxcpm.model.utils",
        "voxcpm.model.voxcpm",
        "voxcpm.model.voxcpm2",
        "voxcpm.zipenhancer",
        "huggingface_hub",
    ]:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    pkg = types.ModuleType("voxcpm")
    pkg.__path__ = [str(ROOT / "src" / "voxcpm")]
    monkeypatch.setitem(sys.modules, "voxcpm", pkg)

    model_pkg = types.ModuleType("voxcpm.model")
    model_pkg.__path__ = [str(ROOT / "src" / "voxcpm" / "model")]
    monkeypatch.setitem(sys.modules, "voxcpm.model", model_pkg)

    utils_stub = types.ModuleType("voxcpm.model.utils")
    utils_stub.next_and_close = _next_and_close
    monkeypatch.setitem(sys.modules, "voxcpm.model.utils", utils_stub)

    v1_stub = types.ModuleType("voxcpm.model.voxcpm")
    v1_stub.VoxCPMModel = DummyVoxCPMModel
    v1_stub.LoRAConfig = DummyLoRAConfig
    monkeypatch.setitem(sys.modules, "voxcpm.model.voxcpm", v1_stub)

    v2_stub = types.ModuleType("voxcpm.model.voxcpm2")
    v2_stub.VoxCPM2Model = DummyVoxCPM2Model
    monkeypatch.setitem(sys.modules, "voxcpm.model.voxcpm2", v2_stub)

    hub_stub = types.ModuleType("huggingface_hub")
    hub_stub.snapshot_download = lambda **kwargs: kwargs["repo_id"]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_stub)

    spec = importlib.util.spec_from_file_location("voxcpm.core", CORE_PATH)
    core = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "voxcpm.core", core)
    assert spec.loader is not None
    spec.loader.exec_module(core)
    return core


def make_model_dir(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architecture": "voxcpm2"}), encoding="utf-8")
    return model_dir


def test_denoiser_is_not_loaded_during_model_init(monkeypatch, tmp_path):
    core = load_core_with_stubs(monkeypatch)
    model_dir = make_model_dir(tmp_path)

    model = core.VoxCPM(
        str(model_dir),
        zipenhancer_model_path="zip-model",
        enable_denoiser=True,
        optimize=False,
    )

    assert model.denoiser is None
    assert model._denoiser_model_path == "zip-model"
    assert "voxcpm.zipenhancer" not in sys.modules


def test_denoiser_is_loaded_when_generation_requests_denoise(monkeypatch, tmp_path):
    core = load_core_with_stubs(monkeypatch)
    model_dir = make_model_dir(tmp_path)
    ref_audio = tmp_path / "reference.wav"
    ref_audio.write_bytes(b"RIFF")

    init_calls = []
    enhance_calls = []

    zipenhancer_stub = types.ModuleType("voxcpm.zipenhancer")

    class FakeZipEnhancer:
        def __init__(self, model_path):
            init_calls.append(model_path)

        def enhance(self, input_path, output_path=None, normalize_loudness=True):
            enhance_calls.append((input_path, output_path, normalize_loudness))
            Path(output_path).write_bytes(b"RIFF")
            return output_path

    zipenhancer_stub.ZipEnhancer = FakeZipEnhancer
    monkeypatch.setitem(sys.modules, "voxcpm.zipenhancer", zipenhancer_stub)

    model = core.VoxCPM(
        str(model_dir),
        zipenhancer_model_path="zip-model",
        enable_denoiser=True,
        optimize=False,
    )

    wav = model.generate("hello", reference_wav_path=str(ref_audio), denoise=True)

    assert init_calls == ["zip-model"]
    assert len(enhance_calls) == 1
    assert enhance_calls[0][0] == str(ref_audio)
    assert enhance_calls[0][1] != str(ref_audio)
    assert model.tts_model.prompt_cache_calls[0]["reference_wav_path"] == enhance_calls[0][1]
    np.testing.assert_array_equal(wav, np.array([0.0], dtype=np.float32))
