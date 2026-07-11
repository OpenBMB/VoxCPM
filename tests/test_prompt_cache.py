from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_voxcpm_modules():
    saved_modules = {name: sys.modules.get(name) for name in ("voxcpm", "voxcpm.core")}
    yield
    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def import_real_core():
    sys.path.insert(0, str(ROOT / "src"))
    sys.modules.pop("voxcpm", None)
    sys.modules.pop("voxcpm.core", None)
    return importlib.import_module("voxcpm.core")


class DummyTensor:
    def __init__(self, values):
        self.values = np.array(values, dtype=np.float32)

    def squeeze(self, axis):
        return DummyTensor(np.squeeze(self.values, axis=axis))

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class DummyVoxCPM2Model:
    sample_rate = 16000

    def __init__(self):
        self.build_calls = []
        self.generate_calls = []

    def build_prompt_cache(
        self,
        prompt_text=None,
        prompt_wav_path=None,
        reference_wav_path=None,
        trim_silence_vad=False,
    ):
        self.build_calls.append(
            {
                "prompt_text": prompt_text,
                "prompt_wav_path": prompt_wav_path,
                "reference_wav_path": reference_wav_path,
                "trim_silence_vad": trim_silence_vad,
            }
        )
        return {
            "mode": "ref_continuation",
            "prompt_text": prompt_text,
            "audio_feat": "prompt-feat",
            "ref_audio_feat": "ref-feat",
        }

    def _generate_with_prompt_cache(self, **kwargs):
        self.generate_calls.append(kwargs)
        yield DummyTensor([[0.0, 0.1, 0.2]]), "tokens", "features"


def make_model(core_module):
    core_module.VoxCPM2Model = DummyVoxCPM2Model
    model = core_module.VoxCPM.__new__(core_module.VoxCPM)
    model.voxcpm_model_path = "dummy-model-path"
    model.tts_model = DummyVoxCPM2Model()
    model.denoiser = None
    model.text_normalizer = None
    return model


def test_generate_from_prompt_cache_reuses_loaded_cache_for_multiple_blocks():
    core = import_real_core()
    model = make_model(core)
    prompt_cache = {"mode": "ref_continuation", "audio_feat": "cached"}

    first = model.generate_from_prompt_cache("hello", prompt_cache, inference_timesteps=12)
    second = model.generate_from_prompt_cache("world", prompt_cache, inference_timesteps=12)

    assert model.tts_model.build_calls == []
    assert len(model.tts_model.generate_calls) == 2
    assert model.tts_model.generate_calls[0]["prompt_cache"] is prompt_cache
    assert model.tts_model.generate_calls[1]["prompt_cache"] is prompt_cache
    assert model.tts_model.generate_calls[0]["inference_timesteps"] == 12
    np.testing.assert_allclose(first, np.array([0.0, 0.1, 0.2], dtype=np.float32))
    np.testing.assert_allclose(second, np.array([0.0, 0.1, 0.2], dtype=np.float32))


def test_load_or_build_prompt_cache_reuses_disk_cache_when_metadata_matches(tmp_path):
    core = import_real_core()
    model = make_model(core)
    wav_path = tmp_path / "voice.wav"
    wav_path.write_bytes(b"voice-v1")
    cache_path = tmp_path / "voice_cache.pt"

    first = model.load_or_build_prompt_cache(
        str(cache_path),
        prompt_text="prompt",
        prompt_wav_path=str(wav_path),
        reference_wav_path=str(wav_path),
        model_id="openbmb/VoxCPM2",
    )
    second = model.load_or_build_prompt_cache(
        str(cache_path),
        prompt_text="prompt",
        prompt_wav_path=str(wav_path),
        reference_wav_path=str(wav_path),
        model_id="openbmb/VoxCPM2",
    )

    assert first == second
    assert len(model.tts_model.build_calls) == 1


def test_load_or_build_prompt_cache_rebuilds_when_prompt_text_changes(tmp_path):
    core = import_real_core()
    model = make_model(core)
    wav_path = tmp_path / "voice.wav"
    wav_path.write_bytes(b"voice-v1")
    cache_path = tmp_path / "voice_cache.pt"

    model.load_or_build_prompt_cache(
        str(cache_path),
        prompt_text="prompt",
        prompt_wav_path=str(wav_path),
        reference_wav_path=str(wav_path),
        model_id="openbmb/VoxCPM2",
    )
    model.load_or_build_prompt_cache(
        str(cache_path),
        prompt_text="changed prompt",
        prompt_wav_path=str(wav_path),
        reference_wav_path=str(wav_path),
        model_id="openbmb/VoxCPM2",
    )

    assert len(model.tts_model.build_calls) == 2


def test_load_or_build_prompt_cache_rebuilds_when_wav_changes(tmp_path):
    core = import_real_core()
    model = make_model(core)
    wav_path = tmp_path / "voice.wav"
    wav_path.write_bytes(b"voice-v1")
    cache_path = tmp_path / "voice_cache.pt"

    model.load_or_build_prompt_cache(
        str(cache_path),
        prompt_text="prompt",
        prompt_wav_path=str(wav_path),
        reference_wav_path=str(wav_path),
        model_id="openbmb/VoxCPM2",
    )
    wav_path.write_bytes(b"voice-v2")
    model.load_or_build_prompt_cache(
        str(cache_path),
        prompt_text="prompt",
        prompt_wav_path=str(wav_path),
        reference_wav_path=str(wav_path),
        model_id="openbmb/VoxCPM2",
    )

    assert len(model.tts_model.build_calls) == 2
