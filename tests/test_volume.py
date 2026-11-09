"""Tests for the volume_multiplier control added for issue #362.

Loads ``_apply_volume`` and the module constants directly from
``src/voxcpm/core.py`` without importing the heavy model dependencies.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "src" / "voxcpm" / "core.py"

# Stub the model submodules that core.py imports at module load time, so we can
# exercise the pure volume helper without pulling in torch / the real weights.
hf_stub = types.ModuleType("huggingface_hub")
hf_stub.snapshot_download = lambda *a, **k: ""
sys.modules.setdefault("huggingface_hub", hf_stub)

pkg = types.ModuleType("voxcpm")
pkg.__path__ = [str(ROOT / "src" / "voxcpm")]
sys.modules.setdefault("voxcpm", pkg)

model_pkg = types.ModuleType("voxcpm.model")
model_pkg.__path__ = [str(ROOT / "src" / "voxcpm" / "model")]
sys.modules.setdefault("voxcpm.model", model_pkg)

voxcpm_stub = types.ModuleType("voxcpm.model.voxcpm")
voxcpm_stub.VoxCPMModel = type("VoxCPMModel", (), {})
voxcpm_stub.LoRAConfig = type("LoRAConfig", (), {})
sys.modules["voxcpm.model.voxcpm"] = voxcpm_stub

voxcpm2_stub = types.ModuleType("voxcpm.model.voxcpm2")
voxcpm2_stub.VoxCPM2Model = type("VoxCPM2Model", (), {})
sys.modules["voxcpm.model.voxcpm2"] = voxcpm2_stub

utils_stub = types.ModuleType("voxcpm.model.utils")
utils_stub.next_and_close = lambda gen: next(gen)
sys.modules["voxcpm.model.utils"] = utils_stub

spec = importlib.util.spec_from_file_location("voxcpm.core", CORE_PATH)
core = importlib.util.module_from_spec(spec)
sys.modules["voxcpm.core"] = core
assert spec.loader is not None
spec.loader.exec_module(core)


def _rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(wav))))


def test_identity_multiplier_is_noop():
    wav = np.array([0.1, -0.2, 0.3, -0.05], dtype=np.float32)
    out = core._apply_volume(wav, core._DEFAULT_VOLUME_MULTIPLIER)
    assert np.array_equal(out, wav)


def test_multiplier_scales_rms_linearly():
    wav = (np.array([0.1, -0.2, 0.15, -0.05], dtype=np.float32))
    out = core._apply_volume(wav, 2.0)
    # Peak is 0.4 < 1.0, so no clipping limiter kicks in: RMS doubles exactly.
    ratio = _rms(out) / _rms(wav)
    print(f"rms in={_rms(wav):.6f} out={_rms(out):.6f} ratio={ratio:.6f}")
    assert ratio == pytest.approx(2.0, rel=1e-5)


def test_peak_limited_to_avoid_clipping():
    wav = np.array([0.6, -0.5, 0.4], dtype=np.float32)
    out = core._apply_volume(wav, 3.0)  # would reach 1.8, must be limited to <= 1.0
    peak = float(np.max(np.abs(out)))
    print(f"peak after 3x = {peak:.6f}")
    assert peak <= core._AUDIO_PEAK_LIMIT + 1e-6
    assert peak == pytest.approx(core._AUDIO_PEAK_LIMIT, rel=1e-5)


def test_empty_waveform_untouched():
    wav = np.array([], dtype=np.float32)
    out = core._apply_volume(wav, 5.0)
    assert out.size == 0


def _make_pipeline(monkeypatch):
    """Build a VoxCPM instance whose generation yields a fixed unit waveform."""
    inst = core.VoxCPM.__new__(core.VoxCPM)
    inst.text_normalizer = None
    inst.denoiser = None
    base = np.array([0.1, -0.2, 0.15, -0.05], dtype=np.float32)

    class _Tensor:
        def __init__(self, arr):
            self._arr = arr

        def squeeze(self, _dim):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    class _FakeModel:
        def _generate_with_prompt_cache(self, **kwargs):
            yield (_Tensor(base), None, None)

    inst.tts_model = _FakeModel()
    # core.py branches on isinstance(..., VoxCPM2Model); force the v1 path.
    monkeypatch.setattr(core, "VoxCPM2Model", type("Other", (), {}))
    return inst, base


def test_generate_applies_multiplier_end_to_end(monkeypatch):
    inst, base = _make_pipeline(monkeypatch)
    out = inst.generate(text="hello world", volume_multiplier=2.0)
    ratio = _rms(out) / _rms(base)
    print(f"end-to-end ratio = {ratio:.6f}")
    assert ratio == pytest.approx(2.0, rel=1e-5)


def test_generate_rejects_nonpositive_multiplier(monkeypatch):
    inst, _ = _make_pipeline(monkeypatch)
    with pytest.raises(ValueError, match="volume_multiplier must be positive"):
        inst.generate(text="hello world", volume_multiplier=0.0)
