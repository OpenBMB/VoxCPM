"""Regression tests for long-text chunking (issue #372).

Long single-shot generations drift into distortion past a few hundred
characters. ``VoxCPM._generate`` now splits long text into sentence-sized
chunks and synthesizes each one separately, so every generation stays short.

The heavy model dependencies are stubbed so ``core.py`` can be loaded in
isolation without torch / the model weights.
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

MAX_CHARS = 40  # small threshold so tests stay short and deterministic


def _load_core():
    """Load voxcpm.core with its heavy imports replaced by lightweight stubs."""
    pkg = types.ModuleType("voxcpm")
    pkg.__path__ = [str(ROOT / "src" / "voxcpm")]
    sys.modules["voxcpm"] = pkg

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *a, **k: None
    sys.modules["huggingface_hub"] = hub

    model_pkg = types.ModuleType("voxcpm.model")
    model_pkg.__path__ = []
    sys.modules["voxcpm.model"] = model_pkg

    voxcpm_mod = types.ModuleType("voxcpm.model.voxcpm")
    voxcpm_mod.VoxCPMModel = type("VoxCPMModel", (), {})
    voxcpm_mod.LoRAConfig = type("LoRAConfig", (), {})
    sys.modules["voxcpm.model.voxcpm"] = voxcpm_mod

    voxcpm2_mod = types.ModuleType("voxcpm.model.voxcpm2")
    voxcpm2_mod.VoxCPM2Model = type("VoxCPM2Model", (), {})
    sys.modules["voxcpm.model.voxcpm2"] = voxcpm2_mod

    utils_mod = types.ModuleType("voxcpm.model.utils")

    def next_and_close(gen):
        try:
            return next(gen)
        finally:
            gen.close()

    utils_mod.next_and_close = next_and_close
    sys.modules["voxcpm.model.utils"] = utils_mod

    spec = importlib.util.spec_from_file_location("voxcpm.core", CORE_PATH)
    core = importlib.util.module_from_spec(spec)
    sys.modules["voxcpm.core"] = core
    assert spec.loader is not None
    spec.loader.exec_module(core)
    return core


core = _load_core()


class _FakeWav:
    """Stand-in for a torch tensor: squeeze(0).cpu().numpy() -> np.ndarray."""

    def __init__(self, arr):
        self._arr = arr

    def squeeze(self, _dim):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _RecordingModel:
    """Records every target_text handed to the generation entry point."""

    def __init__(self):
        self.calls = []

    def _generate_with_prompt_cache(self, target_text, streaming=False, **kwargs):
        self.calls.append(target_text)
        # one sample per character -> audio length reflects chunk length
        wav = _FakeWav(np.zeros(len(target_text), dtype=np.float32))

        def _gen():
            yield wav, None, None

        return _gen()


def _make_vox(model):
    vox = core.VoxCPM.__new__(core.VoxCPM)
    vox.tts_model = model
    vox.denoiser = None
    vox.text_normalizer = None
    return vox


# --------------------------------------------------------------------------- #
# split_text_into_chunks (pure helper)
# --------------------------------------------------------------------------- #


def test_short_text_is_single_chunk():
    text = "你好世界。"
    assert core.split_text_into_chunks(text, max_chars=MAX_CHARS) == [text]


def test_long_text_splits_on_sentence_boundaries():
    text = "第一句话。" * 20  # 100 chars, far over MAX_CHARS
    chunks = core.split_text_into_chunks(text, max_chars=MAX_CHARS)
    assert len(chunks) > 1, "long text must be split into multiple chunks"
    assert all(len(c) <= MAX_CHARS for c in chunks), "no chunk may exceed max_chars"
    assert "".join(chunks) == text, "chunks must reconstruct the original text"


def test_splitting_disabled_when_max_chars_non_positive():
    text = "很长的句子" * 50
    assert core.split_text_into_chunks(text, max_chars=0) == [text]


def test_sentence_without_delimiter_is_hard_sliced():
    text = "字" * 100  # no punctuation to split on
    chunks = core.split_text_into_chunks(text, max_chars=MAX_CHARS)
    assert all(len(c) <= MAX_CHARS for c in chunks)
    assert "".join(chunks) == text


# --------------------------------------------------------------------------- #
# _generate wiring: long text -> multiple model calls, audio concatenated
# --------------------------------------------------------------------------- #


def test_generate_chunks_long_text_into_multiple_calls():
    model = _RecordingModel()
    vox = _make_vox(model)
    text = "这是一段很长的测试文本。" * 10  # ~120 chars

    out = list(vox._generate(text, streaming=False, max_chunk_chars=MAX_CHARS))

    assert len(model.calls) > 1, (
        f"long text should produce >1 generation call, got {len(model.calls)}"
    )
    assert all(len(c) <= MAX_CHARS for c in model.calls)
    # single concatenated waveform whose length == total synthesized characters
    assert len(out) == 1
    assert out[0].shape[0] == sum(len(c) for c in model.calls)
    print(f"long text -> {len(model.calls)} chunks, audio samples={out[0].shape[0]}")


def test_generate_short_text_is_single_call():
    model = _RecordingModel()
    vox = _make_vox(model)
    text = "短文本。"

    list(vox._generate(text, streaming=False, max_chunk_chars=MAX_CHARS))

    assert len(model.calls) == 1, "short text must not be split"
    print(f"short text -> {len(model.calls)} chunk")


def test_generate_streaming_yields_each_chunk():
    model = _RecordingModel()
    vox = _make_vox(model)
    text = "流式测试的句子。" * 8

    out = list(vox._generate(text, streaming=True, max_chunk_chars=MAX_CHARS))

    assert len(model.calls) > 1
    assert len(out) == len(model.calls), "streaming yields one array per chunk"
    print(f"streaming long text -> {len(model.calls)} chunks yielded")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
