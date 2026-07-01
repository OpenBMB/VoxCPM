from __future__ import annotations

import numpy as np
import pytest

import app


class PathLikeObject:
    path = "object.wav"


def test_coerce_audio_filepath_accepts_gradio_file_shapes():
    assert app._coerce_audio_filepath(None) is None
    assert app._coerce_audio_filepath("") is None
    assert app._coerce_audio_filepath("plain.wav") == "plain.wav"
    assert app._coerce_audio_filepath({"path": "dict.wav"}) == "dict.wav"
    assert app._coerce_audio_filepath(PathLikeObject()) == "object.wav"


def test_extract_asr_text_removes_sensevoice_tags():
    result = [{"text": "<|zh|><|NEUTRAL|><|Speech|><|withitn|>你好，世界"}]

    assert app._extract_asr_text(result) == "你好，世界"


def test_resolve_generation_inputs_auto_transcribes_blank_ultimate_prompt():
    class FakeDemo:
        calls = []

        def prompt_wav_recognition(self, path):
            self.calls.append(path)
            return " auto transcript "

    demo = FakeDemo()

    audio_path, prompt_text, control = app._resolve_generation_inputs(
        demo,
        {"path": "ref.wav"},
        True,
        "",
        "warm voice",
    )

    assert audio_path == "ref.wav"
    assert prompt_text == "auto transcript"
    assert control == ""
    assert demo.calls == ["ref.wav"]


def test_resolve_generation_inputs_requires_audio_for_ultimate_mode():
    with pytest.raises(app.gr.Error, match="Upload reference audio"):
        app._resolve_generation_inputs(object(), None, True, "", "")


def test_generate_tts_audio_normalizes_gradio_filedata_path():
    class FakeTTS:
        sample_rate = 24000
        last_successful_seed = 456

    class FakeModel:
        tts_model = FakeTTS()

        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return np.array([0.0], dtype=np.float32)

    fake_model = FakeModel()
    demo = app.VoxCPMDemo.__new__(app.VoxCPMDemo)
    demo.get_or_load_voxcpm = lambda: fake_model

    sr, wav, seed = app.VoxCPMDemo.generate_tts_audio(
        demo,
        text_input="Hello",
        reference_wav_path_input={"path": "ref.wav"},
        prompt_text="reference transcript",
        do_normalize=False,
        denoise=False,
        seed=123,
    )

    assert sr == 24000
    assert seed == 456
    np.testing.assert_array_equal(wav, np.array([0.0], dtype=np.float32))
    assert fake_model.kwargs["reference_wav_path"] == "ref.wav"
    assert fake_model.kwargs["prompt_wav_path"] == "ref.wav"
    assert fake_model.kwargs["prompt_text"] == "reference transcript"
