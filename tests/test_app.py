from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import threading
import time

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


def test_extract_parakeet_asr_text_accepts_batch_decode_output():
    assert app._extract_parakeet_asr_text([" hej ", "", " världen"]) == "hej världen"


def test_normalize_asr_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown ASR backend"):
        app._normalize_asr_backend("whisper")


def test_prepare_asr_audio_keeps_16khz_mono_wav(tmp_path):
    wav_path = tmp_path / "mono.wav"
    sf.write(wav_path, np.zeros(160, dtype=np.float32), 16000)

    prepared_path, temp_path = app._prepare_asr_audio(str(wav_path))

    assert prepared_path == str(wav_path)
    assert temp_path is None


def test_prepare_asr_audio_converts_to_16khz_mono_wav(tmp_path):
    wav_path = tmp_path / "stereo_8k.wav"
    audio = np.zeros((80, 2), dtype=np.float32)
    sf.write(wav_path, audio, 8000)

    prepared_path, temp_path = app._prepare_asr_audio(str(wav_path))

    try:
        info = sf.info(prepared_path)
        assert temp_path == prepared_path
        assert info.samplerate == 16000
        assert info.channels == 1
    finally:
        if temp_path:
            app.os.unlink(temp_path)


def test_resolve_generation_inputs_auto_transcribes_blank_ultimate_prompt():
    class FakeDemo:
        calls = []

        def prompt_wav_recognition(self, path, progress_callback=None):
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


def test_auto_asr_backend_prefers_local_parakeet_on_cuda():
    demo = app.VoxCPMDemo.__new__(app.VoxCPMDemo)
    demo.asr_backend = "auto"
    demo.device = "cuda"
    demo.parakeet_model_id = "models/nvidia__parakeet-tdt-0.6b-v3"

    assert demo._should_use_parakeet_asr() is True
    assert demo._resolved_asr_backend_name() == "parakeet"


def test_auto_asr_backend_uses_sensevoice_without_local_parakeet():
    demo = app.VoxCPMDemo.__new__(app.VoxCPMDemo)
    demo.asr_backend = "auto"
    demo.device = "cuda"
    demo.parakeet_model_id = None

    assert demo._should_use_parakeet_asr() is False
    assert demo._resolved_asr_backend_name() == "sensevoice"


def test_sensevoice_asr_backend_disables_local_parakeet():
    demo = app.VoxCPMDemo.__new__(app.VoxCPMDemo)
    demo.asr_backend = "sensevoice"
    demo.device = "cuda"
    demo.parakeet_model_id = "models/nvidia__parakeet-tdt-0.6b-v3"

    assert demo._should_use_parakeet_asr() is False


def test_parakeet_asr_backend_uses_local_parakeet():
    demo = app.VoxCPMDemo.__new__(app.VoxCPMDemo)
    demo.asr_backend = "parakeet"
    demo.device = "cuda"
    demo.parakeet_model_id = "models/nvidia__parakeet-tdt-0.6b-v3"

    assert demo._should_use_parakeet_asr() is True
    assert demo._resolved_asr_backend_name() == "parakeet"


def test_preload_models_loads_tts_denoiser_parakeet_and_sensevoice_fallback_on_cuda_auto():
    class FakeCoreModel:
        def _get_or_load_denoiser(self):
            calls.append("denoiser")

    calls = []
    demo = app.VoxCPMDemo.__new__(app.VoxCPMDemo)
    demo.asr_backend = "auto"
    demo.device = "cuda"
    demo.parakeet_model_id = "models/nvidia__parakeet-tdt-0.6b-v3"
    demo.get_or_load_voxcpm = lambda: calls.append("tts") or FakeCoreModel()
    demo.get_or_load_parakeet_asr_model = lambda: calls.append("parakeet")
    demo.get_or_load_asr_model = lambda: calls.append("sensevoice")

    demo.preload_models()

    assert calls == ["tts", "denoiser", "parakeet", "sensevoice"]


def test_get_or_load_asr_model_serializes_concurrent_loads(monkeypatch):
    load_count = 0
    load_count_lock = threading.Lock()

    class FakeAutoModel:
        def __init__(self, **kwargs):
            nonlocal load_count
            with load_count_lock:
                load_count += 1
            time.sleep(0.05)

    monkeypatch.setattr(app, "AutoModel", FakeAutoModel)

    demo = app.VoxCPMDemo.__new__(app.VoxCPMDemo)
    demo.asr_model = None
    demo.asr_model_id = "fake/asr"
    demo.asr_device = "cpu"

    results = []
    errors = []

    def load_model():
        try:
            results.append(demo.get_or_load_asr_model())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=load_model) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert load_count == 1
    assert len(results) == len(threads)
    assert all(result is results[0] for result in results)


def test_run_demo_preloads_models_before_launching_web_ui(monkeypatch):
    events = []
    launch_kwargs = {}

    class FakeDemo:
        def __init__(self, model_id, device, asr_backend):
            events.append(("demo", model_id, device, asr_backend))

        def preload_models(self, **kwargs):
            events.append(("preload", kwargs))

    class FakeQueuedInterface:
        def launch(self, **kwargs):
            events.append("launch")
            launch_kwargs.update(kwargs)

    class FakeInterface:
        def queue(self, **kwargs):
            events.append(("queue", kwargs))
            return FakeQueuedInterface()

    monkeypatch.setattr(app, "VoxCPMDemo", FakeDemo)
    monkeypatch.setattr(app, "create_demo_interface", lambda demo: FakeInterface())

    app.run_demo()

    assert events == [
        ("demo", "openbmb/VoxCPM2", "auto", "auto"),
        ("preload", {"preload_asr": True, "preload_tts": True, "preload_denoiser": True}),
        ("queue", {"max_size": 10, "default_concurrency_limit": 1}),
        "launch",
    ]
    assert launch_kwargs["server_name"] == "127.0.0.1"
    assert launch_kwargs["server_port"] == 8808
    assert launch_kwargs["inbrowser"] is True


def test_prompt_wav_recognition_reports_progress_and_uses_parakeet(monkeypatch):
    progress_events = []
    calls = []
    demo = app.VoxCPMDemo.__new__(app.VoxCPMDemo)
    demo.asr_backend = "parakeet"
    demo.device = "cuda"
    demo.parakeet_model_id = "models/nvidia__parakeet-tdt-0.6b-v3"

    monkeypatch.setattr(app, "_prepare_asr_audio", lambda path: ("prepared.wav", None))

    def fake_parakeet(path, progress_callback=None):
        calls.append(("parakeet", path))
        app._emit_progress(progress_callback, 0.55, "Transcribing reference audio with Parakeet, 55%")
        return "transcript"

    demo._recognize_with_parakeet = fake_parakeet
    demo._recognize_with_sensevoice = lambda path, progress_callback=None: calls.append(("sensevoice", path)) or ""

    text = demo.prompt_wav_recognition(
        "ref.wav", progress_callback=lambda value, label: progress_events.append((value, label))
    )

    assert text == "transcript"
    assert calls == [("parakeet", "prepared.wav")]
    assert progress_events[0][0] == 0.05
    assert "Parakeet" in progress_events[-1][1]


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

    progress_callback = lambda step, total: None

    sr, wav, seed = app.VoxCPMDemo.generate_tts_audio(
        demo,
        text_input="Hello",
        reference_wav_path_input={"path": "ref.wav"},
        prompt_text="reference transcript",
        do_normalize=False,
        denoise=False,
        seed=123,
        progress_callback=progress_callback,
    )

    assert sr == 24000
    assert seed == 456
    np.testing.assert_array_equal(wav, np.array([0.0], dtype=np.float32))
    assert fake_model.kwargs["reference_wav_path"] == "ref.wav"
    assert fake_model.kwargs["prompt_wav_path"] == "ref.wav"
    assert fake_model.kwargs["prompt_text"] == "reference transcript"
    assert fake_model.kwargs["progress_callback"] is progress_callback
