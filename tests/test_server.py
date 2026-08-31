"""Tests for the FastAPI REST server (server.py).

The VoxCPM model is replaced by a stub engine so these tests run without
torch or model weights.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "server.py"

spec = importlib.util.spec_from_file_location("voxcpm_server", SERVER_PATH)
server = importlib.util.module_from_spec(spec)
sys.modules["voxcpm_server"] = server
spec.loader.exec_module(server)

from fastapi.testclient import TestClient  # noqa: E402


class StubEngine:
    """Mimics server.TTSEngine without loading any model."""

    model_id = "stub/VoxCPM2"
    device = "cpu"
    sample_rate = 16000

    def __init__(self):
        self.loaded = True
        self.last_kwargs = None

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        return np.zeros(self.sample_rate, dtype=np.float32)  # 1 second of silence


@pytest.fixture()
def client(monkeypatch):
    stub = StubEngine()
    monkeypatch.setattr(server, "engine", stub)
    with TestClient(server.app) as c:
        c.stub = stub
        yield c


def _make_wav_bytes(duration_s: float = 0.5, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(duration_s * sr), dtype=np.float32), sr, format="WAV")
    return buf.getvalue()


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["sample_rate"] == 16000


def test_tts_basic(client):
    res = client.post("/api/tts", data={"text": "Hello world"})
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert res.headers["X-Sample-Rate"] == "16000"
    wav, sr = sf.read(io.BytesIO(res.content))
    assert sr == 16000
    assert len(wav) == 16000


def test_tts_control_is_prefixed(client):
    res = client.post("/api/tts", data={"text": "Hello", "control": "(warm) female (voice)"})
    assert res.status_code == 200
    # Parentheses inside the control text are stripped; the whole control is wrapped once.
    assert client.stub.last_kwargs["text"] == "(warm female voice)Hello"


def test_tts_empty_text_rejected(client):
    res = client.post("/api/tts", data={"text": "   "})
    assert res.status_code == 400


def test_tts_bad_cfg_rejected(client):
    res = client.post("/api/tts", data={"text": "Hello", "cfg_value": "50"})
    assert res.status_code == 400


def test_tts_bad_format_rejected(client):
    res = client.post("/api/tts", data={"text": "Hello", "response_format": "mp3"})
    assert res.status_code == 400


def test_tts_prompt_audio_requires_prompt_text(client):
    res = client.post(
        "/api/tts",
        data={"text": "Hello"},
        files={"prompt_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
    )
    assert res.status_code == 400
    assert "prompt_text" in res.json()["detail"]


def test_tts_with_reference_audio(client):
    res = client.post(
        "/api/tts",
        data={"text": "Cloned speech", "denoise": "true", "seed": "42"},
        files={"reference_audio": ("ref.wav", _make_wav_bytes(), "audio/wav")},
    )
    assert res.status_code == 200
    kwargs = client.stub.last_kwargs
    assert kwargs["reference_wav_path"] is not None
    assert kwargs["denoise"] is True
    assert kwargs["seed"] == 42


def test_tts_ultimate_cloning(client):
    wav_bytes = _make_wav_bytes()
    res = client.post(
        "/api/tts",
        data={"text": "Ultimate clone", "prompt_text": "reference transcript"},
        files={
            "prompt_audio": ("p.wav", wav_bytes, "audio/wav"),
            "reference_audio": ("r.wav", wav_bytes, "audio/wav"),
        },
    )
    assert res.status_code == 200
    kwargs = client.stub.last_kwargs
    assert kwargs["prompt_wav_path"] is not None
    assert kwargs["reference_wav_path"] is not None
    assert kwargs["prompt_text"] == "reference transcript"


def test_openai_endpoint(client):
    res = client.post(
        "/v1/audio/speech",
        json={"model": "voxcpm", "input": "Hello!", "voice": "a warm female voice"},
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert client.stub.last_kwargs["text"] == "(a warm female voice)Hello!"


def test_openai_endpoint_default_voice(client):
    res = client.post("/v1/audio/speech", json={"input": "Hi", "voice": "default"})
    assert res.status_code == 200
    assert client.stub.last_kwargs["text"] == "Hi"


def test_openai_endpoint_flac(client):
    res = client.post("/v1/audio/speech", json={"input": "Hi", "response_format": "flac"})
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/flac"
    wav, sr = sf.read(io.BytesIO(res.content))
    assert sr == 16000


def test_index_serves_web_ui(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "VoxCPM Studio" in res.text
