import sys
import types

import voxcpm_server


class DummyTTSModel:
    sample_rate = 10


class DummyModel:
    def __init__(self):
        self.tts_model = DummyTTSModel()
        self.build_calls = []
        self.generate_calls = []

    def build_prompt_cache(self, **kwargs):
        self.build_calls.append(kwargs)
        return {"cache": len(self.build_calls)}

    def generate_from_prompt_cache(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [0.0] * 20


class DummyVoxCPM:
    calls = []
    model = DummyModel()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.calls.append({"args": args, "kwargs": kwargs})
        cls.model = DummyModel()
        return cls.model


def install_dummy_voxcpm(monkeypatch):
    module = types.ModuleType("voxcpm")
    module.VoxCPM = DummyVoxCPM
    DummyVoxCPM.calls = []
    monkeypatch.setitem(sys.modules, "voxcpm", module)


def fake_sf_write(buffer, wav, sample_rate, format):
    buffer.write(b"RIFF")


def test_service_reuses_prompt_cache_and_reports_metrics(monkeypatch, tmp_path):
    install_dummy_voxcpm(monkeypatch)
    monkeypatch.setattr(voxcpm_server.sf, "write", fake_sf_write)
    wav_path = tmp_path / "ref.wav"
    wav_path.write_bytes(b"wav")

    service = voxcpm_server.VoxCPMService(optimize=False)
    payload = {
        "text": "hola",
        "prompt_text": "prompt",
        "prompt_wav_path": str(wav_path),
        "reference_wav_path": str(wav_path),
        "cfg_value": 2.0,
        "inference_timesteps": 12,
        "normalize": False,
        "denoise": False,
        "trim_silence_vad": False,
    }

    first_body, first_metrics = service.generate(payload)
    second_body, second_metrics = service.generate(payload)

    assert first_body == b"RIFF"
    assert second_body == b"RIFF"
    assert len(DummyVoxCPM.model.build_calls) == 1
    assert len(DummyVoxCPM.model.generate_calls) == 2
    assert first_metrics["cache_hit"] is False
    assert second_metrics["cache_hit"] is True
    assert second_metrics["audio_seconds"] == 2.0
    assert service.health()["last_metrics"] == second_metrics


def test_service_passes_optimize_flag_to_model_loader(monkeypatch):
    install_dummy_voxcpm(monkeypatch)

    voxcpm_server.VoxCPMService(optimize=True)

    assert DummyVoxCPM.calls[0]["kwargs"]["optimize"] is True
    assert DummyVoxCPM.calls[0]["kwargs"]["load_denoiser"] is False
