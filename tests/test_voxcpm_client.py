import json

import voxcpm_client


class FakeResponse:
    def __init__(self, body=b"RIFF", metrics=None):
        self.body = body
        self.headers = {}
        if metrics is not None:
            self.headers["X-VoxCPM-Metrics"] = json.dumps(metrics)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.body


def test_generate_wav_bytes_with_metrics_reads_header(monkeypatch):
    metrics = {
        "cache_seconds": 0.1,
        "inference_seconds": 2.3,
        "wav_write_seconds": 0.2,
    }
    monkeypatch.setattr(voxcpm_client.urllib.request, "urlopen", lambda req: FakeResponse(metrics=metrics))

    wav_bytes, parsed_metrics = voxcpm_client.generate_wav_bytes_with_metrics({"text": "hola"})

    assert wav_bytes == b"RIFF"
    assert parsed_metrics == metrics


def test_generate_wav_bytes_remains_bytes_only(monkeypatch):
    monkeypatch.setattr(
        voxcpm_client.urllib.request,
        "urlopen",
        lambda req: FakeResponse(metrics={"inference_seconds": 1.0}),
    )

    assert voxcpm_client.generate_wav_bytes({"text": "hola"}) == b"RIFF"


def test_generate_wav_bytes_with_metrics_handles_missing_header(monkeypatch):
    monkeypatch.setattr(voxcpm_client.urllib.request, "urlopen", lambda req: FakeResponse())

    wav_bytes, parsed_metrics = voxcpm_client.generate_wav_bytes_with_metrics({"text": "hola"})

    assert wav_bytes == b"RIFF"
    assert parsed_metrics == {}


def test_check_server_forwards_timeout(monkeypatch):
    calls = []

    def fake_request_json(url, timeout):
        calls.append((url, timeout))
        return {"ok": True}

    monkeypatch.setattr(voxcpm_client, "_request_json", fake_request_json)

    assert voxcpm_client.check_server("http://localhost:9999", timeout=7) == {"ok": True}
    assert calls == [("http://localhost:9999/health", 7)]
