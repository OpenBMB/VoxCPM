import json

import pytest

import voxcpm_client


def _fake_http_request(status=200, body=b"RIFF", metrics=None):
    headers = {}
    if metrics is not None:
        headers["X-VoxCPM-Metrics"] = json.dumps(metrics)

    def fake(method, server_url, path, body_arg=None, headers_arg=None, timeout=None, **kwargs):
        return status, headers, body

    return fake


def test_generate_wav_bytes_with_metrics_reads_header(monkeypatch):
    metrics = {
        "cache_seconds": 0.1,
        "inference_seconds": 2.3,
        "wav_write_seconds": 0.2,
    }
    monkeypatch.setattr(voxcpm_client, "_http_request", _fake_http_request(metrics=metrics))

    wav_bytes, parsed_metrics = voxcpm_client.generate_wav_bytes_with_metrics({"text": "hola"})

    assert wav_bytes == b"RIFF"
    assert parsed_metrics == metrics


def test_generate_wav_bytes_remains_bytes_only(monkeypatch):
    monkeypatch.setattr(voxcpm_client, "_http_request", _fake_http_request(metrics={"inference_seconds": 1.0}))

    assert voxcpm_client.generate_wav_bytes({"text": "hola"}) == b"RIFF"


def test_generate_wav_bytes_with_metrics_handles_missing_header(monkeypatch):
    monkeypatch.setattr(voxcpm_client, "_http_request", _fake_http_request())

    wav_bytes, parsed_metrics = voxcpm_client.generate_wav_bytes_with_metrics({"text": "hola"})

    assert wav_bytes == b"RIFF"
    assert parsed_metrics == {}


def test_generate_raises_server_error_on_non_200(monkeypatch):
    monkeypatch.setattr(voxcpm_client, "_http_request", _fake_http_request(status=500, body=b'{"error": "boom"}'))

    with pytest.raises(voxcpm_client.VoxCPMServerError):
        voxcpm_client.generate_wav_bytes_with_metrics({"text": "hola"})


def test_generate_raises_server_error_on_connection_failure(monkeypatch):
    def fake(*args, **kwargs):
        raise ConnectionRefusedError("nadie escucha")

    monkeypatch.setattr(voxcpm_client, "_http_request", fake)

    with pytest.raises(voxcpm_client.VoxCPMServerError):
        voxcpm_client.generate_wav_bytes_with_metrics({"text": "hola"})


def test_check_server_forwards_timeout(monkeypatch):
    calls = []

    def fake_request_json(url_path, server_url, timeout):
        calls.append((url_path, server_url, timeout))
        return {"ok": True}

    monkeypatch.setattr(voxcpm_client, "_request_json", fake_request_json)

    assert voxcpm_client.check_server("http://localhost:9999", timeout=7) == {"ok": True}
    assert calls == [("/health", "http://localhost:9999", 7)]


def test_http_request_retries_once_on_stale_keepalive(monkeypatch):
    import http.client

    class FakeResponse:
        status = 200
        headers = {}

        @staticmethod
        def read():
            return b"RIFF"

    class FakeConnection:
        def __init__(self):
            self.sock = None
            self.timeout = None
            self.requests = 0
            self.closed = False

        def request(self, *args, **kwargs):
            self.requests += 1

        def getresponse(self):
            if attempts["n"] == 0:
                attempts["n"] += 1
                raise http.client.RemoteDisconnected("stale")
            return FakeResponse()

        def close(self):
            self.closed = True

    attempts = {"n": 0}
    connections = []

    def fake_get_connection(host, port):
        conn = FakeConnection()
        connections.append(conn)
        return conn

    monkeypatch.setattr(voxcpm_client, "_get_connection", fake_get_connection)
    monkeypatch.setattr(voxcpm_client, "_drop_connection", lambda host, port: None)

    status, headers, data = voxcpm_client._http_request("GET", "http://127.0.0.1:8765", "/health")

    assert status == 200
    assert data == b"RIFF"
    assert len(connections) == 2
