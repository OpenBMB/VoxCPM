import json
import urllib.error
import urllib.request


DEFAULT_SERVER_URL = "http://127.0.0.1:8765"


class VoxCPMServerError(RuntimeError):
    pass


def _request_json(url, timeout=2):
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError) as exc:
        raise VoxCPMServerError(
            "No pude conectar con el servidor VoxCPM. "
            "Abre otra terminal y ejecuta: python voxcpm_server.py"
        ) from exc


def check_server(server_url=DEFAULT_SERVER_URL):
    return _request_json(f"{server_url.rstrip('/')}/health", timeout=2)


def generate_wav_bytes(payload, server_url=DEFAULT_SERVER_URL, timeout=None):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url.rstrip('/')}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        if timeout is None:
            response = urllib.request.urlopen(req)
        else:
            response = urllib.request.urlopen(req, timeout=timeout)
        with response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise VoxCPMServerError(f"El servidor VoxCPM devolvio un error: {detail}") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise VoxCPMServerError(
            "No pude conectar con el servidor VoxCPM. "
            "Abre otra terminal y ejecuta: python voxcpm_server.py"
        ) from exc
