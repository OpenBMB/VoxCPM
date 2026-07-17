import http.client
import json
import threading
import urllib.parse

DEFAULT_SERVER_URL = "http://127.0.0.1:8765"

# Conexiones keep-alive por hilo (el servidor habla HTTP/1.1). Thread-local para
# que un posible uso concurrente del cliente no comparta sockets.
_local = threading.local()


class VoxCPMServerError(RuntimeError):
    pass


def _split_server_url(server_url):
    parsed = urllib.parse.urlsplit(server_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    return host, port


def _get_connection(host, port):
    connections = getattr(_local, "connections", None)
    if connections is None:
        connections = {}
        _local.connections = connections
    conn = connections.get((host, port))
    if conn is None:
        conn = http.client.HTTPConnection(host, port)
        connections[(host, port)] = conn
    return conn


def _drop_connection(host, port):
    connections = getattr(_local, "connections", {})
    conn = connections.pop((host, port), None)
    if conn is not None:
        conn.close()


def _http_request(method, server_url, path, body=None, headers=None, timeout=None):
    """Hace una peticion reutilizando la conexion del hilo; devuelve (status, headers, body).

    Reintenta una vez si la conexion keep-alive quedo obsoleta (el servidor la
    cerro entre peticiones) — en ese caso el servidor no llego a procesar nada.
    """
    host, port = _split_server_url(server_url)
    stale_errors = (
        http.client.RemoteDisconnected,
        http.client.BadStatusLine,
        http.client.CannotSendRequest,
        ConnectionResetError,
        ConnectionAbortedError,
        BrokenPipeError,
    )
    last_exc = None
    for attempt in range(2):
        conn = _get_connection(host, port)
        conn.timeout = timeout
        if conn.sock is not None:
            conn.sock.settimeout(timeout)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            data = response.read()
            return response.status, response.headers, data
        except stale_errors as exc:
            last_exc = exc
            _drop_connection(host, port)
        except OSError:
            _drop_connection(host, port)
            raise
    raise last_exc


def _request_json(url_path, server_url, timeout=2):
    try:
        status, _headers, data = _http_request("GET", server_url, url_path, timeout=timeout)
    except OSError as exc:
        raise VoxCPMServerError(
            "No pude conectar con el servidor VoxCPM. " "Abre otra terminal y ejecuta: python voxcpm_server.py"
        ) from exc
    if status != 200:
        raise VoxCPMServerError(f"El servidor VoxCPM devolvio un error: {data.decode('utf-8', errors='replace')}")
    return json.loads(data.decode("utf-8"))


def check_server(server_url=DEFAULT_SERVER_URL, timeout=2):
    return _request_json("/health", server_url, timeout=timeout)


def _parse_metrics_header(headers):
    raw_metrics = headers.get("X-VoxCPM-Metrics")
    if not raw_metrics:
        return {}
    try:
        return json.loads(raw_metrics)
    except json.JSONDecodeError:
        return {}


def generate_wav_bytes_with_metrics(payload, server_url=DEFAULT_SERVER_URL, timeout=None):
    body = json.dumps(payload).encode("utf-8")
    try:
        status, headers, data = _http_request(
            "POST",
            server_url,
            "/generate",
            body=body,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
    except OSError as exc:
        raise VoxCPMServerError(
            "No pude conectar con el servidor VoxCPM. " "Abre otra terminal y ejecuta: python voxcpm_server.py"
        ) from exc
    if status != 200:
        raise VoxCPMServerError(f"El servidor VoxCPM devolvio un error: {data.decode('utf-8', errors='replace')}")
    return data, _parse_metrics_header(headers)


def generate_wav_bytes(payload, server_url=DEFAULT_SERVER_URL, timeout=None):
    wav_bytes, _metrics = generate_wav_bytes_with_metrics(payload, server_url, timeout=timeout)
    return wav_bytes
