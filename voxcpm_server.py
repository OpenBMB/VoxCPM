import argparse
import io
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import soundfile as sf


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from voxcpm import VoxCPM


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_MODEL_ID = "openbmb/VoxCPM2"


class VoxCPMService:
    def __init__(self, model_id=DEFAULT_MODEL_ID):
        self.model_id = model_id
        self.prompt_caches = {}
        self.lock = threading.Lock()
        print(f"Cargando modelo VoxCPM2 una sola vez: {model_id}", flush=True)
        self.model = VoxCPM.from_pretrained(model_id, load_denoiser=False, optimize=False)
        print("Modelo cargado.", flush=True)

    @staticmethod
    def _audio_fingerprint(path):
        if not path:
            return None
        stat = os.stat(path)
        return {
            "path": os.path.abspath(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def _cache_key(self, payload):
        key = {
            "model_id": payload.get("model_id") or self.model_id,
            "prompt_text": payload.get("prompt_text"),
            "prompt_wav": self._audio_fingerprint(payload.get("prompt_wav_path")),
            "reference_wav": self._audio_fingerprint(payload.get("reference_wav_path")),
            "denoise": bool(payload.get("denoise", False)),
            "trim_silence_vad": bool(payload.get("trim_silence_vad", False)),
        }
        return json.dumps(key, sort_keys=True, ensure_ascii=False)

    def _get_prompt_cache(self, payload):
        key = self._cache_key(payload)
        if key in self.prompt_caches:
            print("Usando cache de voz en memoria.", flush=True)
            return self.prompt_caches[key]

        print("Construyendo cache de voz en memoria.", flush=True)
        prompt_cache = self.model.build_prompt_cache(
            prompt_text=payload.get("prompt_text"),
            prompt_wav_path=payload.get("prompt_wav_path"),
            reference_wav_path=payload.get("reference_wav_path"),
            denoise=bool(payload.get("denoise", False)),
            trim_silence_vad=bool(payload.get("trim_silence_vad", False)),
        )
        self.prompt_caches[key] = prompt_cache
        return prompt_cache

    def generate(self, payload):
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("El campo 'text' es obligatorio.")

        # Keep the GPU/model state single-file. Multiple HTTP clients can connect,
        # but generation runs one request at a time.
        with self.lock:
            prompt_cache = self._get_prompt_cache(payload)
            wav = self.model.generate_from_prompt_cache(
                text=text,
                prompt_cache=prompt_cache,
                cfg_value=float(payload.get("cfg_value", 2.0)),
                inference_timesteps=int(payload.get("inference_timesteps", 10)),
                normalize=bool(payload.get("normalize", False)),
                retry_badcase=bool(payload.get("retry_badcase", True)),
                retry_badcase_max_times=int(payload.get("retry_badcase_max_times", 3)),
                retry_badcase_ratio_threshold=float(payload.get("retry_badcase_ratio_threshold", 6.0)),
                seed=payload.get("seed"),
            )

        buffer = io.BytesIO()
        sf.write(buffer, wav, self.model.tts_model.sample_rate, format="WAV")
        return buffer.getvalue()

    def health(self):
        return {
            "ok": True,
            "model_loaded": self.model is not None,
            "model_id": self.model_id,
            "prompt_caches": len(self.prompt_caches),
        }


class VoxCPMRequestHandler(BaseHTTPRequestHandler):
    service = None

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_wav(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            self._send_json(404, {"error": "not found"})
            return
        self._send_json(200, self.service.health())

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            wav_bytes = self.service.generate(payload)
            self._send_wav(wav_bytes)
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"error": str(exc)})

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Servidor local persistente para VoxCPM2.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()

    service = VoxCPMService(model_id=args.model_id)
    VoxCPMRequestHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), VoxCPMRequestHandler)
    print(f"Servidor listo en http://{args.host}:{args.port}", flush=True)
    print("Deja esta ventana abierta mientras generas audios.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
