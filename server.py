#!/usr/bin/env python3
"""VoxCPM REST API server.

A lightweight FastAPI server exposing VoxCPM speech generation over HTTP,
plus a standalone web interface (see ``web/index.html``).

Endpoints
---------
- ``GET  /``                 → web interface
- ``GET  /api/health``       → server / model status
- ``POST /api/tts``          → speech generation (multipart form, supports file upload)
- ``POST /v1/audio/speech``  → OpenAI-compatible text-to-speech endpoint (JSON)

Usage
-----
    pip install -e ".[server]"
    python server.py --port 8000
    # then open http://localhost:8000

The model is loaded lazily on the first request by default; pass ``--preload``
to load it at startup instead.
"""

import argparse
import io
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("voxcpm.server")

WEB_DIR = Path(__file__).resolve().parent / "web"

SUPPORTED_FORMATS = {"wav", "flac", "ogg"}


# ---------------------------------------------------------------------------
# Engine wrapper
# ---------------------------------------------------------------------------


class TTSEngine:
    """Thread-safe lazy wrapper around the VoxCPM model."""

    def __init__(
        self,
        model_id: str = "openbmb/VoxCPM2",
        device: str = "auto",
        load_denoiser: bool = True,
        optimize: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.load_denoiser = load_denoiser
        self.optimize = optimize
        self._model = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    import voxcpm

                    logger.info("Loading VoxCPM model: %s (device=%s)", self.model_id, self.device)
                    self._model = voxcpm.VoxCPM.from_pretrained(
                        self.model_id,
                        load_denoiser=self.load_denoiser,
                        device=None if self.device == "auto" else self.device,
                        optimize=self.optimize,
                    )
                    logger.info("Model loaded (sample rate: %d Hz)", self.sample_rate)
        return self._model

    @property
    def sample_rate(self) -> int:
        if self._model is None:
            raise RuntimeError("Model not loaded yet")
        return self._model.tts_model.sample_rate

    def generate(self, **kwargs) -> np.ndarray:
        model = self.load()
        # The underlying model is not thread-safe; serialize generation.
        with self._generate_lock:
            return model.generate(**kwargs)


engine: Optional[TTSEngine] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_final_text(text: str, control: Optional[str]) -> str:
    """Prefix the text with a ``(control instruction)`` if provided.

    Parentheses are stripped from the control text to avoid breaking the
    ``(control)text`` prompt format expected by the model.
    """
    control = (control or "").strip()
    control = control.replace("(", "").replace(")", "").replace("（", "").replace("）", "").strip()
    return f"({control}){text}" if control else text


def encode_audio(wav: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    import soundfile as sf

    fmt = (fmt or "wav").lower()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{fmt}'. Supported: {sorted(SUPPORTED_FORMATS)}",
        )
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format=fmt.upper())
    return buf.getvalue()


MEDIA_TYPES = {"wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg"}


async def save_upload(upload: Optional[UploadFile], temp_files: list) -> Optional[str]:
    if upload is None or not upload.filename:
        return None
    suffix = Path(upload.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await upload.read())
        temp_files.append(tmp.name)
        return tmp.name


def cleanup(temp_files: list) -> None:
    for path in temp_files:
        try:
            os.unlink(path)
        except OSError:
            pass


def run_generation(
    *,
    text: str,
    control: Optional[str] = None,
    reference_wav_path: Optional[str] = None,
    prompt_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    normalize: bool = False,
    denoise: bool = False,
    seed: Optional[int] = None,
    response_format: str = "wav",
) -> Response:
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="'text' must be a non-empty string")
    if not (0.1 <= cfg_value <= 10.0):
        raise HTTPException(status_code=400, detail="'cfg_value' must be between 0.1 and 10.0")
    if not (1 <= inference_timesteps <= 100):
        raise HTTPException(status_code=400, detail="'inference_timesteps' must be between 1 and 100")
    if prompt_wav_path and not (prompt_text or "").strip():
        raise HTTPException(status_code=400, detail="'prompt_audio' requires 'prompt_text'")
    if (prompt_text or "").strip() and not prompt_wav_path:
        raise HTTPException(status_code=400, detail="'prompt_text' requires 'prompt_audio'")

    final_text = build_final_text(text, control)

    try:
        wav = engine.generate(
            text=final_text,
            reference_wav_path=reference_wav_path,
            prompt_wav_path=prompt_wav_path,
            prompt_text=(prompt_text or "").strip() or None,
            cfg_value=float(cfg_value),
            inference_timesteps=int(inference_timesteps),
            normalize=normalize,
            denoise=denoise and (reference_wav_path is not None or prompt_wav_path is not None),
            seed=seed,
        )
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — surface model errors as 500s
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    audio_bytes = encode_audio(wav, engine.sample_rate, response_format)
    return Response(
        content=audio_bytes,
        media_type=MEDIA_TYPES[response_format],
        headers={
            "X-Sample-Rate": str(engine.sample_rate),
            "X-Duration-Seconds": f"{len(wav) / engine.sample_rate:.2f}",
            "Content-Disposition": f'inline; filename="voxcpm_output.{response_format}"',
        },
    )


# ---------------------------------------------------------------------------
# App & routes
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VoxCPM API",
    description="REST API for VoxCPM tokenizer-free text-to-speech: voice design, "
    "controllable cloning, and ultimate cloning.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def index():
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"message": "VoxCPM API is running. See /docs for the API reference."})


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_id": engine.model_id if engine else None,
        "model_loaded": engine.loaded if engine else False,
        "device": engine.device if engine else None,
        "sample_rate": engine.sample_rate if engine and engine.loaded else None,
    }


@app.post("/api/tts")
async def tts(
    text: str = Form(..., description="Text to synthesize"),
    control: Optional[str] = Form(None, description="Voice/style description, e.g. 'young female voice, warm'"),
    prompt_text: Optional[str] = Form(None, description="Transcript of the prompt audio (ultimate cloning)"),
    cfg_value: float = Form(2.0, description="CFG guidance scale (1.0–3.0 recommended)"),
    inference_timesteps: int = Form(10, description="Diffusion steps (4–30 recommended)"),
    normalize: bool = Form(False, description="Normalize numbers/dates/abbreviations before synthesis"),
    denoise: bool = Form(False, description="Denoise the reference/prompt audio before cloning"),
    seed: Optional[int] = Form(None, description="Random seed for reproducible generation"),
    response_format: str = Form("wav", description="Audio format: wav, flac or ogg"),
    reference_audio: Optional[UploadFile] = File(None, description="Reference audio for voice cloning"),
    prompt_audio: Optional[UploadFile] = File(None, description="Prompt audio for ultimate cloning (with transcript)"),
):
    """Generate speech. Three modes:

    - **Voice design**: only `text` (+ optional `control`) — creates a new voice from the description.
    - **Controllable cloning**: `text` + `reference_audio` (+ optional `control`).
    - **Ultimate cloning**: `text` + `prompt_audio` + `prompt_text` (+ optional `reference_audio`).
    """
    temp_files: list = []
    try:
        reference_path = await save_upload(reference_audio, temp_files)
        prompt_path = await save_upload(prompt_audio, temp_files)
        return run_generation(
            text=text,
            control=control,
            reference_wav_path=reference_path,
            prompt_wav_path=prompt_path,
            prompt_text=prompt_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise,
            seed=seed,
            response_format=response_format,
        )
    finally:
        cleanup(temp_files)


class SpeechRequest(BaseModel):
    """OpenAI-compatible /v1/audio/speech request body."""

    model: str = Field(default="voxcpm", description="Ignored — the server's loaded model is used")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(
        default="",
        description="Voice description used as a VoxCPM control instruction "
        "(e.g. 'a calm mature male voice'). 'default' or empty for none.",
    )
    response_format: str = Field(default="wav", description="wav, flac or ogg")
    speed: Optional[float] = Field(default=None, description="Unsupported — included for API compatibility")
    cfg_value: float = Field(default=2.0, description="VoxCPM extension: CFG guidance scale")
    inference_timesteps: int = Field(default=10, description="VoxCPM extension: diffusion steps")
    seed: Optional[int] = Field(default=None, description="VoxCPM extension: random seed")


@app.post("/v1/audio/speech")
def openai_speech(req: SpeechRequest):
    """OpenAI-compatible text-to-speech endpoint.

    Works with any OpenAI client:

        client.audio.speech.create(model="voxcpm", voice="a warm female voice",
                                   input="Hello!", response_format="wav")
    """
    voice = req.voice.strip()
    control = "" if voice.lower() in {"", "default", "alloy"} else voice
    return run_generation(
        text=req.input,
        control=control,
        cfg_value=req.cfg_value,
        inference_timesteps=req.inference_timesteps,
        seed=req.seed,
        response_format=req.response_format,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def create_engine_from_args(args) -> TTSEngine:
    return TTSEngine(
        model_id=args.model_id,
        device=args.device,
        load_denoiser=not args.no_denoiser,
        optimize=not args.no_optimize,
    )


def main():
    global engine

    parser = argparse.ArgumentParser(description="VoxCPM REST API server")
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.environ.get("VOXCPM_MODEL_ID", "openbmb/VoxCPM2"),
        help="Local path or HuggingFace repo id (default: openbmb/VoxCPM2, env: VOXCPM_MODEL_ID)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("VOXCPM_DEVICE", "auto"),
        help="Runtime device: auto, cpu, mps, cuda, or cuda:N (default: auto, env: VOXCPM_DEVICE)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind address. Use 0.0.0.0 to expose the unauthenticated API to the network (default: 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--no-denoiser", action="store_true", help="Disable the ZipEnhancer denoiser")
    parser.add_argument("--no-optimize", action="store_true", help="Disable torch.compile optimization")
    parser.add_argument("--preload", action="store_true", help="Load the model at startup instead of on first request")
    args = parser.parse_args()

    engine = create_engine_from_args(args)
    if args.preload:
        engine.load()

    import uvicorn

    logger.info("Starting VoxCPM API server at http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
