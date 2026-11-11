"""
VoxCPM WebUI launcher for the ROCm container.

Runs the upstream app.py unchanged, with a few runtime tweaks controlled by
environment variables:

  PORT / HOST          Gradio bind (default 8808 / 0.0.0.0)
  VOXCPM_MODEL         HF repo id or local path (default openbmb/VoxCPM2)
  VOXCPM_DEVICE        auto | cuda | cuda:N | cpu (ROCm shows up as "cuda")
  VOXCPM_DTYPE         float16 (default) | bfloat16 | float32. RDNA2 has no
                       fast bf16 path: fp16 is ~2.7x faster end-to-end on a
                       6700 XT (RTF 1.6 vs 4.3). Empty = checkpoint dtype.
  VOXCPM_OPTIMIZE      1 = torch.compile (CUDA graphs). Off by default: on
                       gfx1031 "reduce-overhead" crashes the GPU (memory
                       access fault) during warm-up.
  ASR_BACKEND          whisper (default, good for Indonesian) | sensevoice
                       (upstream: zh/en/ja/ko/yue only)
  ASR_MODEL            whisper model id (default openai/whisper-large-v3-turbo)
  ASR_LANGUAGE         force whisper language, e.g. "id"; empty = auto-detect
  ASR_DEVICE           device for ASR (default: same GPU as VoxCPM)
  AUTO_SPLIT_WORDS     text longer than this (default 150 words) is split into
                       paragraph/sentence chunks, generated with one seed and
                       a locked voice, and joined; 0 = off (one generate call,
                       which fails beyond ~1,400 words). Long results are also
                       saved to /app/output/webui_<time>.wav/.mp3.
  AUTO_SPLIT_PAUSE     pause between paragraphs in seconds (default 0.7)
  DIT_STEPS            starting value of the "LocDiT flow-matching steps"
                       slider (default 10; fewer = faster, rougher audio)
  VOICES_DIR           voice presets folder (default /app/data/voices): each
                       audio file is a preset in the WebUI dropdown, with an
                       optional <name>.txt transcript for Ultimate Cloning
  TEXT_NUMBER_LANG     expand digits to words before synthesis, e.g. "id"
                       ("2026" -> "dua ribu dua puluh enam"); empty = off.
                       Upstream "normalize" only knows zh/en, so leave the
                       Text Normalization checkbox off for Indonesian.
"""

import argparse
import logging
import os
import re
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(APP_DIR, "app.py")) and APP_DIR != "/":
    APP_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)

import app as vox  # noqa: E402
import voxcpm.model.voxcpm as _vox_m1  # noqa: E402
import voxcpm.model.voxcpm2 as _vox_m2  # noqa: E402

logger = logging.getLogger("voxcpm.launch")


# ── runtime dtype override ───────────────────────────────────────────
def _make_dtype_picker(orig):
    def pick(device, configured_dtype):
        forced = os.environ.get("VOXCPM_DTYPE", "").strip().lower()
        if forced and device.startswith("cuda"):
            return forced
        return orig(device, configured_dtype)

    return pick


for _mod in (_vox_m1, _vox_m2):
    if hasattr(_mod, "pick_runtime_dtype"):
        _mod.pick_runtime_dtype = _make_dtype_picker(_mod.pick_runtime_dtype)


def env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ── torch.compile toggle ─────────────────────────────────────────────
_orig_init = vox.VoxCPMDemo.__init__


def _init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self.optimize = env_bool("VOXCPM_OPTIMIZE") and self.device.startswith("cuda")
    if os.environ.get("ASR_DEVICE"):
        self.asr_device = os.environ["ASR_DEVICE"]
    self._whisper = None
    logger.info(f"torch.compile optimize={self.optimize}, ASR={os.environ.get('ASR_BACKEND', 'whisper')}")


vox.VoxCPMDemo.__init__ = _init


# ── Whisper ASR (multilingual prompt transcription) ──────────────────
def _get_whisper(self):
    if self._whisper is not None:
        return self._whisper
    import torch
    from transformers import pipeline

    model_id = os.environ.get("ASR_MODEL", "openai/whisper-large-v3-turbo")
    on_gpu = self.asr_device.startswith("cuda")
    logger.info(f"Loading Whisper ASR: {model_id} on {self.asr_device}")
    self._whisper = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=self.asr_device,
        torch_dtype=torch.float16 if on_gpu else torch.float32,
    )
    return self._whisper


_orig_asr = vox.VoxCPMDemo.prompt_wav_recognition


def _prompt_wav_recognition(self, prompt_wav):
    if os.environ.get("ASR_BACKEND", "whisper").lower() != "whisper":
        return _orig_asr(self, prompt_wav)
    if not prompt_wav:
        return ""
    gen_kwargs = {"task": "transcribe"}
    lang = os.environ.get("ASR_LANGUAGE", "").strip()
    if lang:
        gen_kwargs["language"] = lang
    try:
        out = _get_whisper(self)(prompt_wav, chunk_length_s=30, generate_kwargs=gen_kwargs)
    finally:
        # Whisper is needed for a few seconds per reference; keeping it on the
        # GPU costs ~1.5 GB that long renders need (reloading takes a few s).
        if self.asr_device.startswith("cuda"):
            _free_gpu_model(self, "_whisper")
    return out["text"].strip()


def _free_gpu_model(obj, attr):
    import gc

    import torch

    setattr(obj, attr, None)
    gc.collect()
    torch.cuda.empty_cache()


vox.VoxCPMDemo.prompt_wav_recognition = _prompt_wav_recognition


# ── Number expansion (Indonesian by default) ─────────────────────────
_NUM_RE = re.compile(r"(?<![\w.,])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(?![\w])")


def expand_numbers(text: str, lang: str) -> str:
    from num2words import num2words

    comma = {"id": "koma", "ms": "perpuluhan"}.get(lang, "point")
    percent = {"id": "persen", "ms": "peratus"}.get(lang, "percent")

    def repl(m):
        whole = int(m.group(1).replace(".", ""))
        words = num2words(whole, lang=lang)
        if m.group(2):
            # read decimals digit by digit: 3,25 -> tiga koma dua lima
            words += f" {comma} " + " ".join(num2words(int(d), lang=lang) for d in m.group(2))
        return words

    text = re.sub(r"(\d)\s*%", rf"\1 {percent}", text)
    if lang == "id":
        # Rp 1.500 -> 1.500 rupiah
        text = re.sub(r"\bRp\.?\s*(\d[\d.,]*\d|\d)", r"\1 rupiah", text)
        # 07.30 / 7:30 -> tujuh tiga puluh (only HH.MM with a valid hour/minute)
        text = re.sub(
            r"(?<![\d.,])([01]?\d|2[0-3])[.:]([0-5]\d)(?!\d|[.,]\d)",
            lambda m: num2words(int(m.group(1)), lang=lang)
            + ("" if m.group(2) == "00" else " " + num2words(int(m.group(2)), lang=lang)),
            text,
        )
    return _NUM_RE.sub(repl, text)


_orig_build = vox.VoxCPMDemo._build_generate_kwargs


def _build_generate_kwargs(self, *, final_text, **kwargs):
    lang = os.environ.get("TEXT_NUMBER_LANG", "").strip()
    if lang and final_text:
        try:
            final_text = expand_numbers(final_text, lang)
        except Exception as e:  # never block synthesis on normalization
            logger.warning(f"number expansion failed: {e}")
    return _orig_build(self, final_text=final_text, **kwargs)


vox.VoxCPMDemo._build_generate_kwargs = _build_generate_kwargs


# ── Long text: split, generate chunk by chunk, join ──────────────────
# One generate() call tops out around 1,400 words (max_len 4096 steps of
# 0.16 s, 8192-token KV cache), and decoding a long latent in one go needs
# several GB of extra VRAM. Longer text is split into paragraph/sentence
# chunks, each generated with the same seed and a locked voice, then joined.
_SENT_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+")


def _split_long(text: str, max_words: int) -> list:
    if len(text.split()) <= max_words:
        return [text]
    out, cur = [], []
    for s in (x for x in _SENT_END.split(text) if x.strip()):
        words = s.split()
        sub = [s]
        if len(words) > max_words:
            sub = re.split(r"(?<=[,;:])\s+", s)
            if any(len(x.split()) > max_words for x in sub):
                sub = [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]
        for x in sub:
            if cur and len(" ".join(cur + [x]).split()) > max_words:
                out.append(" ".join(cur))
                cur = []
            cur.append(x)
    if cur:
        out.append(" ".join(cur))
    return out


def split_text(raw: str, max_words: int) -> list:
    """[(chunk_text, ends_paragraph)] — paragraphs are blank-line separated."""
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", raw)]
    chunks, buf = [], ""
    for p in (p for p in paras if p):
        if len(p.split()) > max_words:
            if buf:
                chunks.append((buf, True))
                buf = ""
            parts = _split_long(p, max_words)
            chunks += [(x, i == len(parts) - 1) for i, x in enumerate(parts)]
        elif buf and len((buf + " " + p).split()) > max_words:
            chunks.append((buf, True))
            buf = p
        else:
            buf = f"{buf} {p}".strip()
    if buf:
        chunks.append((buf, True))
    return chunks


_orig_generate_tts = vox.VoxCPMDemo.generate_tts_audio


def _generate_tts_audio(
    self,
    text_input,
    control_instruction="",
    reference_wav_path_input=None,
    prompt_text="",
    cfg_value_input=2.0,
    do_normalize=True,
    denoise=True,
    inference_timesteps=10,
    seed=None,
):
    import random
    import tempfile
    import time

    import numpy as np
    import soundfile as sf
    import torch

    import gc

    max_words = int(os.environ.get("AUTO_SPLIT_WORDS", "150") or 0)
    chunks = split_text(text_input or "", max_words) if max_words > 0 else []
    if len(chunks) <= 1:
        try:
            return _orig_generate_tts(
                self, text_input, control_instruction, reference_wav_path_input, prompt_text,
                cfg_value_input, do_normalize, denoise, inference_timesteps, seed,
            )
        except torch.OutOfMemoryError:
            # fall through to the chunked path, which retries in smaller parts
            gc.collect()
            torch.cuda.empty_cache()
            logger.warning("[long text] out of GPU memory in one call, retrying in smaller parts")
            chunks = [(re.sub(r"\s+", " ", text_input).strip(), True)]

    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    words = sum(len(c.split()) for c, _ in chunks)
    logger.info(f"[long text] {words} words -> {len(chunks)} chunks, seed {seed}")
    ref, user_ref = reference_wav_path_input, bool(reference_wav_path_input)
    pieces, sr, t0 = [], None, time.time()
    tmpdir = tempfile.mkdtemp(prefix="voxcpm_long_")

    def gen_piece(text, ref_wav, depth=0):
        """Generate one chunk; on GPU OOM free memory and retry it as halves."""
        oom = None
        try:
            sr_, wav_, _ = _orig_generate_tts(
                self,
                text,
                control_instruction,
                ref_wav,
                prompt_text if user_ref else "",
                cfg_value_input,
                do_normalize,
                denoise and user_ref,  # only the user's recording needs denoising
                inference_timesteps,
                seed,
            )
            return sr_, np.asarray(wav_, dtype=np.float32)
        except torch.OutOfMemoryError as e:
            oom = str(e).split(".")[0]
        # Retry outside the except block: the live traceback would otherwise
        # keep the failed attempt's tensors allocated during the retry.
        gc.collect()
        torch.cuda.empty_cache()
        n = len(text.split())
        if n < 8 or depth >= 4:
            raise torch.OutOfMemoryError(f"{oom} (even after splitting down to {n} words)")
        parts = _split_long(text, (n + 1) // 2)
        if len(parts) < 2:
            w = text.split()
            parts = [" ".join(w[: n // 2]), " ".join(w[n // 2 :])]
        logger.warning(f"[long text] out of GPU memory on {n} words, retrying as {len(parts)} smaller parts")
        out, sr_ = [], None
        for j, part in enumerate(parts):
            sr_, w = gen_piece(part, ref_wav, depth + 1)
            out.append(w)
            if j < len(parts) - 1:
                out.append(np.zeros(int(sr_ * 0.25), dtype=np.float32))
        return sr_, np.concatenate(out)

    for i, (chunk, para_end) in enumerate(chunks, 1):
        t = time.time()
        sr, wav = gen_piece(chunk, ref)
        pieces.append(wav)
        if i < len(chunks):
            pause = float(os.environ.get("AUTO_SPLIT_PAUSE", "0.7")) if para_end else 0.25
            pieces.append(np.zeros(int(sr * pause), dtype=np.float32))
        if ref is None:
            # lock the voice: the first 12 s of chunk 1 become the reference
            ref = os.path.join(tmpdir, "voice_ref.wav")
            sf.write(ref, wav[: int(sr * 12)], sr)
        torch.cuda.empty_cache()
        done = time.time() - t0
        logger.info(
            f"[long text] chunk {i}/{len(chunks)}: {len(wav) / sr:.1f} s audio in {time.time() - t:.0f} s,"
            f" about {done / i * (len(chunks) - i) / 60:.0f} min left"
        )

    audio = np.concatenate(pieces)
    peak = float(np.abs(audio).max())
    if peak > 0.99:
        audio *= 0.99 / peak
    # Long renders also land in /app/output, so a closed browser tab doesn't lose hours of work.
    outdir = os.environ.get("AUTO_SPLIT_OUTDIR", "/app/output")
    if os.path.isdir(outdir) and os.access(outdir, os.W_OK):
        base = os.path.join(outdir, time.strftime("webui_%Y%m%d_%H%M%S"))
        sf.write(base + ".wav", audio, sr)
        try:
            import subprocess

            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", base + ".wav", "-codec:a", "libmp3lame", "-q:a", "2", base + ".mp3"],
                check=True,
            )
        except Exception as e:
            logger.warning(f"mp3 encode failed: {e}")
        logger.info(f"[long text] saved {base}.wav / .mp3 ({len(audio) / sr / 60:.1f} min)")
    return sr, audio, seed


vox.VoxCPMDemo.generate_tts_audio = _generate_tts_audio


# ── Default for the "LocDiT flow-matching steps" slider ──────────────
# app.py hard-codes value=10. DIT_STEPS changes only the slider's starting
# value; users can still move it per request.
_orig_create_ui = vox.create_demo_interface


def _create_demo_interface(demo):
    steps = os.environ.get("DIT_STEPS", "").strip()
    if not steps:
        return _orig_create_ui(demo)
    steps = max(1, min(50, int(steps)))
    real_slider = vox.gr.Slider

    def slider(*args, **kwargs):
        # the DiT slider is the only one with range 1-50 and default 10
        if kwargs.get("minimum") == 1 and kwargs.get("maximum") == 50 and kwargs.get("value") == 10:
            kwargs["value"] = steps
        return real_slider(*args, **kwargs)

    vox.gr.Slider = slider
    try:
        return _orig_create_ui(demo)
    finally:
        vox.gr.Slider = real_slider


vox.create_demo_interface = _create_demo_interface


# ── Voice presets ────────────────────────────────────────────────────
# Audio files in VOICES_DIR (default /app/data/voices) show up in a "Voice
# preset" dropdown above the reference audio. Picking one fills the reference
# audio, turns on Ultimate Cloning and fills the transcript from <name>.txt.
# Without a .txt, Whisper transcribes it once and the result is saved there
# (editable). Presets are matched by file content, so Gradio's copy of the
# file still gets the stored transcript instead of a fresh Whisper run.
VOICES_DIR = os.environ.get("VOICES_DIR", "/app/data/voices")
_AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")
_NO_PRESET = "—"


def _voice_presets() -> dict:
    """{name: audio_path} for every audio file in VOICES_DIR."""
    if not os.path.isdir(VOICES_DIR):
        return {}
    return {
        os.path.splitext(f)[0]: os.path.join(VOICES_DIR, f)
        for f in sorted(os.listdir(VOICES_DIR))
        if f.lower().endswith(_AUDIO_EXT) and not f.startswith(".")
    }


def _file_sha1(path: str) -> str:
    import hashlib

    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _preset_for_audio(path: str):
    """Preset audio path whose content equals `path`, else None."""
    if not path or not os.path.isfile(path):
        return None
    size, digest = os.path.getsize(path), None
    for preset in _voice_presets().values():
        if os.path.getsize(preset) == size:
            digest = digest or _file_sha1(path)
            if _file_sha1(preset) == digest:
                return preset
    return None


_asr_without_presets = vox.VoxCPMDemo.prompt_wav_recognition


def _prompt_wav_recognition_presets(self, prompt_wav):
    preset = _preset_for_audio(prompt_wav)
    if preset is None:
        return _asr_without_presets(self, prompt_wav)
    txt = os.path.splitext(preset)[0] + ".txt"
    if os.path.isfile(txt):
        return open(txt, encoding="utf-8").read().strip()
    text = _asr_without_presets(self, preset)
    try:
        with open(txt, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        logger.info(f"[voice preset] saved transcript {txt}")
    except OSError as e:
        logger.warning(f"[voice preset] could not save {txt}: {e}")
    return text


vox.VoxCPMDemo.prompt_wav_recognition = _prompt_wav_recognition_presets

_create_ui_without_presets = vox.create_demo_interface


def _create_demo_interface_presets(demo):
    gr = vox.gr
    real_audio, found = gr.Audio, {}

    def audio(*args, **kwargs):
        # the reference input is the only Audio with upload + microphone sources
        if kwargs.get("sources") == ["upload", "microphone"] and "preset" not in found:
            found["preset"] = gr.Dropdown(
                choices=[_NO_PRESET] + list(_voice_presets()),
                value=_NO_PRESET,
                label="Voice preset",
                info=f"Audio files in {VOICES_DIR} (refresh the page after adding one)",
            )
            found["ref"] = real_audio(*args, **kwargs)
            return found["ref"]
        return real_audio(*args, **kwargs)

    gr.Audio = audio
    try:
        interface = _create_ui_without_presets(demo)
    finally:
        gr.Audio = real_audio
    if "preset" not in found:
        logger.warning("[voice preset] reference audio component not found; presets disabled")
        return interface

    # find the Ultimate Cloning checkbox and transcript box by their place in
    # the layout: they are the two components right after the reference audio
    blocks = list(interface.blocks.values())
    i = blocks.index(found["ref"])
    use_prompt, prompt_box = blocks[i + 1], blocks[i + 2]

    def pick(name):
        path = _voice_presets().get(name)
        if not path:
            return None, False, ""
        return path, True, demo.prompt_wav_recognition(path)

    with interface:
        found["preset"].change(
            fn=pick,
            inputs=[found["preset"]],
            outputs=[found["ref"], use_prompt, prompt_box],
        )
        interface.load(
            fn=lambda: gr.update(choices=[_NO_PRESET] + list(_voice_presets())),
            outputs=[found["preset"]],
        )
    return interface


vox.create_demo_interface = _create_demo_interface_presets

# Gradio only serves files from allowed paths; presets live in VOICES_DIR.
_real_launch = vox.gr.Blocks.launch


def _launch(self, *args, **kwargs):
    kwargs["allowed_paths"] = list(kwargs.get("allowed_paths") or []) + [VOICES_DIR]
    return _real_launch(self, *args, **kwargs)


vox.gr.Blocks.launch = _launch


def check_gpu_kernels():
    """Fail fast with a useful hint instead of 'invalid device function' mid-request."""
    import torch

    if not torch.cuda.is_available():
        logger.warning("No ROCm GPU visible (check /dev/kfd, /dev/dri and render group); running on CPU")
        return
    props = torch.cuda.get_device_properties(0)
    try:
        (torch.ones(8, device="cuda") * 2).sum().item()
    except RuntimeError as e:
        override = os.environ.get("HSA_OVERRIDE_GFX_VERSION") or "unset"
        sys.exit(
            f"GPU {props.name} ({props.gcnArchName}) can't run the bundled ROCm kernels: {e}\n"
            f"HSA_OVERRIDE_GFX_VERSION={override}. RDNA2 cards (gfx1031-1036) need 10.3.0, "
            "RDNA3 gfx1101-1103 need 11.0.0; docker/rocm/run.sh sets this automatically."
        )
    logger.info(f"GPU: {props.name} ({props.gcnArchName}), {props.total_memory / 2**30:.1f} GB")


if __name__ == "__main__":
    check_gpu_kernels()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8808")))
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--device", default=os.environ.get("VOXCPM_DEVICE", "auto"))
    args = parser.parse_args()
    vox.run_demo(
        model_id=args.model_id,
        server_name=args.host,
        server_port=args.port,
        device=args.device,
    )
