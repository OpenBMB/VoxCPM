import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import soundfile as sf

CACHE_DIR_NAME = "_voxcpm_cache"
MANIFEST_FILENAME = "voxcpm_manifest.json"


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    frames: int
    duration: float


def audio_fingerprint(path):
    if not path:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return {"path": os.path.abspath(path), "missing": True}
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def payload_cache_identity(payload):
    return {
        "text": payload.get("text"),
        "model_id": payload.get("model_id"),
        "prompt_text": payload.get("prompt_text"),
        "prompt_wav": audio_fingerprint(payload.get("prompt_wav_path")),
        "reference_wav": audio_fingerprint(payload.get("reference_wav_path")),
        "cfg_value": payload.get("cfg_value"),
        "inference_timesteps": payload.get("inference_timesteps"),
        "normalize": payload.get("normalize"),
        "denoise": payload.get("denoise"),
        "trim_silence_vad": payload.get("trim_silence_vad"),
    }


def payload_cache_key(payload):
    body = json.dumps(payload_cache_identity(payload), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def cache_manifest_path(output_dir):
    return os.path.join(output_dir, MANIFEST_FILENAME)


def cache_wav_path(output_dir, cache_key):
    return os.path.join(output_dir, CACHE_DIR_NAME, f"{cache_key}.wav")


def load_cache_manifest(output_dir):
    path = cache_manifest_path(output_dir)
    if not os.path.exists(path):
        return {"version": 1, "entries": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "entries": {}}
    data.setdefault("version", 1)
    data.setdefault("entries", {})
    return data


def save_cache_manifest(output_dir, manifest):
    ensure_dirs(output_dir)
    with open(cache_manifest_path(output_dir), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def _copy_wav(source, destination):
    if os.path.abspath(source) == os.path.abspath(destination):
        return
    ensure_dirs(os.path.dirname(destination))
    shutil.copy2(source, destination)


def prepare_cached_wav(output_dir, block_path, payload):
    cache_key = payload_cache_key(payload)
    cache_path = cache_wav_path(output_dir, cache_key)
    manifest = load_cache_manifest(output_dir)
    entry = manifest.get("entries", {}).get(cache_key, {})

    if os.path.exists(cache_path):
        _copy_wav(cache_path, block_path)
        return cache_key, "hash-cache"

    manifest_path = entry.get("path")
    if manifest_path and os.path.exists(manifest_path):
        _copy_wav(manifest_path, cache_path)
        _copy_wav(cache_path, block_path)
        return cache_key, "manifest"

    if os.path.exists(block_path):
        # Legacy compatibility: existing index-based WAVs are trusted once,
        # then promoted into the hash cache for future runs.
        _copy_wav(block_path, cache_path)
        return cache_key, "existing-path"

    return cache_key, None


def record_cached_wav(output_dir, cache_key, payload, block_path, info, metrics=None):
    cache_path = cache_wav_path(output_dir, cache_key)
    if os.path.exists(block_path) and not os.path.exists(cache_path):
        _copy_wav(block_path, cache_path)

    manifest = load_cache_manifest(output_dir)
    manifest["entries"][cache_key] = {
        "path": os.path.abspath(block_path),
        "cache_path": os.path.abspath(cache_path),
        "text": payload.get("text"),
        "text_chars": len(payload.get("text") or ""),
        "identity": payload_cache_identity(payload),
        "sample_rate": info.sample_rate,
        "frames": info.frames,
        "duration": info.duration,
        "metrics": metrics or {},
    }
    save_cache_manifest(output_dir, manifest)


def audit_audio_cache(blocks, output_dir, payload_builder, block_path_builder):
    reusable = 0
    missing = 0
    for index, block in enumerate(blocks, start=1):
        payload = payload_builder(block)
        cache_key = payload_cache_key(payload)
        cache_path = cache_wav_path(output_dir, cache_key)
        block_path = block_path_builder(index)
        entry = load_cache_manifest(output_dir).get("entries", {}).get(cache_key, {})
        manifest_path = entry.get("path")
        if (
            os.path.exists(cache_path)
            or os.path.exists(block_path)
            or (manifest_path and os.path.exists(manifest_path))
        ):
            reusable += 1
        else:
            missing += 1
    return {"total": len(blocks), "reusable": reusable, "missing": missing}


def split_text_blocks(text, max_chars):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    blocks = []
    current = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            blocks.append(current)

        if len(sentence) > max_chars:
            parts = re.split(r"(?<=[,;])\s+", sentence)
            sub_block = ""
            for part in parts:
                candidate = (sub_block + " " + part).strip() if sub_block else part
                if len(candidate) <= max_chars:
                    sub_block = candidate
                else:
                    if sub_block:
                        blocks.append(sub_block)
                    sub_block = part
            current = sub_block
        else:
            current = sentence

    if current:
        blocks.append(current)

    return blocks


def read_text(path, encoding="ansi"):
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def read_paragraph_blocks(path, max_chars, encoding="ansi"):
    content = read_text(path, encoding=encoding)
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    blocks = []
    for paragraph in paragraphs:
        blocks.extend(split_text_blocks(paragraph, max_chars))
    return paragraphs, blocks


def parse_structured_shorts(path, encoding="ansi"):
    content = read_text(path, encoding=encoding)
    sections = re.split(r"^---\s*$", content, flags=re.MULTILINE)
    shorts = []

    for section in sections:
        section = section.strip()
        if not section:
            continue
        match = re.search(r"^Script:\s*(.+)", section, re.MULTILINE)
        if match:
            text = match.group(1).strip()
            if text:
                shorts.append(text)

    return shorts


def seconds_to_srt(seconds):
    total_ms = int(round(seconds * 1000))
    h = total_ms // 3_600_000
    total_ms %= 3_600_000
    m = total_ms // 60_000
    total_ms %= 60_000
    sec = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def build_srt_entries(entries):
    srt_lines = []
    cursor = 0.0

    for idx, (text, duration) in enumerate(entries, start=1):
        start = seconds_to_srt(cursor)
        end = seconds_to_srt(cursor + duration)
        srt_lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
        cursor += duration

    return srt_lines


def write_srt(entries, output_path, encoding="ansi"):
    srt_lines = build_srt_entries(entries)
    with open(output_path, "w", encoding=encoding) as f:
        f.write("\n".join(srt_lines))
    return len(srt_lines)


def get_audio_info(path):
    info = sf.info(path)
    return AudioInfo(sample_rate=info.samplerate, frames=info.frames, duration=info.duration)


def read_wav(path):
    return sf.read(path)


def write_wav_bytes(path, wav_bytes):
    with open(path, "wb") as f:
        f.write(wav_bytes)


def concatenate_wavs(wavs: Iterable[np.ndarray]):
    return np.concatenate(list(wavs))


def build_voxcpm_payload(
    text,
    model_id,
    prompt_text,
    reference_wav,
    cfg_value,
    inference_timesteps,
    normalize,
):
    return {
        "text": text,
        "model_id": model_id,
        "prompt_text": prompt_text,
        "prompt_wav_path": reference_wav,
        "reference_wav_path": reference_wav,
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
        "normalize": normalize,
        "denoise": False,
        "trim_silence_vad": False,
    }


def ensure_dirs(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)
