import os
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    frames: int
    duration: float


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
