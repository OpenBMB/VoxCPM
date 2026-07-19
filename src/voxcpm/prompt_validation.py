from __future__ import annotations

import hashlib
import os
import unicodedata
import warnings

PROMPT_MIN_FEATURE_TOKEN_RATIO = 0.25
PROMPT_MAX_FEATURE_TOKEN_RATIO = 8.0
PROMPT_MAX_NORMALIZED_DISTANCE = 0.35
_VALID_PROMPT_VALIDATION_MODES = {"off", "warn", "error"}


class PromptMismatchError(ValueError):
    """Raised when prompt audio and its transcript cannot be safely paired."""


def audio_file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable SHA-256 fingerprint for the exact audio file bytes."""
    if not path or not os.path.isfile(path):
        raise PromptMismatchError(f"Prompt audio does not exist: {path}")

    digest = hashlib.sha256()
    with open(path, "rb") as audio_file:
        for chunk in iter(lambda: audio_file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt_text(text: str) -> str:
    """Normalize transcript text for language-agnostic character comparison."""
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return "".join(char for char in normalized if char.isalnum())


def _levenshtein_distance(left: str, right: str) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, right_char in enumerate(right, start=1):
        current = [row]
        for column, left_char in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def normalized_text_distance(left: str, right: str) -> float:
    """Return normalized character edit distance in the inclusive range [0, 1]."""
    left_normalized = normalize_prompt_text(left)
    right_normalized = normalize_prompt_text(right)
    if not left_normalized and not right_normalized:
        return 0.0
    if not left_normalized or not right_normalized:
        return 1.0
    denominator = max(len(left_normalized), len(right_normalized))
    return _levenshtein_distance(left_normalized, right_normalized) / denominator


def validate_prompt_transcript(
    *,
    audio_path: str,
    prompt_text: str,
    recognized_text: str,
    recognized_audio_hash: str,
    max_distance: float = PROMPT_MAX_NORMALIZED_DISTANCE,
) -> float:
    """Verify that a transcript belongs to the current audio and matches fresh ASR."""
    if not 0 <= max_distance <= 1:
        raise ValueError("max_distance must be between 0 and 1")
    if not recognized_audio_hash:
        raise PromptMismatchError("Prompt transcript is not bound to an audio fingerprint; run ASR again.")

    current_audio_hash = audio_file_sha256(audio_path)
    if current_audio_hash != recognized_audio_hash:
        raise PromptMismatchError("Reference audio changed after ASR; run transcription again before generation.")

    if not normalize_prompt_text(prompt_text):
        raise PromptMismatchError("Prompt transcript is empty after normalization.")
    if not normalize_prompt_text(recognized_text):
        raise PromptMismatchError("ASR transcript is empty after normalization.")

    distance = normalized_text_distance(prompt_text, recognized_text)
    if distance > max_distance:
        raise PromptMismatchError(
            "Prompt transcript does not match the current audio "
            f"(normalized edit distance {distance:.3f} > {max_distance:.3f}). "
            "Run ASR again, correct the transcript, or use reference-only mode."
        )
    return distance


def validate_prompt_feature_ratio(
    audio_feature_length: int,
    text_token_length: int,
    *,
    mode: str = "warn",
    min_ratio: float = PROMPT_MIN_FEATURE_TOKEN_RATIO,
    max_ratio: float = PROMPT_MAX_FEATURE_TOKEN_RATIO,
) -> float:
    """Validate broad prompt audio-feature/text-token length plausibility."""
    normalized_mode = (mode or "").strip().lower()
    if normalized_mode not in _VALID_PROMPT_VALIDATION_MODES:
        raise ValueError(f"prompt_validation must be one of {sorted(_VALID_PROMPT_VALIDATION_MODES)}")
    if audio_feature_length < 1:
        raise PromptMismatchError("Prompt audio produced no usable features.")
    if text_token_length < 1:
        raise PromptMismatchError("Prompt transcript produced no text tokens.")
    if min_ratio <= 0 or max_ratio <= min_ratio:
        raise ValueError("prompt ratio bounds must satisfy 0 < min_ratio < max_ratio")

    ratio = audio_feature_length / text_token_length
    if normalized_mode == "off" or min_ratio <= ratio <= max_ratio:
        return ratio

    message = (
        "Prompt audio/text length mismatch: "
        f"feature_token_ratio={ratio:.3f}, expected {min_ratio:.3f}..{max_ratio:.3f}. "
        "Use the exact transcript or switch to reference-only mode."
    )
    if normalized_mode == "error":
        raise PromptMismatchError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return ratio
