"""
Pre-flight validation for VoxCPM training data manifests.

Validates JSONL manifest files before starting expensive fine-tuning jobs,
catching format issues, missing files, and data quality problems early.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ValidationResult:
    """Structured result of a manifest validation run."""

    total_samples: int = 0
    valid_samples: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    audio_durations: List[float] = field(default_factory=list)
    text_lengths: List[int] = field(default_factory=list)
    has_ref_audio: int = 0

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0 and self.valid_samples > 0


def _check_audio_file(audio_path: str, sample_rate: int) -> Optional[str]:
    """Check if an audio file exists, is readable, and matches expected sample rate.

    Returns an error message, or None if the file is valid.
    """
    if not os.path.isfile(audio_path):
        return f"Audio file not found: {audio_path}"
    try:
        import soundfile as sf

        info = sf.info(audio_path)
        if info.frames == 0:
            return f"Audio file is empty: {audio_path}"
        if info.samplerate != sample_rate:
            return (
                f"Sample rate mismatch in {audio_path}: "
                f"expected {sample_rate} Hz, got {info.samplerate} Hz"
            )
        return None
    except ImportError:
        # soundfile not available; just check existence
        return None
    except Exception as e:
        return f"Cannot read audio file {audio_path}: {e}"


def _get_audio_duration(audio_path: str) -> Optional[float]:
    """Get audio duration in seconds. Returns None if unavailable."""
    try:
        import soundfile as sf

        info = sf.info(audio_path)
        return info.duration
    except Exception:
        return None


def validate_manifest(
    manifest_path: str,
    sample_rate: int = 16_000,
    max_samples: int = 0,
    verbose: bool = False,
) -> ValidationResult:
    """Validate a JSONL training manifest file.

    Checks:
        1. File exists and is readable
        2. Each line is valid JSON
        3. Required columns present (text, audio)
        4. Audio files exist and are readable
        5. Text content is non-empty
        6. Collects duration and text length statistics
        7. Validates optional ref_audio column

    Args:
        manifest_path: Path to the JSONL manifest file.
        sample_rate: Expected audio sample rate (for informational purposes).
        max_samples: Maximum number of samples to validate (0 = all).
        verbose: Print per-sample progress.

    Returns:
        ValidationResult with errors, warnings, and statistics.
    """
    result = ValidationResult()
    path = Path(manifest_path)

    if not path.exists():
        result.errors.append(f"Manifest file not found: {manifest_path}")
        return result

    if not path.is_file():
        result.errors.append(f"Manifest path is not a file: {manifest_path}")
        return result

    manifest_dir = path.parent

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        result.errors.append(f"Cannot read manifest file: {e}")
        return result

    if not lines:
        result.errors.append("Manifest file is empty")
        return result

    samples_to_check = len(lines)
    if max_samples > 0:
        samples_to_check = min(samples_to_check, max_samples)

    missing_audio_count = 0
    empty_text_count = 0

    for i, line in enumerate(lines[:samples_to_check]):
        line = line.strip()
        if not line:
            continue

        result.total_samples += 1

        # Check JSON validity
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            result.errors.append(f"Line {i + 1}: Invalid JSON — {e}")
            continue

        if not isinstance(entry, dict):
            result.errors.append(f"Line {i + 1}: Expected JSON object, got {type(entry).__name__}")
            continue

        # Check required columns
        has_error = False

        if "text" not in entry:
            result.errors.append(f"Line {i + 1}: Missing required column 'text'")
            has_error = True

        if "audio" not in entry:
            result.errors.append(f"Line {i + 1}: Missing required column 'audio'")
            has_error = True

        if has_error:
            continue

        # Validate text
        text = entry["text"]
        if not isinstance(text, str) or not text.strip():
            empty_text_count += 1
            if empty_text_count <= 5:
                result.warnings.append(f"Line {i + 1}: Empty or non-string text")
        else:
            result.text_lengths.append(len(text))

        # Validate audio path
        audio_path = entry["audio"]
        if isinstance(audio_path, dict):
            # HuggingFace Audio format with {"path": ..., "array": ...}
            audio_path = audio_path.get("path", "")

        if isinstance(audio_path, str) and audio_path:
            # Resolve relative paths against manifest directory
            if not os.path.isabs(audio_path):
                audio_path = str(manifest_dir / audio_path)

            audio_error = _check_audio_file(audio_path, sample_rate)
            if audio_error:
                missing_audio_count += 1
                if missing_audio_count <= 5:
                    result.errors.append(f"Line {i + 1}: {audio_error}")
                has_error = True
            else:
                duration = _get_audio_duration(audio_path)
                if duration is not None:
                    result.audio_durations.append(duration)
                    if duration < 0.3:
                        result.warnings.append(
                            f"Line {i + 1}: Very short audio ({duration:.2f}s)"
                        )
                    elif duration > 30.0:
                        result.warnings.append(
                            f"Line {i + 1}: Very long audio ({duration:.1f}s), may cause OOM"
                        )
        else:
            result.errors.append(f"Line {i + 1}: Invalid audio path")
            has_error = True

        # Validate optional ref_audio
        if "ref_audio" in entry:
            ref_path = entry["ref_audio"]
            if isinstance(ref_path, dict):
                ref_path = ref_path.get("path", "")
            if isinstance(ref_path, str) and ref_path:
                if not os.path.isabs(ref_path):
                    ref_path = str(manifest_dir / ref_path)
                if os.path.isfile(ref_path):
                    result.has_ref_audio += 1
                else:
                    result.warnings.append(
                        f"Line {i + 1}: ref_audio file not found: {ref_path}"
                    )

        if not has_error:
            result.valid_samples += 1

        if verbose and (i + 1) % 100 == 0:
            print(f"  Validated {i + 1}/{samples_to_check} samples...", file=sys.stderr)

    # Summarize truncated errors
    if missing_audio_count > 5:
        result.errors.append(
            f"... and {missing_audio_count - 5} more missing audio files "
            f"({missing_audio_count} total)"
        )
    if empty_text_count > 5:
        result.warnings.append(
            f"... and {empty_text_count - 5} more empty text entries "
            f"({empty_text_count} total)"
        )

    return result


def print_validation_report(result: ValidationResult, manifest_path: str) -> None:
    """Print a human-readable validation report to stderr."""
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  VoxCPM Training Data Validation Report", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  Manifest : {manifest_path}", file=sys.stderr)
    print(f"  Samples  : {result.valid_samples}/{result.total_samples} valid", file=sys.stderr)

    if result.has_ref_audio > 0:
        print(
            f"  Ref Audio: {result.has_ref_audio} samples with reference audio",
            file=sys.stderr,
        )

    # Audio duration statistics
    if result.audio_durations:
        durations = sorted(result.audio_durations)
        total_hrs = sum(durations) / 3600
        print(f"\n  Audio Duration Statistics:", file=sys.stderr)
        print(f"    Total    : {total_hrs:.2f} hours", file=sys.stderr)
        print(
            f"    Range    : {durations[0]:.2f}s — {durations[-1]:.1f}s",
            file=sys.stderr,
        )
        print(
            f"    Mean     : {sum(durations) / len(durations):.2f}s",
            file=sys.stderr,
        )
        median_idx = len(durations) // 2
        print(f"    Median   : {durations[median_idx]:.2f}s", file=sys.stderr)

    # Text length statistics
    if result.text_lengths:
        lengths = sorted(result.text_lengths)
        print(f"\n  Text Length Statistics (characters):", file=sys.stderr)
        print(
            f"    Range    : {lengths[0]} — {lengths[-1]}",
            file=sys.stderr,
        )
        print(
            f"    Mean     : {sum(lengths) / len(lengths):.0f}",
            file=sys.stderr,
        )

    # Errors
    if result.errors:
        print(f"\n  ERRORS ({len(result.errors)}):", file=sys.stderr)
        for err in result.errors[:20]:
            print(f"    x {err}", file=sys.stderr)
        if len(result.errors) > 20:
            print(
                f"    ... ({len(result.errors) - 20} more errors omitted)",
                file=sys.stderr,
            )

    # Warnings
    if result.warnings:
        print(f"\n  WARNINGS ({len(result.warnings)}):", file=sys.stderr)
        for warn in result.warnings[:10]:
            print(f"    ! {warn}", file=sys.stderr)
        if len(result.warnings) > 10:
            print(
                f"    ... ({len(result.warnings) - 10} more warnings omitted)",
                file=sys.stderr,
            )

    # Summary
    print(f"\n{'=' * 60}", file=sys.stderr)
    if result.is_valid:
        print("  PASSED: Manifest is valid for training.", file=sys.stderr)
    else:
        print("  FAILED: Fix errors above before starting training.", file=sys.stderr)
    print(f"{'=' * 60}\n", file=sys.stderr)
