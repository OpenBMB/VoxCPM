"""Tests for the training data validation module."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Stub voxcpm package so imports work without full dependencies
pkg = types.ModuleType("voxcpm")
pkg.__path__ = [str(ROOT / "src" / "voxcpm")]
sys.modules.setdefault("voxcpm", pkg)

training_pkg = types.ModuleType("voxcpm.training")
training_pkg.__path__ = [str(ROOT / "src" / "voxcpm" / "training")]
sys.modules.setdefault("voxcpm.training", training_pkg)

from voxcpm.training.validate import ValidationResult, validate_manifest


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _create_wav(path: Path, duration_s: float = 1.0, sr: int = 16000):
    """Create a minimal valid WAV file."""
    try:
        import soundfile as sf
        import numpy as np

        samples = int(duration_s * sr)
        data = np.zeros(samples, dtype=np.float32)
        sf.write(str(path), data, sr)
    except ImportError:
        # If soundfile is not available, create a minimal WAV header
        import struct

        samples = int(duration_s * sr)
        data_size = samples * 2  # 16-bit PCM
        with open(path, "wb") as f:
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + data_size))
            f.write(b"WAVEfmt ")
            f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            f.write(b"\x00" * data_size)


def _write_manifest(path: Path, entries: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class TestValidateManifest:
    def test_valid_manifest(self, tmp_dir):
        audio1 = tmp_dir / "audio1.wav"
        audio2 = tmp_dir / "audio2.wav"
        _create_wav(audio1, 2.0)
        _create_wav(audio2, 3.0)

        manifest = tmp_dir / "train.jsonl"
        _write_manifest(
            manifest,
            [
                {"text": "Hello world", "audio": str(audio1)},
                {"text": "Goodbye world", "audio": str(audio2)},
            ],
        )

        result = validate_manifest(str(manifest))
        assert result.total_samples == 2
        assert result.valid_samples == 2
        assert result.is_valid
        assert len(result.errors) == 0

    def test_missing_manifest(self):
        result = validate_manifest("/nonexistent/path.jsonl")
        assert not result.is_valid
        assert any("not found" in e for e in result.errors)

    def test_empty_manifest(self, tmp_dir):
        manifest = tmp_dir / "empty.jsonl"
        manifest.write_text("")
        result = validate_manifest(str(manifest))
        assert not result.is_valid

    def test_invalid_json(self, tmp_dir):
        manifest = tmp_dir / "bad.jsonl"
        manifest.write_text("not json\n{bad json}\n")
        result = validate_manifest(str(manifest))
        assert len(result.errors) >= 2
        assert any("Invalid JSON" in e for e in result.errors)

    def test_missing_columns(self, tmp_dir):
        manifest = tmp_dir / "missing.jsonl"
        _write_manifest(
            manifest,
            [
                {"text": "hello"},  # missing audio
                {"audio": "test.wav"},  # missing text
            ],
        )
        result = validate_manifest(str(manifest))
        assert len(result.errors) >= 2
        assert any("'audio'" in e for e in result.errors)
        assert any("'text'" in e for e in result.errors)

    def test_missing_audio_file(self, tmp_dir):
        manifest = tmp_dir / "missing_audio.jsonl"
        _write_manifest(
            manifest,
            [{"text": "hello", "audio": "/nonexistent/audio.wav"}],
        )
        result = validate_manifest(str(manifest))
        assert not result.is_valid
        assert any("not found" in e for e in result.errors)

    def test_empty_text_warning(self, tmp_dir):
        audio = tmp_dir / "audio.wav"
        _create_wav(audio)
        manifest = tmp_dir / "empty_text.jsonl"
        _write_manifest(
            manifest,
            [{"text": "", "audio": str(audio)}],
        )
        result = validate_manifest(str(manifest))
        assert len(result.warnings) > 0
        assert any("Empty" in w for w in result.warnings)

    def test_relative_audio_path(self, tmp_dir):
        audio = tmp_dir / "audio.wav"
        _create_wav(audio)
        manifest = tmp_dir / "rel.jsonl"
        _write_manifest(
            manifest,
            [{"text": "hello", "audio": "audio.wav"}],
        )
        result = validate_manifest(str(manifest))
        assert result.valid_samples == 1
        assert result.is_valid

    def test_max_samples_limit(self, tmp_dir):
        audio = tmp_dir / "audio.wav"
        _create_wav(audio)
        manifest = tmp_dir / "many.jsonl"
        _write_manifest(
            manifest,
            [{"text": f"sample {i}", "audio": str(audio)} for i in range(100)],
        )
        result = validate_manifest(str(manifest), max_samples=10)
        assert result.total_samples == 10

    def test_ref_audio_counted(self, tmp_dir):
        audio = tmp_dir / "audio.wav"
        ref = tmp_dir / "ref.wav"
        _create_wav(audio)
        _create_wav(ref)
        manifest = tmp_dir / "ref.jsonl"
        _write_manifest(
            manifest,
            [{"text": "hello", "audio": str(audio), "ref_audio": str(ref)}],
        )
        result = validate_manifest(str(manifest))
        assert result.has_ref_audio == 1

    def test_validation_result_properties(self):
        r = ValidationResult(total_samples=5, valid_samples=5)
        assert r.is_valid

        r2 = ValidationResult(total_samples=5, valid_samples=5, errors=["err"])
        assert not r2.is_valid

        r3 = ValidationResult(total_samples=0, valid_samples=0)
        assert not r3.is_valid

    def test_invalid_audio_not_counted_as_valid(self, tmp_dir):
        """A row with a bad audio path must not increment valid_samples."""
        manifest = tmp_dir / "bad_audio.jsonl"
        _write_manifest(
            manifest,
            [{"text": "hello", "audio": "/nonexistent/audio.wav"}],
        )
        result = validate_manifest(str(manifest))
        assert result.total_samples == 1
        assert result.valid_samples == 0
        assert not result.is_valid
        assert any("not found" in e for e in result.errors)

    def test_sample_rate_mismatch(self, tmp_dir):
        """A file with a different sample rate should be reported as an error."""
        try:
            import soundfile as sf
            import numpy as np
        except ImportError:
            pytest.skip("soundfile not available")

        audio = tmp_dir / "audio_8k.wav"
        import numpy as np
        samples = np.zeros(8000, dtype=np.float32)
        sf.write(str(audio), samples, 8000)

        manifest = tmp_dir / "sr_mismatch.jsonl"
        _write_manifest(manifest, [{"text": "hello", "audio": str(audio)}])

        result = validate_manifest(str(manifest), sample_rate=16000)
        assert result.valid_samples == 0
        assert not result.is_valid
        assert any("Sample rate mismatch" in e or "sample rate" in e.lower() for e in result.errors)

    def test_mixed_ref_audio_warns_for_each_missing(self, tmp_dir):
        """Missing ref_audio entries should each generate a warning independently."""
        audio = tmp_dir / "audio.wav"
        ref_good = tmp_dir / "ref_good.wav"
        _create_wav(audio)
        _create_wav(ref_good)

        manifest = tmp_dir / "mixed_ref.jsonl"
        _write_manifest(
            manifest,
            [
                {"text": "row1", "audio": str(audio), "ref_audio": str(ref_good)},
                {"text": "row2", "audio": str(audio), "ref_audio": "/nonexistent/ref.wav"},
            ],
        )
        result = validate_manifest(str(manifest))
        assert result.has_ref_audio == 1
        assert any("ref_audio file not found" in w for w in result.warnings)

    def test_cli_validate_exit_code(self, tmp_dir):
        """validate subcommand must exit 1 on validation error (missing audio)."""
        import subprocess
        manifest = tmp_dir / "bad.jsonl"
        _write_manifest(manifest, [{"text": "hi", "audio": "/nonexistent/x.wav"}])

        proc = subprocess.run(
            [sys.executable, "-m", "voxcpm.cli", "validate", "--manifest", str(manifest)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 1, f"Expected exit 1, got {proc.returncode}"
        assert "FAILED" in proc.stderr or "Audio file not found" in proc.stderr
