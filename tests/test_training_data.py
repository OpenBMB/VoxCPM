"""Tests for fine-tuning dataset metadata preparation."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]

argbind_module = types.ModuleType("argbind")
argbind_module.bind = lambda *args, **kwargs: lambda function: function
sys.modules.setdefault("argbind", argbind_module)

pkg = types.ModuleType("voxcpm")
pkg.__path__ = [str(ROOT / "src" / "voxcpm")]
sys.modules.setdefault("voxcpm", pkg)

training_pkg = types.ModuleType("voxcpm.training")
training_pkg.__path__ = [str(ROOT / "src" / "voxcpm" / "training")]
sys.modules.setdefault("voxcpm.training", training_pkg)

model_module = types.ModuleType("voxcpm.model.voxcpm")
model_module.VoxCPMConfig = object
sys.modules.setdefault("voxcpm.model.voxcpm", model_module)

audiovae_module = types.ModuleType("voxcpm.modules.audiovae")
audiovae_module.AudioVAE = object
sys.modules.setdefault("voxcpm.modules.audiovae", audiovae_module)

packers_module = types.ModuleType("voxcpm.training.packers")
packers_module.AudioFeatureProcessingPacker = object
sys.modules.setdefault("voxcpm.training.packers", packers_module)

from voxcpm.training.data import compute_sample_lengths, load_audio_text_datasets


def _write_wav(path: Path, duration: float, sample_rate: int = 16_000) -> None:
    sf.write(path, np.zeros(int(duration * sample_rate), dtype=np.float32), sample_rate)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_load_mixed_duration_manifest_and_fill_missing_metadata(tmp_path):
    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    _write_wav(first_audio, 0.5)
    _write_wav(second_audio, 0.75)

    manifest = tmp_path / "train.jsonl"
    _write_manifest(
        manifest,
        [
            {"audio": str(first_audio), "text": "first", "duration": 0.5, "speaker": "a"},
            {"audio": str(second_audio), "text": "second", "speaker": "b"},
        ],
    )

    train_ds, _ = load_audio_text_datasets(
        train_manifest=str(manifest),
        prepare_durations=True,
        num_proc=2,
    )

    assert train_ds["duration"] == pytest.approx([0.5, 0.75])
    assert train_ds["dataset_id"] == [0, 0]
    assert train_ds["speaker"] == ["a", "b"]
    assert "ref_audio" not in train_ds.column_names


def test_existing_duration_does_not_read_audio_file(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(
        manifest,
        [{"audio": str(tmp_path / "missing.wav"), "text": "text", "duration": 1.25}],
    )

    train_ds, _ = load_audio_text_datasets(
        train_manifest=str(manifest),
        prepare_durations=True,
    )

    assert train_ds["duration"] == [1.25]


def test_reference_duration_is_filled_and_used_for_length(tmp_path):
    audio = tmp_path / "audio.wav"
    ref_audio = tmp_path / "ref.wav"
    _write_wav(audio, 1.0)
    _write_wav(ref_audio, 0.5)

    manifest = tmp_path / "train.jsonl"
    _write_manifest(
        manifest,
        [{"audio": str(audio), "ref_audio": str(ref_audio), "text": "text"}],
    )

    train_ds, _ = load_audio_text_datasets(
        train_manifest=str(manifest),
        prepare_durations=True,
    )
    train_ds = train_ds.add_column("text_ids", [[1, 2, 3]])

    lengths = compute_sample_lengths(train_ds, audio_vae_fps=25, patch_size=1)

    assert train_ds["ref_duration"] == pytest.approx([0.5])
    assert lengths == [3 + 25 + 13 + 4]


def test_compute_lengths_rejects_missing_duration():
    ds = Dataset.from_dict(
        {
            "audio": ["unused.wav"],
            "text_ids": [[1, 2]],
            "duration": [None],
        }
    )

    with pytest.raises(ValueError, match="without duration metadata"):
        compute_sample_lengths(ds)


def test_relative_audio_paths_are_resolved_from_manifest_directory(tmp_path, monkeypatch):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = audio_dir / "sample.wav"
    _write_wav(audio, 0.25)

    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, [{"audio": "audio/sample.wav", "text": "text"}])
    monkeypatch.chdir(tmp_path.parent)

    train_ds, _ = load_audio_text_datasets(
        train_manifest=str(manifest),
        prepare_durations=True,
    )

    assert train_ds["duration"] == pytest.approx([0.25])
    assert train_ds.data.column("audio")[0].as_py()["path"] == str(audio)


def test_rejects_non_positive_worker_count(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, [{"audio": "unused.wav", "text": "text"}])

    with pytest.raises(ValueError, match="num_proc must be at least 1"):
        load_audio_text_datasets(str(manifest), num_proc=0)


def test_empty_first_list_column_does_not_coerce_later_rows(tmp_path):
    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    _write_wav(first_audio, 0.5)
    _write_wav(second_audio, 0.5)

    # The first row's token id list is empty, so its element type is indeterminate.
    # Inference must not fix the column as Sequence(string) and coerce the integer
    # ids in later rows (which then crash collation via torch.tensor(..., int32)).
    manifest = tmp_path / "train.jsonl"
    _write_manifest(
        manifest,
        [
            {"audio": str(first_audio), "text": "first", "token_ids": []},
            {"audio": str(second_audio), "text": "second", "token_ids": [7, 8, 9]},
        ],
    )

    train_ds, _ = load_audio_text_datasets(
        train_manifest=str(manifest),
        prepare_durations=True,
    )

    assert train_ds["token_ids"] == [[], [7, 8, 9]]
    assert all(isinstance(v, int) for v in train_ds["token_ids"][1])
