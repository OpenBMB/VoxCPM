"""Tests for narrating a whole book in a cloned voice.

The engine already cloned from a reference recording in the Studio tab; what it
could not do was narrate a book that way, because the script had no way to pass
one. The delicate part is not the plumbing but the cache: a segment spoken by a
cloned voice and the same segment spoken from a description must never share an
address.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from narration.cache import VoiceSpec  # noqa: E402

from test_narrate_book_qc import CALLS, SR, _noise, narrate_book  # noqa: E402,F401
from test_narrate_book_qc import book, reset_stub  # noqa: E402,F401  (fixtures)


@pytest.fixture
def reference(tmp_path):
    """A short recording standing in for the user's own voice."""
    path = tmp_path / "ma_voix.wav"
    sf.write(str(path), _noise(3.0), SR, subtype="PCM_16")
    return path


def run(monkeypatch, book, outdir, *extra) -> int:
    monkeypatch.setattr(
        sys, "argv",
        ["narrate_book.py", str(book), "--outdir", str(outdir), "--no-credits", *extra],
    )
    return narrate_book.main()


class TestHashingTheReference:
    """The cache key has to follow the audio, not the file name."""

    def test_the_same_take_moved_keeps_its_hash(self, tmp_path, reference):
        moved = tmp_path / "ailleurs.wav"
        moved.write_bytes(reference.read_bytes())
        assert VoiceSpec.hash_reference(reference) == VoiceSpec.hash_reference(moved)

    def test_a_new_take_at_the_same_path_changes_it(self, tmp_path, reference):
        before = VoiceSpec.hash_reference(reference)
        sf.write(str(reference), _noise(2.0, level=0.3), SR, subtype="PCM_16")
        assert VoiceSpec.hash_reference(reference) != before

    def test_no_reference_hashes_to_nothing(self, tmp_path):
        assert VoiceSpec.hash_reference(None) == ""
        assert VoiceSpec.hash_reference("") == ""
        assert VoiceSpec.hash_reference(tmp_path / "absent.wav") == ""

    def test_a_cloned_voice_never_collides_with_a_described_one(self, reference):
        described = VoiceSpec(description="voix grave", seed=42)
        cloned = VoiceSpec(
            description="voix grave", seed=42, reference=VoiceSpec.hash_reference(reference)
        )
        assert described.fingerprint() != cloned.fingerprint()

    def test_the_transcript_counts_too(self, reference):
        digest = VoiceSpec.hash_reference(reference)
        without = VoiceSpec(reference=digest)
        with_text = VoiceSpec(reference=digest, reference_text="Le vent se lève.")
        assert without.fingerprint() != with_text.fingerprint()


class TestNarratingWithIt:
    def test_the_reference_reaches_the_engine(self, monkeypatch, book, tmp_path, reference):
        seen = {}

        class CloningDemo:
            def __init__(self, **_kwargs):
                pass

            def generate_tts_audio(self, *, text_input, seed=None, **kwargs):
                seen["reference"] = kwargs.get("reference_wav_path_input")
                seen["prompt"] = kwargs.get("prompt_text")
                seen["denoise"] = kwargs.get("denoise")
                CALLS.append((text_input, seed))
                return SR, _noise(max(0.5, len(text_input) / 17.0)), None

        monkeypatch.setattr(narrate_book.app, "VoxCPMDemo", CloningDemo)
        assert run(
            monkeypatch, book, tmp_path / "out",
            "--reference-audio", str(reference),
            "--reference-text", "Le vent se lève.",
        ) == 0

        assert seen["reference"] == str(reference)
        assert seen["prompt"] == "Le vent se lève."
        # The denoiser is not loaded during narration, so asking for it would
        # be asking for something that is not there.
        assert seen["denoise"] is False

    def test_it_needs_no_voice_and_no_description(self, monkeypatch, book, tmp_path, reference):
        """A recording is a complete answer to "which voice?"."""
        assert run(monkeypatch, book, tmp_path / "out", "--reference-audio", str(reference)) == 0

    def test_a_missing_recording_is_refused_before_anything_runs(
        self, monkeypatch, book, tmp_path
    ):
        with pytest.raises(SystemExit) as raised:
            run(monkeypatch, book, tmp_path / "out", "--reference-audio", str(tmp_path / "nope.wav"))
        assert "not found" in str(raised.value)

    def test_naming_no_voice_at_all_is_still_refused(self, monkeypatch, book, tmp_path):
        with pytest.raises(SystemExit) as raised:
            monkeypatch.setattr(
                sys, "argv",
                ["narrate_book.py", str(book), "--outdir", str(tmp_path / "out")],
            )
            narrate_book.main()
        assert "reference-audio" in str(raised.value)

    def test_the_plan_says_the_voice_was_cloned(self, monkeypatch, book, tmp_path, reference, capsys):
        run(monkeypatch, book, tmp_path / "out", "--reference-audio", str(reference))
        out = capsys.readouterr().out
        assert "clonée" in out
        assert reference.name in out


class TestCheckingTheReferenceBeforeAnythingRuns:
    """The recording is measured against its transcript in the pre-flight block.

    It runs during ``--dry-run``, which is the point: the defect it catches
    otherwise shows up as truncated narration, minutes of CPU from its cause.
    """

    def dry_run(self, monkeypatch, book, tmp_path, capsys, *extra) -> str:
        run(monkeypatch, book, tmp_path / "out", "--dry-run", *extra)
        return capsys.readouterr().out

    def test_a_matching_recording_is_reported_sane(
        self, monkeypatch, book, tmp_path, reference, capsys
    ):
        # 3s of voice for 51 characters — 17 char/s.
        out = self.dry_run(
            monkeypatch, book, tmp_path, capsys,
            "--reference-audio", str(reference),
            "--reference-text", "a" * 51,
        )
        assert "référence saine" in out

    def test_a_transcript_that_stops_short_is_called_out(
        self, monkeypatch, book, tmp_path, reference, capsys
    ):
        out = self.dry_run(
            monkeypatch, book, tmp_path, capsys,
            "--reference-audio", str(reference),
            "--reference-text", "a" * 10,
        )
        assert "tronquée" in out

    def test_a_recording_with_no_transcript_is_called_out(
        self, monkeypatch, book, tmp_path, reference, capsys
    ):
        out = self.dry_run(monkeypatch, book, tmp_path, capsys, "--reference-audio", str(reference))
        assert "sans transcription" in out

    def test_an_unreadable_recording_does_not_stop_the_pre_flight(
        self, monkeypatch, book, tmp_path, capsys
    ):
        """It is the engine's job to fail on it; the summary still prints."""
        broken = tmp_path / "cassé.wav"
        broken.write_bytes(b"not a wav at all")
        out = self.dry_run(monkeypatch, book, tmp_path, capsys, "--reference-audio", str(broken))
        assert "illisible" in out
        assert "Chapitres" in out
