"""End-to-end tests that a narrated book carries its credits.

Reuses the stub engine from the quality tests, so a whole run — planning,
synthesis, mastering, chapter files, titles — happens in milliseconds without
importing torch. What is under test is that the credits are really narrated as
the first and last chapters, in the same voice as the book, and that the ACX
shape of every delivered file holds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from narration import audio, credits

from test_narrate_book_qc import CALLS, SR, narrate_book  # noqa: F401
from test_narrate_book_qc import book, reset_stub  # noqa: F401  (fixtures)


def run(monkeypatch, book, outdir, *extra) -> int:
    """Run the script over the fixture book with credits left at their default.

    Not the runner from the quality tests: that one switches credits off so its
    segment counts stay stable, which is exactly what these tests need on.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        ["narrate_book.py", str(book), "--voice", "Voix de test", "--outdir", str(outdir), *extra],
    )
    return narrate_book.main()


def titles_of(outdir: Path) -> list[str]:
    return (outdir / "titles.txt").read_text(encoding="utf-8").strip().split("\n")


def chapters_of(outdir: Path) -> list[Path]:
    return sorted(outdir.glob("chapitre_*.wav"))


class TestCreditsArePresent:
    def test_the_book_is_wrapped_in_its_credits(self, monkeypatch, book, tmp_path):
        outdir = tmp_path / "out"
        assert run(monkeypatch, book, outdir, "--title", "Le Livre", "--author", "Une Autrice") == 0

        titles = titles_of(outdir)
        assert titles[0] == credits.OPENING_TITLE
        assert titles[-1] == credits.CLOSING_TITLE
        # Two chapters of book, plus the two credits.
        assert len(chapters_of(outdir)) == 4

    def test_the_credits_name_the_work(self, monkeypatch, book, tmp_path):
        run(monkeypatch, book, tmp_path / "out", "--title", "Le Livre", "--author", "Une Autrice")
        spoken = " ".join(text for text, _ in CALLS)
        assert "Le Livre" in spoken
        assert "Une Autrice" in spoken
        assert "Vous venez d'écouter" in spoken

    def test_they_are_read_in_the_same_voice_as_the_book(self, monkeypatch, book, tmp_path):
        """Same seed everywhere is what keeps one narrator across the file."""
        run(monkeypatch, book, tmp_path / "out", "--title", "Le Livre")
        seeds = {seed for _, seed in CALLS}
        assert len(seeds) == 1

    def test_a_named_narrator_is_credited(self, monkeypatch, book, tmp_path):
        run(monkeypatch, book, tmp_path / "out", "--title", "Le Livre", "--narrator", "Edwin")
        spoken = " ".join(text for text, _ in CALLS)
        assert "Edwin" in spoken
        assert credits.SYNTHETIC_DISCLOSURE not in spoken

    def test_an_unnamed_narrator_is_disclosed_as_synthetic(self, monkeypatch, book, tmp_path):
        run(monkeypatch, book, tmp_path / "out", "--title", "Le Livre")
        spoken = " ".join(text for text, _ in CALLS)
        assert credits.SYNTHETIC_DISCLOSURE in spoken

    def test_public_domain_is_stated_when_asked(self, monkeypatch, book, tmp_path):
        run(monkeypatch, book, tmp_path / "out", "--title", "Le Livre", "--public-domain")
        assert "domaine public" in " ".join(text for text, _ in CALLS)


class TestCreditsCanBeRefused:
    def test_no_credits_leaves_the_book_alone(self, monkeypatch, book, tmp_path):
        outdir = tmp_path / "out"
        assert run(monkeypatch, book, outdir, "--no-credits") == 0
        assert len(chapters_of(outdir)) == 2
        spoken = " ".join(text for text, _ in CALLS)
        assert "Vous venez d'écouter" not in spoken


class TestChainedExport:
    def test_export_acx_produces_the_delivery_folder(self, monkeypatch, book, tmp_path):
        """One command from text to the folder that gets uploaded."""
        outdir = tmp_path / "out"
        run(monkeypatch, book, outdir, "--title", "Le Livre", "--export-acx")

        acx = outdir / "acx"
        assert acx.is_dir()
        assert (acx / "rapport_acx.json").is_file()

    def test_a_failed_export_does_not_lose_the_chapters(self, monkeypatch, book, tmp_path):
        """Nine hours of narration must survive anything the exporter does.

        Surviving is not succeeding, though. The exit code used to be zero
        here, which told the queue the book was done — and done is never
        replayed. The chapters stay; the code now says the delivery is not
        there.
        """
        outdir = tmp_path / "out"
        monkeypatch.setattr(
            narrate_book.subprocess,
            "run",
            lambda *a, **k: type("Result", (), {"returncode": 1})(),
        )
        assert run(monkeypatch, book, outdir, "--title", "Le Livre", "--export-acx") == 1
        assert sorted(outdir.glob("chapitre_*.wav"))


class TestDeliveredShape:
    """Every written file has to satisfy ACX on shape, not only on level."""

    def test_every_chapter_carries_its_room_tone(self, monkeypatch, book, tmp_path):
        outdir = tmp_path / "out"
        run(monkeypatch, book, outdir, "--title", "Le Livre", "--author", "Une Autrice")

        for path in chapters_of(outdir):
            data, sample_rate = sf.read(str(path), dtype="float32")
            report = audio.acx_report(data, sample_rate)
            assert report["head_room_ok"], (path.name, report["head_room_sec"])
            assert report["tail_room_ok"], (path.name, report["tail_room_sec"])

    def test_the_credits_are_mastered_like_the_book(self, monkeypatch, book, tmp_path):
        outdir = tmp_path / "out"
        run(monkeypatch, book, outdir, "--title", "Le Livre", "--author", "Une Autrice")

        levels = []
        for path in chapters_of(outdir):
            data, sample_rate = sf.read(str(path), dtype="float32")
            levels.append(audio.speech_rms_db(data, sample_rate))
        # No file stands out: the credits went through the same loudness pass.
        assert max(levels) - min(levels) < 1.0
