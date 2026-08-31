"""End-to-end tests of scripts/export_acx.py.

ffmpeg is not assumed to exist — on the machine this was written on it does
not. That is precisely the case worth testing: the WAVs and the encode commands
must still be produced, because the hours of synthesis behind a book cannot
depend on a binary being installed.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from narration import audio, credits, delivery  # noqa: E402

spec = importlib.util.spec_from_file_location("export_acx", ROOT / "scripts" / "export_acx.py")
export_acx = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(export_acx)

SR = 22050


def chapter_audio(seconds: float) -> np.ndarray:
    """A mastered-looking chapter: room tone, speech at level, room tone."""
    rng = np.random.default_rng(int(seconds * 100))
    samples = max(1, int(SR * seconds))
    body = rng.normal(0.0, 1.0, samples).astype(np.float32)
    body, _ = audio.normalize_level(body, SR, target_rms_db=-20.0)
    settings = audio.MasteringSettings()
    return np.concatenate(
        [audio.silence(SR, settings.lead_sec), body, audio.silence(SR, settings.tail_sec)]
    )


@pytest.fixture
def book(tmp_path):
    """A narrated book on disk: credits, two chapters, titles.txt."""
    directory = tmp_path / "book_test"
    directory.mkdir()
    titles = [credits.OPENING_TITLE, "Chapitre premier", "Chapitre second", credits.CLOSING_TITLE]
    for index, seconds in enumerate([2.0, 8.0, 8.0, 2.0], start=1):
        sf.write(str(directory / f"chapitre_{index:03d}.wav"), chapter_audio(seconds), SR,
                 subtype="PCM_16")
    (directory / "titles.txt").write_text("\n".join(titles) + "\n", encoding="utf-8")
    return directory


def run(monkeypatch, *argv) -> int:
    monkeypatch.setattr(sys, "argv", ["export_acx.py", *argv])
    return export_acx.main()


def report_of(outdir: Path) -> dict:
    return json.loads((outdir / export_acx.REPORT_NAME).read_text(encoding="utf-8"))


class TestExport:
    def test_every_chapter_becomes_a_delivered_file(self, monkeypatch, book, capsys):
        assert run(monkeypatch, str(book)) == 0
        files = report_of(book / "acx")["files"]
        # Four chapters plus the retail sample.
        assert len(files) == 5

    def test_the_report_records_the_whole_specification(self, monkeypatch, book):
        run(monkeypatch, str(book))
        entry = report_of(book / "acx")["files"][0]
        for key in ("rms_db", "peak_db", "noise_floor_db", "head_room_sec",
                    "tail_room_sec", "duration_sec", "encoded_bytes", "compliant"):
            assert key in entry

    def test_file_names_survive_an_upload_form(self, monkeypatch, book):
        """Accents in a file name are a silly way to lose an evening."""
        run(monkeypatch, str(book))
        names = [entry["file"] for entry in report_of(book / "acx")["files"]]
        assert any(name.startswith("001 - Generique de debut") for name in names)
        for name in names:
            assert name.isascii(), name

    def test_the_order_of_the_book_is_kept(self, monkeypatch, book):
        run(monkeypatch, str(book))
        names = [entry["file"] for entry in report_of(book / "acx")["files"]]
        assert names[:4] == sorted(names[:4])


class TestRetailSample:
    def test_a_sample_is_taken_from_the_book_not_the_credits(self, monkeypatch, book):
        run(monkeypatch, str(book))
        files = report_of(book / "acx")["files"]
        sample = [entry for entry in files if entry["title"] == "Extrait commercial"]
        assert len(sample) == 1

    def test_it_can_be_refused(self, monkeypatch, book):
        run(monkeypatch, str(book), "--no-sample")
        files = report_of(book / "acx")["files"]
        assert all(entry["title"] != "Extrait commercial" for entry in files)

    def test_a_chapter_can_be_chosen_explicitly(self, monkeypatch, book):
        assert run(monkeypatch, str(book), "--sample-chapter", "3") == 0

    def test_a_byte_order_mark_does_not_hide_the_credits(self, monkeypatch, book):
        """A titles.txt edited on Windows arrives with a BOM on its first line.

        Left in, the opening credits stop matching their own name — and the
        retail sample ends up being the title read out loud.
        """
        titles = (book / "titles.txt").read_text(encoding="utf-8")
        (book / "titles.txt").write_text(chr(0xFEFF) + titles, encoding="utf-8")
        run(monkeypatch, str(book))
        files = report_of(book / "acx")["files"]
        assert files[0]["title"] == credits.OPENING_TITLE


class TestWithoutFfmpeg:
    """The machine this was written on has no ffmpeg. Neither may the user's."""

    @pytest.fixture(autouse=True)
    def no_ffmpeg(self, monkeypatch):
        monkeypatch.setattr(export_acx.assembly, "find_ffmpeg", lambda: None)

    def test_the_audio_is_still_produced(self, monkeypatch, book):
        run(monkeypatch, str(book))
        assert sorted(p.name for p in (book / "acx").glob("*.wav"))

    def test_the_commands_to_finish_are_written_down(self, monkeypatch, book):
        run(monkeypatch, str(book))
        script = (book / "acx" / "encoder.txt").read_text(encoding="utf-8")
        assert "libmp3lame" in script
        assert script.count("ffmpeg") == 5

    def test_it_says_so_rather_than_failing_silently(self, monkeypatch, book, capsys):
        run(monkeypatch, str(book))
        assert "ffmpeg absent" in capsys.readouterr().out

    def test_the_exit_code_says_nothing_was_delivered(self, monkeypatch, book):
        """A directory of WAVs and an ``encoder.txt`` is what to encode, not
        what to upload. Returning zero here told the caller the book was
        delivered, and the queue marked it done — which it never replays."""
        assert run(monkeypatch, str(book)) == 1
        assert not list((book / "acx").glob("*.mp3"))


class TestCheckOnly:
    def test_check_writes_nothing(self, monkeypatch, book):
        run(monkeypatch, str(book), "--check")
        assert not (book / "acx").exists()

    def test_check_is_not_a_delivery_and_does_not_fail_for_not_being_one(
        self, monkeypatch, book
    ):
        """``--check`` reports and writes nothing, by contract: the files it
        did not encode are not files it failed to encode."""
        monkeypatch.setattr(export_acx.assembly, "find_ffmpeg", lambda: None)
        assert run(monkeypatch, str(book), "--check") == 0

    def test_check_still_reports_every_file(self, monkeypatch, book, capsys):
        run(monkeypatch, str(book), "--check")
        out = capsys.readouterr().out
        assert "Generique de debut" in out
        assert "satisfont la norme" in out or "hors norme" in out

    def test_a_defective_chapter_is_reported_and_exits_non_zero(self, monkeypatch, book, capsys):
        """A chapter with no room tone passes on level and fails on shape."""
        raw = np.concatenate([chapter_audio(3.0)[int(SR * 0.75):]])
        sf.write(str(book / "chapitre_002.wav"), raw, SR, subtype="PCM_16")
        assert run(monkeypatch, str(book), "--check") == 1
        assert "silence de tête" in capsys.readouterr().out


class TestSplitting:
    def test_an_over_long_chapter_becomes_several_files(self, monkeypatch, book):
        """The limit is the distributor's; here it is forced down to test it."""
        monkeypatch.setattr(delivery, "max_seconds_for", lambda profile=None: 4.0)
        run(monkeypatch, str(book))
        titles = [entry["title"] for entry in report_of(book / "acx")["files"]]
        assert any("partie 1" in title for title in titles)
        assert any("partie 2" in title for title in titles)


class TestSampleChapterChoice:
    """L'extrait commercial doit venir de l'introduction.

    Auparavant la règle était « le premier chapitre qui n'est pas un générique »,
    et sur un livre réel elle a choisi la page de titre : l'acheteur entendait le
    nom du livre récité et n'apprenait rien. Ce qui décide quelqu'un, c'est le
    propos du livre, et c'est exactement ce que contient une introduction.
    """

    def test_the_introduction_wins_over_the_title_page(self):
        titres = [
            "Générique de début",
            "Rebâtir l'Intimité Après Divorce",
            "Introduction — Le mur invisible",
            "Chapitre 1 — Les Ruines Invisibles",
            "Générique de fin",
        ]
        assert export_acx.sample_chapter_index(titres) == 3

    def test_front_matter_is_skipped_when_there_is_no_introduction(self):
        titres = ["Générique de début", "Dédicace", "Avertissement médical",
                  "Chapitre 1 — Le début", "Générique de fin"]
        assert export_acx.sample_chapter_index(titres) == 4

    @pytest.mark.parametrize("ouverture", ["Avant-propos", "Préface", "Prologue", "Préambule"])
    def test_every_kind_of_opening_matter_counts(self, ouverture):
        assert export_acx.sample_chapter_index(["Générique de début", ouverture, "Chapitre 1"]) == 2

    def test_a_book_with_nothing_but_chapters_takes_the_first(self):
        assert export_acx.sample_chapter_index(["Générique de début", "Chapitre 1", "Générique de fin"]) == 2

    def test_a_book_of_nothing_but_credits_has_no_sample(self):
        assert export_acx.sample_chapter_index(["Générique de début", "Générique de fin"]) is None
