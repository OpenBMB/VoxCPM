"""End-to-end tests of scripts/repair_segment.py.

Same stub engine as the quality tests, so a narration and a repair both run in
milliseconds. What matters here is the promise the script makes: a book that
cost nine hours can have one sentence fixed without re-synthesizing anything
else — and that promise only holds if narrate_book leaves a plan behind.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from narration import repair  # noqa: E402

from test_narrate_book_qc import CALLS, SR, _noise, narrate_book  # noqa: E402,F401
from test_narrate_book_qc import book, reset_stub  # noqa: E402,F401  (fixtures)

spec = importlib.util.spec_from_file_location(
    "repair_segment", ROOT / "scripts" / "repair_segment.py"
)
repair_segment = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(repair_segment)


def narrate(monkeypatch, book, outdir, *extra) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        ["narrate_book.py", str(book), "--voice", "Voix de test", "--outdir", str(outdir),
         "--no-credits", *extra],
    )
    return narrate_book.main()


def repair_run(monkeypatch, outdir, *extra) -> int:
    monkeypatch.setattr(sys, "argv", ["repair_segment.py", str(outdir), *extra])
    return repair_segment.main()


@pytest.fixture
def narrated(monkeypatch, book, tmp_path):
    """A narrated book, with its plan and its cache."""
    outdir = tmp_path / "out"
    assert narrate(monkeypatch, book, outdir) == 0
    CALLS.clear()
    return outdir


class TestThePlanIsWritten:
    """Without it, a book narrated by the script can never be repaired."""

    def test_narration_leaves_a_plan_behind(self, narrated):
        assert (narrated / repair.PLAN_FILENAME).is_file()

    def test_the_plan_names_every_segment(self, narrated):
        plan = repair.BookPlan.load(narrated)
        assert [chapter.index for chapter in plan.chapters] == [1, 2]
        assert all(chapter.segments for chapter in plan.chapters)

    def test_the_plan_reproduces_the_cache_keys(self, narrated):
        """A plan that cannot find the audio it describes is worthless."""
        from narration import cache as cache_tools

        plan = repair.BookPlan.load(narrated)
        cache = cache_tools.ChunkCache(narrated / ".cache")
        spec = plan.voice_spec()
        for chapter in plan.chapters:
            for segment in chapter.segments:
                assert cache.get(cache.key(segment.text, spec)) is not None

    def test_it_is_written_before_the_audio(self, monkeypatch, book, tmp_path):
        """An interrupted nine-hour narration must still be repairable."""
        outdir = tmp_path / "interrupted"

        written = {}

        def stop_after_the_plan(path, *args, **kwargs):
            written["plan"] = (outdir / repair.PLAN_FILENAME).is_file()
            raise KeyboardInterrupt

        monkeypatch.setattr(narrate_book.sf, "write", stop_after_the_plan)
        with pytest.raises(KeyboardInterrupt):
            narrate(monkeypatch, book, outdir)
        assert written["plan"] is True


class TestListing:
    def test_a_healthy_book_reports_no_defect(self, monkeypatch, narrated, capsys):
        assert repair_run(monkeypatch, narrated, "--list") == 0
        assert "aucun défaut détecté" in capsys.readouterr().out

    def test_listing_never_loads_the_model(self, monkeypatch, narrated):
        """Reading the cache must not cost a minute of model load."""
        monkeypatch.setattr(
            repair_segment.app, "VoxCPMDemo", lambda **_: pytest.fail("model loaded")
        )
        assert repair_run(monkeypatch, narrated, "--list") == 0

    def test_a_missing_plan_says_how_to_get_one(self, monkeypatch, tmp_path):
        empty = tmp_path / "vide"
        empty.mkdir()
        with pytest.raises(SystemExit) as raised:
            repair_run(monkeypatch, empty, "--list")
        assert "narrate_book.py" in str(raised.value)


class TestRepairing:
    def test_one_segment_is_regenerated_and_its_chapter_restitched(
        self, monkeypatch, narrated, capsys
    ):
        before = (narrated / "chapitre_001.wav").read_bytes()
        assert repair_run(monkeypatch, narrated, "--segment", "ch001/seg001") == 0

        out = capsys.readouterr().out
        assert "reconstruit" in out
        assert (narrated / "chapitre_001.wav").read_bytes() != before

    def test_only_that_segment_is_synthesized(self, monkeypatch, narrated):
        repair_run(monkeypatch, narrated, "--segment", "ch001/seg001")
        assert len(CALLS) == 1

    def test_the_other_chapter_is_left_alone(self, monkeypatch, narrated):
        untouched = (narrated / "chapitre_002.wav").read_bytes()
        repair_run(monkeypatch, narrated, "--segment", "ch001/seg001")
        assert (narrated / "chapitre_002.wav").read_bytes() == untouched

    def test_a_worse_take_is_refused_and_said_so(self, monkeypatch, narrated, capsys):
        """A re-roll can come back worse; the old take then stays.

        The engine is made to return a truncated take for every seed, which is
        what a bad re-roll looks like from here.
        """
        class TruncatingDemo:
            def __init__(self, **_kwargs):
                pass

            def generate_tts_audio(self, *, text_input, seed=None, **_kwargs):
                return SR, _noise(0.3), None

        monkeypatch.setattr(repair_segment.app, "VoxCPMDemo", TruncatingDemo)
        assert repair_run(monkeypatch, narrated, "--segment", "ch001/seg001") == 1
        assert "moins bon" in capsys.readouterr().out

    def test_an_unknown_label_is_reported_not_crashed(self, monkeypatch, narrated, capsys):
        assert repair_run(monkeypatch, narrated, "--segment", "ch009/seg001") == 1
        assert "ch009/seg001" in capsys.readouterr().out

    def test_nothing_happens_without_a_target(self, monkeypatch, narrated, capsys):
        assert repair_run(monkeypatch, narrated) == 0
        assert "Rien à faire" in capsys.readouterr().out
