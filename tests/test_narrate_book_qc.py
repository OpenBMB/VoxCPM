"""End-to-end tests of the quality pass inside scripts/narrate_book.py.

The engine is replaced by a stub, so the whole narration run — planning, cache,
quality inspection, re-rolls, mastering, report — executes in milliseconds and
without importing torch. What is under test is the wiring: that a defective take
really does trigger a second call, that the good one is what lands in the
chapter, and that the report says so.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SR = 24000
BASE_SEED = 4242
#: Characters per second the stub engine reads at, inside the accepted band.
STUB_RATE = 17.0

#: Every (text, seed) the stub engine was asked for, in order.
CALLS: list[tuple[str, int | None]] = []
#: Seeds the stub should fail on. Empty means every take is clean.
FAIL_SEEDS: set[int | None] = set()


def _noise(seconds: float, level: float = 0.2) -> np.ndarray:
    """A signal that passes every check except the ones a test is targeting."""
    samples = max(1, int(SR * seconds))
    rng = np.random.default_rng(abs(hash(round(seconds, 3))) % (2**32))
    syllables = max(2, int(seconds * 6))
    gains = rng.uniform(0.3, 1.0, syllables + 1)
    envelope = np.interp(np.linspace(0.0, syllables, samples), np.arange(syllables + 1), gains)
    body = (rng.normal(0.0, 1.0, samples) * envelope).astype(np.float32)
    body *= level / max(float(np.max(np.abs(body))), 1e-9)
    tail = np.zeros(int(SR * 0.4), dtype=np.float32)
    return np.concatenate([tail, body, tail])


class StubDemo:
    """Stands in for VoxCPMDemo, returning audio whose defects are scripted."""

    def __init__(self, **_kwargs) -> None:
        pass

    def generate_tts_audio(self, *, text_input, seed=None, **_kwargs):
        CALLS.append((text_input, seed))
        characters = len((text_input or "").strip())
        if seed in FAIL_SEEDS:
            # Far too little audio for the text: a truncation, which is fatal.
            return SR, _noise(0.3), None
        return SR, _noise(max(0.5, characters / STUB_RATE)), None


app_stub = types.ModuleType("app")
app_stub.PRESET_VOICES = [
    {"name": "Voix de test", "description": "voix française de test", "seed": BASE_SEED}
]
app_stub._PRESET_BY_NAME = {"Voix de test": app_stub.PRESET_VOICES[0]}
app_stub._OUTPUT_DIR = ROOT / "output"
app_stub._sanitize_filename = lambda name: name
app_stub.VoxCPMDemo = StubDemo
sys.modules["app"] = app_stub

spec = importlib.util.spec_from_file_location("narrate_book", ROOT / "scripts" / "narrate_book.py")
narrate_book = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(narrate_book)


@pytest.fixture(autouse=True)
def reset_stub():
    CALLS.clear()
    FAIL_SEEDS.clear()
    yield
    CALLS.clear()
    FAIL_SEEDS.clear()


@pytest.fixture
def book(tmp_path):
    """A two-chapter book, one segment each."""
    path = tmp_path / "livre.txt"
    path.write_text(
        "Chapitre premier. Une phrase de longueur raisonnable pour un segment.\n"
        "---\n"
        "Chapitre second. Une autre phrase, de longueur comparable au premier.\n",
        encoding="utf-8",
    )
    return path


def run(monkeypatch, book, outdir, *extra) -> int:
    """Run the script over the fixture book, credits off.

    These tests count segments and inspect their quality; the opening and
    closing credits are two more chapters of real narration, and letting them in
    would tie every count here to their wording.
    """
    argv = [
        "narrate_book.py",
        str(book),
        "--voice",
        "Voix de test",
        "--outdir",
        str(outdir),
        "--no-credits",
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return narrate_book.main()


def read_report(outdir: Path) -> dict:
    return json.loads((outdir / "qc_report.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------


def test_clean_run_reports_every_segment_as_healthy(monkeypatch, book, tmp_path):
    outdir = tmp_path / "out"
    assert run(monkeypatch, book, outdir) == 0

    report = read_report(outdir)
    assert report["segments"] == 2
    assert report["flagged"] == 0
    assert report["fatal"] == 0
    assert len(CALLS) == 2  # one call per segment, no re-roll
    assert sorted(p.name for p in outdir.glob("chapitre_*.wav")) == [
        "chapitre_001.wav",
        "chapitre_002.wav",
    ]


def test_defective_take_is_rerolled_with_a_derived_seed(monkeypatch, book, tmp_path):
    FAIL_SEEDS.add(BASE_SEED)  # the first attempt of every segment fails
    outdir = tmp_path / "out"
    assert run(monkeypatch, book, outdir) == 0

    # Two segments, each generated twice: the failing base seed, then a derived one.
    assert len(CALLS) == 4
    seeds = [seed for _, seed in CALLS]
    assert seeds[0] == BASE_SEED and seeds[1] != BASE_SEED
    assert seeds[2] == BASE_SEED and seeds[3] != BASE_SEED

    # The repaired take is the one kept, so nothing is left flagged.
    report = read_report(outdir)
    assert report["flagged"] == 0


def test_reroll_seed_matches_the_documented_derivation(monkeypatch, book, tmp_path):
    from narration import quality

    FAIL_SEEDS.add(BASE_SEED)
    run(monkeypatch, book, tmp_path / "out")

    text, retry = CALLS[0][0], CALLS[1][1]
    assert retry == quality.retry_seed(BASE_SEED, 1, text)


def test_unrepairable_segment_is_reported_not_hidden(monkeypatch, book, tmp_path):
    # Every seed fails, so no number of re-rolls can save it.
    FAIL_SEEDS.update({BASE_SEED, None})
    monkeypatch.setattr(narrate_book.quality, "retry_seed", lambda *a, **k: None)
    outdir = tmp_path / "out"
    assert run(monkeypatch, book, outdir, "--qc-retries", "2") == 0

    report = read_report(outdir)
    assert report["flagged"] == 2
    assert report["fatal"] == 2
    assert report["by_code"]["truncated"] == 2
    assert report["details"][0]["issues"][0]["code"] == "truncated"
    # A chapter is still written — a defective book beats no book, and the
    # report says exactly which segments to listen to.
    assert (outdir / "chapitre_001.wav").is_file()


def test_qc_strict_exits_non_zero_when_a_defect_survives(monkeypatch, book, tmp_path):
    FAIL_SEEDS.update({BASE_SEED, None})
    monkeypatch.setattr(narrate_book.quality, "retry_seed", lambda *a, **k: None)
    assert run(monkeypatch, book, tmp_path / "out", "--qc-strict") == 1


def test_qc_strict_exits_zero_on_a_clean_run(monkeypatch, book, tmp_path):
    assert run(monkeypatch, book, tmp_path / "out", "--qc-strict") == 0


def test_zero_retries_reports_without_regenerating(monkeypatch, book, tmp_path):
    FAIL_SEEDS.add(BASE_SEED)
    outdir = tmp_path / "out"
    assert run(monkeypatch, book, outdir, "--qc-retries", "0") == 0

    assert len(CALLS) == 2  # inspected, never re-rolled
    assert read_report(outdir)["fatal"] == 2


def test_no_qc_skips_inspection_entirely(monkeypatch, book, tmp_path):
    FAIL_SEEDS.add(BASE_SEED)
    outdir = tmp_path / "out"
    assert run(monkeypatch, book, outdir, "--no-qc") == 0

    assert len(CALLS) == 2
    assert not (outdir / "qc_report.json").exists()


def test_repaired_audio_is_what_reaches_the_chapter(monkeypatch, book, tmp_path):
    FAIL_SEEDS.add(BASE_SEED)
    outdir = tmp_path / "out"
    run(monkeypatch, book, outdir)

    audio, sample_rate = sf.read(str(outdir / "chapitre_001.wav"), dtype="float32")
    # The rejected take was 0.3s of speech; the kept one is several seconds.
    assert len(audio) / sample_rate > 2.0


def test_cached_segments_are_still_inspected(monkeypatch, book, tmp_path):
    """A resumed run must not skip the report for what it reused.

    Segments cached by an earlier run — possibly one that predates the quality
    pass — would otherwise pass silently and never appear in the report.
    """
    outdir = tmp_path / "out"
    run(monkeypatch, book, outdir)
    assert len(CALLS) == 2

    # Second run over the same directory: chapters exist, so force them to be
    # rebuilt from the cache rather than skipped wholesale.
    CALLS.clear()
    run(monkeypatch, book, outdir, "--force")
    assert CALLS == []  # everything served from the cache
    report = read_report(outdir)
    assert report["segments"] == 2
    assert report["flagged"] == 0
