"""Tests for narration.repair — plan persistence and per-segment repair.

The engine is a callable, so a full repair cycle — inspect a book, re-roll the
bad segment, restitch its chapter — runs here in milliseconds against synthetic
audio, with no model and no torch.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from narration import cache as cache_tools
from narration import quality, repair

SR = 24000
BASE_SEED = 777


def speech(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = max(1, int(SR * seconds))
    syllables = max(2, int(seconds * 6))
    gains = rng.uniform(0.3, 1.0, syllables + 1)
    envelope = np.interp(np.linspace(0.0, syllables, samples), np.arange(syllables + 1), gains)
    body = (rng.normal(0.0, 1.0, samples) * envelope).astype(np.float32)
    body *= 0.2 / max(float(np.max(np.abs(body))), 1e-9)
    pad = np.zeros(int(SR * 0.4), dtype=np.float32)
    return np.concatenate([pad, body, pad])


def sentence(characters: int) -> str:
    return "a" * characters


#: 170 characters at the engine's ~20 char/s is about eight seconds.
TEXT_A = sentence(170)
TEXT_B = sentence(170)[:-1] + "b"
TEXT_C = sentence(120)


@pytest.fixture
def plan():
    return repair.BookPlan(
        voice={
            "description": "voix de test",
            "seed": BASE_SEED,
            "cfg": 2.0,
            "steps": 10,
            "normalize": True,
            "model_id": "test-model",
        },
        mastering={"target_rms_db": -20.0},
        chapters=(
            repair.PlannedChapter(
                index=1,
                title="Chapitre premier",
                segments=(
                    repair.PlannedSegment(TEXT_A, 0.35),
                    repair.PlannedSegment(TEXT_B, 0.7),
                ),
            ),
            repair.PlannedChapter(
                index=2,
                title="Chapitre second",
                segments=(repair.PlannedSegment(TEXT_C, 0.35),),
            ),
        ),
    )


@pytest.fixture
def cache(tmp_path):
    return cache_tools.ChunkCache(tmp_path / ".cache")


def fill_cache(plan, cache, *, bad_labels=()):
    """Populate the cache as a narration would, optionally with bad takes."""
    spec = plan.voice_spec()
    for chapter in plan.chapters:
        for position, segment in enumerate(chapter.segments, 1):
            label = repair.segment_label(chapter.index, position)
            wav = speech(0.3) if label in bad_labels else speech(len(segment.text) / 20.0)
            cache.put(cache.key(segment.text, spec), SR, wav, text=segment.text)


# --------------------------------------------------------------------------
# Plan persistence
# --------------------------------------------------------------------------


def test_plan_round_trips_through_disk(plan, tmp_path):
    plan.save(tmp_path)
    loaded = repair.BookPlan.load(tmp_path)
    assert loaded.to_dict() == plan.to_dict()
    assert loaded.segment(1, 2).text == TEXT_B
    assert loaded.segment(1, 2).pause_after == 0.7
    assert loaded.chapter(2).title == "Chapitre second"


def test_plan_reproduces_the_exact_cache_key(plan, cache):
    """The whole repair path rests on this: a plan that cannot reproduce the
    key points at no audio at all."""
    spec = plan.voice_spec()
    fill_cache(plan, cache)
    assert cache.get(cache.key(TEXT_A, spec)) is not None
    assert spec.seed == BASE_SEED and spec.model_id == "test-model"


def test_loading_a_missing_plan_explains_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="Re-run the narration"):
        repair.BookPlan.load(tmp_path)


def test_a_newer_plan_version_is_refused(tmp_path):
    (tmp_path / repair.PLAN_FILENAME).write_text(
        json.dumps({"version": repair.PLAN_VERSION + 1, "chapters": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="version"):
        repair.BookPlan.load(tmp_path)


def test_save_is_atomic_leaving_no_temporary(plan, tmp_path):
    plan.save(tmp_path)
    assert not list(tmp_path.glob("*.tmp"))


def test_unknown_chapter_or_segment_is_a_clear_error(plan):
    with pytest.raises(KeyError, match="No chapter 9"):
        plan.chapter(9)
    with pytest.raises(KeyError, match="asked for 5"):
        plan.segment(1, 5)


def test_labels_round_trip():
    assert repair.segment_label(1, 2) == "ch001/seg002"
    assert repair.parse_label("ch001/seg002") == (1, 2)
    with pytest.raises(ValueError):
        repair.parse_label("pas un label")


# --------------------------------------------------------------------------
# Inspecting a finished book
# --------------------------------------------------------------------------


def test_inspect_book_reports_every_cached_segment(plan, cache):
    fill_cache(plan, cache)
    reports = repair.inspect_book(plan, cache)
    assert [label for label, _ in reports] == ["ch001/seg001", "ch001/seg002", "ch002/seg001"]
    assert all(report.ok for _, report in reports)


def test_inspect_book_finds_the_bad_segment(plan, cache):
    fill_cache(plan, cache, bad_labels={"ch001/seg002"})
    flagged = repair.flagged_segments(repair.inspect_book(plan, cache))
    assert [label for label, _ in flagged] == ["ch001/seg002"]
    assert flagged[0][1].fatal
    assert "truncated" in flagged[0][1].codes


def test_inspect_book_skips_segments_that_are_not_cached(plan, cache):
    # Only chapter 2 was ever generated.
    spec = plan.voice_spec()
    cache.put(cache.key(TEXT_C, spec), SR, speech(6.0), text=TEXT_C)
    assert [label for label, _ in repair.inspect_book(plan, cache)] == ["ch002/seg001"]


def test_flagged_segments_puts_fatal_defects_first(plan, cache):
    fill_cache(plan, cache, bad_labels={"ch002/seg001"})
    reports = repair.inspect_book(plan, cache)
    # Force a suspect-only report onto an earlier label.
    suspect = quality.SegmentReport(
        duration_sec=5.0, characters=100, chars_per_second=20.0, rms_db=-20.0,
        peak_db=-6.0, longest_silence_sec=2.0,
        issues=(quality.Issue("gap", quality.SUSPECT, "silence"),),
    )
    mixed = [("ch001/seg001", suspect)] + reports
    assert repair.flagged_segments(mixed)[0][0] == "ch002/seg001"


def test_fatal_only_filters_out_suspects(plan, cache):
    suspect = quality.SegmentReport(
        duration_sec=5.0, characters=100, chars_per_second=20.0, rms_db=-20.0,
        peak_db=-6.0, longest_silence_sec=2.0,
        issues=(quality.Issue("gap", quality.SUSPECT, "silence"),),
    )
    assert repair.flagged_segments([("ch001/seg001", suspect)], fatal_only=True) == []


# --------------------------------------------------------------------------
# Re-rolling a segment
# --------------------------------------------------------------------------


def test_reroll_replaces_the_cached_take(plan, cache):
    fill_cache(plan, cache, bad_labels={"ch001/seg002"})
    spec = plan.voice_spec()
    key = cache.key(TEXT_B, spec)

    result = repair.reroll_segment(
        plan, 1, 2, cache, lambda seed: (SR, speech(8.5, seed=1))
    )
    assert result.report.ok
    assert result.previous is not None and result.previous.fatal
    assert result.improved

    sample_rate, stored = cache.get(key)
    assert len(stored) / sample_rate > 5.0  # the good take, not the 0.3s one


def test_reroll_uses_a_derived_reproducible_seed(plan, cache):
    fill_cache(plan, cache)
    seen = []
    repair.reroll_segment(
        plan, 1, 1, cache, lambda seed: (seen.append(seed), (SR, speech(8.5)))[1], attempt=3
    )
    assert seen == [quality.retry_seed(BASE_SEED, 3, TEXT_A)]
    assert seen[0] != BASE_SEED


def test_repairing_twice_gives_a_different_take(plan, cache):
    fill_cache(plan, cache)
    seen = []

    def render(seed):
        seen.append(seed)
        return SR, speech(8.5, seed=len(seen))

    repair.reroll_segment(plan, 1, 1, cache, render)
    repair.reroll_segment(plan, 1, 1, cache, render)
    # The attempt number is remembered, so the second repair is not the first.
    assert seen[0] != seen[1]


def test_a_worse_take_is_discarded(plan, cache):
    fill_cache(plan, cache)
    spec = plan.voice_spec()
    key = cache.key(TEXT_A, spec)
    before = cache.get(key)[1]

    result = repair.reroll_segment(plan, 1, 1, cache, lambda seed: (SR, speech(0.3)))
    assert not result.improved
    assert result.report.fatal
    # The cache still holds the take that was there.
    assert np.array_equal(cache.get(key)[1], before)


def test_keep_worse_overrides_the_guard(plan, cache):
    fill_cache(plan, cache)
    spec = plan.voice_spec()
    key = cache.key(TEXT_A, spec)

    repair.reroll_segment(
        plan, 1, 1, cache, lambda seed: (SR, speech(0.3)), keep_worse=True
    )
    sample_rate, stored = cache.get(key)
    assert len(stored) / sample_rate < 2.0


# --------------------------------------------------------------------------
# Rebuilding a chapter
# --------------------------------------------------------------------------


def test_rebuild_writes_the_chapter_from_cache_only(plan, cache, tmp_path):
    fill_cache(plan, cache)
    result = repair.rebuild_chapter(plan, 1, cache, tmp_path)
    assert result.ok
    assert result.path.name == "chapitre_001.wav"
    audio, sample_rate = sf.read(str(result.path), dtype="float32")
    # Two ~8.5s segments plus the pause and the lead/tail silence.
    assert len(audio) / sample_rate > 16.0


def test_rebuild_applies_the_plans_mastering_target(plan, cache, tmp_path):
    from narration import audio as audio_tools

    # A target below the material's natural level, so it is reachable: aiming
    # louder than the peak ceiling allows would be clamped by design, and would
    # test the ceiling rather than whether the plan's setting was read at all.
    plan.mastering = {"target_rms_db": -24.0}
    fill_cache(plan, cache)
    result = repair.rebuild_chapter(plan, 1, cache, tmp_path)
    audio, sample_rate = sf.read(str(result.path), dtype="float32")
    assert audio_tools.speech_rms_db(audio, sample_rate) == pytest.approx(-24.0, abs=0.5)


def test_rebuild_never_breaches_the_peak_ceiling(plan, cache, tmp_path):
    from narration import audio as audio_tools

    # -12 dBFS RMS is louder than this material can go without clipping, so the
    # gain must be held back to protect the ceiling rather than hit the target.
    plan.mastering = {"target_rms_db": -12.0}
    fill_cache(plan, cache)
    result = repair.rebuild_chapter(plan, 1, cache, tmp_path)
    audio, sample_rate = sf.read(str(result.path), dtype="float32")
    assert audio_tools.peak_db(audio) <= audio_tools.ACX_PEAK_CEILING_DB + 0.1
    assert audio_tools.speech_rms_db(audio, sample_rate) < -12.0


def test_rebuild_refuses_rather_than_writing_a_short_chapter(plan, cache, tmp_path):
    """A silently shortened chapter is worse than one that failed to rebuild."""
    spec = plan.voice_spec()
    cache.put(cache.key(TEXT_A, spec), SR, speech(8.5), text=TEXT_A)  # only segment 1

    result = repair.rebuild_chapter(plan, 1, cache, tmp_path)
    assert not result.ok
    assert result.missing == ("ch001/seg002",)
    assert not repair.chapter_path(tmp_path, 1).exists()


def test_repair_then_rebuild_is_a_complete_cycle(plan, cache, tmp_path):
    """The whole point: fix one segment, get a correct chapter back, and never
    re-synthesize the segments that were already fine."""
    fill_cache(plan, cache, bad_labels={"ch001/seg001"})
    assert repair.flagged_segments(repair.inspect_book(plan, cache))

    calls = []
    repair.reroll_segment(
        plan, 1, 1, cache, lambda seed: (calls.append(seed), (SR, speech(8.5, seed=9)))[1]
    )
    assert len(calls) == 1  # exactly one segment was generated

    result = repair.rebuild_chapter(plan, 1, cache, tmp_path)
    assert result.ok
    assert not repair.flagged_segments(repair.inspect_book(plan, cache))
