"""Catch the segments the engine got wrong, and re-roll only those.

A neural TTS engine fails occasionally and locally: one segment in a few dozen
comes back cut off mid-word, silent, or babbling well past the end of its text.
On a GPU that hardly matters — regenerate the chapter. On a CPU-only machine a
chapter is hours, so the only affordable repair is at the level of the single
bad segment, which means the bad segment has to be *identified* automatically.
Listening to four hours of narration to find eleven seconds of it is not a
workflow.

Everything here reads the waveform against the text that produced it. That
pairing is what makes the checks possible at all: audio alone cannot say whether
a two-second segment is complete, but two seconds for a sentence of two hundred
characters is a truncation, full stop.

What is detected, and why each one is worth a check:

``silent``      nothing came back — the failure that is trivially detectable and
                catastrophic if shipped.
``truncated``   far less audio than the text implies; the engine stopped early.
``runaway``     far more; the engine looped or hallucinated past the text.
``clipped``     samples pinned at full scale, which no amount of later
                mastering can undo.
``gap``         a long silence in the middle, the signature of a skipped clause.
``abrupt_end``  still at full speech level on the last sample — cut mid-word.
``looped``      the level envelope repeats, as it does when a phrase is spoken
                twice.

The same pairing catches a defect one step earlier, before anything has been
generated at all. A *cloned* voice is a recording plus the words spoken in it,
and :func:`inspect_reference` measures the one against the other. A recording
that says more than its transcript admits teaches the engine that the text runs
out before the audio does, and it then ends every narrated segment early — the
whole book truncated, from a mismatch visible in a millisecond.

Deliberately torch-free, like the rest of the package: the checks run on a
finished waveform, so they are unit-testable against synthetic signals without
loading a model.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from . import audio as audio_tools

__all__ = [
    "FATAL",
    "SUSPECT",
    "Issue",
    "QualityThresholds",
    "ReferenceReport",
    "ReferenceThresholds",
    "RenderResult",
    "SegmentReport",
    "ends_abruptly",
    "envelope_repetition",
    "inspect_reference",
    "inspect_segment",
    "longest_internal_silence_sec",
    "render_checked",
    "retry_seed",
    "summarize",
]

#: A defect bad enough that the segment should be generated again.
FATAL = "fatal"
#: Worth a human ear, not worth spending minutes of CPU on by itself.
SUSPECT = "suspect"

_SEVERITY_RANK = {SUSPECT: 1, FATAL: 2}


@dataclass(frozen=True)
class QualityThresholds:
    """Where each defect starts.

    The speech-rate limits are the load-bearing ones, and they are set from
    measurement rather than from the nominal figure. The fourteen preset voices,
    given the same 81-character sentence, come back between 15.8 and 24.1
    characters per second, median 20.2 — a spread of more than 50% between the
    slowest and the fastest voice. The bounds therefore sit well outside that
    range, so that choosing a brisk voice is never mistaken for a defect; a
    truncation that drops half of a sentence still doubles the rate and lands
    outside them.

    The range held exactly when the voice set grew from seven to fourteen, which
    is the reason to trust it: doubling the sample moved neither end.

    **English was measured too, and needs no bounds of its own.** The same four
    voices reading a sentence of the same length come back at 14.7 to 17.7
    characters per second against 17.4 to 20.9 in French — around a tenth
    slower, and the nearest limit is still more than twice away. Adding a
    language knob here would be configuration for a difference that does not
    exist, so there is none; if a language ever does fall outside, these numbers
    are what to compare its measurement against.
    """

    #: Median measured across the preset voices. Explains a report, and breaks
    #: ties between attempts; never judges one on its own.
    expected_chars_per_second: float = 20.0
    #: Above this, the text cannot have been spoken in the audio returned.
    truncated_chars_per_second: float = 35.0
    #: Below this, there is far more audio than the text can account for.
    runaway_chars_per_second: float = 6.0
    #: En deçà, le débit ne mesure plus rien. Un mot seul — « Dédicace »,
    #: « ÉPILOGUE », un titre de chapitre isolé — prend une seconde quelle que
    #: soit sa longueur, et la règle du débit le déclarait alors emballé.
    #: Mesuré : deux « défauts » sur trois d'un livre étaient des titres, et
    #: leurs réparations échouaient parce qu'il n'y avait rien à réparer.
    min_chars_for_rate: int = 25
    #: Pour un texte trop court pour que le débit signifie quelque chose, c'est
    #: la durée seule qui juge : un mot met une seconde, jamais dix. Sans cette
    #: borne, borner la règle du débit laisserait passer un vrai emballement
    #: sur un titre — « Dédicace » en vingt secondes est aussi cassé que la
    #: même chose sur un paragraphe.
    max_short_segment_sec: float = 4.0
    #: Segments shorter than this are treated as a failed generation outright.
    min_duration_sec: float = 0.2
    #: A segment whose peak sits below this carries no speech at all.
    silence_peak_db: float = -50.0
    #: Longest silence tolerated *between* the first and last word.
    max_internal_silence_sec: float = 1.5
    #: How far below the segment's own speech level counts as silence.
    silence_relative_db: float = 30.0
    #: The last few ms should have decayed at least this far below speech level.
    abrupt_end_margin_db: float = 20.0
    edge_window_ms: float = 60.0
    #: Fraction of samples at full scale that means clipping rather than a stray peak.
    clipping_sample_ratio: float = 0.0005
    clipping_threshold: float = 0.999
    #: Envelope self-similarity above which a segment looks like a repeat.
    loop_correlation: float = 0.92
    min_loop_lag_sec: float = 0.5


@dataclass(frozen=True)
class Issue:
    """One defect found in one segment."""

    code: str
    severity: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.code} ({self.severity}): {self.detail}"


@dataclass(frozen=True)
class SegmentReport:
    """What the audio measures, and what is wrong with it."""

    duration_sec: float
    characters: int
    chars_per_second: float
    rms_db: float
    peak_db: float
    longest_silence_sec: float
    issues: Tuple[Issue, ...] = ()
    #: Rate the segment was judged against, carried so :attr:`penalty` can rank
    #: attempts without needing the thresholds that produced the report.
    expected_chars_per_second: float = 20.0

    @property
    def ok(self) -> bool:
        """True when nothing at all was flagged."""
        return not self.issues

    @property
    def fatal(self) -> bool:
        """True when the segment should be generated again."""
        return any(issue.severity == FATAL for issue in self.issues)

    @property
    def severity(self) -> Optional[str]:
        if not self.issues:
            return None
        return max((issue.severity for issue in self.issues), key=lambda s: _SEVERITY_RANK.get(s, 0))

    @property
    def codes(self) -> Tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    @property
    def penalty(self) -> Tuple[int, int, float]:
        """Sort key for choosing between attempts — lower is better.

        Fatal count first, then suspect count, then distance from the expected
        speech rate. The last term only ever breaks ties between attempts that
        are equally defective, and prefers the one whose length best matches its
        text.
        """
        fatal = sum(1 for issue in self.issues if issue.severity == FATAL)
        suspect = len(self.issues) - fatal
        rate_error = (
            abs(self.chars_per_second - self.expected_chars_per_second)
            if self.chars_per_second
            else 1e6
        )
        return (fatal, suspect, rate_error)

    def describe(self) -> str:
        if self.ok:
            return f"ok ({self.duration_sec:.1f}s, {self.chars_per_second:.0f} car/s)"
        return f"{self.severity}: " + ", ".join(f"{i.code} — {i.detail}" for i in self.issues)


# --------------------------------------------------------------------------
# Individual measurements
# --------------------------------------------------------------------------


def longest_internal_silence_sec(
    wav: np.ndarray,
    sr: int,
    *,
    relative_db: float = 30.0,
    frame_ms: float = 30.0,
) -> float:
    """Longest silence *between* the first and last word, in seconds.

    Leading and trailing silence is excluded deliberately: every segment has
    some, it is trimmed later anyway, and counting it would flag every segment
    the engine padded generously.
    """
    wav = audio_tools.as_float_mono(wav)
    if wav.size == 0 or sr <= 0:
        return 0.0

    hop_ms = frame_ms / 2.0
    levels = audio_tools.frame_rms_db(wav, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    if levels.size == 0:
        return 0.0

    threshold = audio_tools.speech_rms_db(wav, sr) - relative_db
    if not np.isfinite(threshold):
        return 0.0

    loud = np.flatnonzero(levels > threshold)
    if loud.size < 2:
        return 0.0

    # Only the stretch that actually contains speech is examined.
    inner = levels[loud[0] : loud[-1] + 1] <= threshold
    if not inner.any():
        return 0.0

    longest = current = 0
    for quiet in inner:
        current = current + 1 if quiet else 0
        longest = max(longest, current)
    return longest * hop_ms / 1000.0


def ends_abruptly(
    wav: np.ndarray,
    sr: int,
    *,
    margin_db: float = 20.0,
    window_ms: float = 60.0,
) -> bool:
    """True when the segment never decays into silence before it stops.

    A finished phrase trails off; a truncated one stops while sound is still
    coming out. The margin is generous on purpose — a segment may legitimately
    end on a weak final syllable several dB down, but a properly terminated one
    ends tens of dB down, in its own noise floor.
    """
    wav = audio_tools.as_float_mono(wav)
    if wav.size == 0 or sr <= 0:
        return False

    tail = wav[-max(1, int(sr * window_ms / 1000.0)) :]
    if tail.size == 0:
        return False

    speech = audio_tools.speech_rms_db(wav, sr)
    tail_level = 20.0 * float(np.log10(max(float(np.sqrt(np.mean(np.square(tail, dtype=np.float64)))), 1e-12)))
    if not np.isfinite(speech):
        return False
    return tail_level > speech - margin_db


def envelope_repetition(
    wav: np.ndarray,
    sr: int,
    *,
    min_lag_sec: float = 0.5,
    hop_ms: float = 25.0,
    floor_db: float = 40.0,
) -> float:
    """How strongly the level envelope repeats itself, in 0..1.

    A phrase spoken twice produces a level curve that matches itself when
    shifted by the length of the phrase. Comparing the envelope with delayed
    copies of itself surfaces that without any transcription. Normal prose does
    not reach the default threshold — the rhythm of speech is not that regular.

    Two details decide whether this measures anything at all. The envelope is
    restricted to the speech itself and floored ``floor_db`` below it, because
    digital silence lands at -120 dBFS and a segment's own padding would
    otherwise dominate the curve and drown the part being compared. And each lag
    is scored as a correlation over its own overlap, not against the whole
    signal, so a repeat is not penalised for how late it occurs.
    """
    wav = audio_tools.as_float_mono(wav)
    if wav.size == 0 or sr <= 0:
        return 0.0

    levels = audio_tools.frame_rms_db(wav, sr, frame_ms=hop_ms * 2.0, hop_ms=hop_ms)
    speech_level = audio_tools.speech_rms_db(wav, sr)
    if levels.size == 0 or not np.isfinite(speech_level):
        return 0.0

    floor = speech_level - floor_db
    loud = np.flatnonzero(levels > floor)
    if loud.size == 0:
        return 0.0
    envelope = np.maximum(levels[loud[0] : loud[-1] + 1], floor).astype(np.float64)

    min_lag = int(round(min_lag_sec * 1000.0 / hop_ms))
    # Below twice the minimum lag there is no room for a repeat to show up.
    if min_lag < 1 or envelope.size < 2 * min_lag:
        return 0.0

    best = 0.0
    for lag in range(min_lag, envelope.size // 2 + 1):
        head = envelope[:-lag]
        tail = envelope[lag:]
        head = head - head.mean()
        tail = tail - tail.mean()
        denominator = float(np.sqrt(np.dot(head, head) * np.dot(tail, tail)))
        if denominator <= 0.0:
            continue
        best = max(best, float(np.dot(head, tail)) / denominator)
    return float(np.clip(best, 0.0, 1.0))


def _clipped_ratio(wav: np.ndarray, threshold: float) -> float:
    wav = audio_tools.as_float_mono(wav)
    if wav.size == 0:
        return 0.0
    return float(np.count_nonzero(np.abs(wav) >= threshold)) / float(wav.size)


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def inspect_segment(
    wav: np.ndarray,
    sr: int,
    text: str,
    thresholds: QualityThresholds = QualityThresholds(),
) -> SegmentReport:
    """Measure one generated segment against the text it was generated from."""
    wav = audio_tools.as_float_mono(wav)
    duration = float(wav.size) / sr if sr > 0 else 0.0
    characters = len((text or "").strip())
    rate = characters / duration if duration > 0 else 0.0
    peak = audio_tools.peak_db(wav)
    rms = audio_tools.speech_rms_db(wav, sr) if wav.size else -np.inf

    issues: List[Issue] = []

    # -- is there audio at all --------------------------------------------
    if wav.size == 0 or duration < thresholds.min_duration_sec:
        issues.append(Issue("silent", FATAL, f"durée {duration:.2f}s, quasi nulle"))
    elif peak < thresholds.silence_peak_db:
        issues.append(Issue("silent", FATAL, f"crête {peak:.1f} dBFS, aucun signal audible"))

    # -- does its length match its text ------------------------------------
    # Only meaningful once there is both text and audio; an empty segment has
    # already been flagged above and would divide by zero here.
    if characters and duration >= thresholds.min_duration_sec:
        if rate > thresholds.truncated_chars_per_second:
            expected = characters / thresholds.expected_chars_per_second
            issues.append(
                Issue(
                    "truncated",
                    FATAL,
                    f"{characters} caractères en {duration:.1f}s (~{expected:.1f}s attendues)",
                )
            )
        elif (
            rate < thresholds.runaway_chars_per_second
            and characters >= thresholds.min_chars_for_rate
        ) or (
            characters < thresholds.min_chars_for_rate
            and duration > thresholds.max_short_segment_sec
        ):
            expected = characters / thresholds.expected_chars_per_second
            issues.append(
                Issue(
                    "runaway",
                    FATAL,
                    f"{duration:.1f}s pour {characters} caractères (~{expected:.1f}s attendues)",
                )
            )

    # -- damage that mastering cannot repair -------------------------------
    clipped = _clipped_ratio(wav, thresholds.clipping_threshold)
    if clipped > thresholds.clipping_sample_ratio:
        issues.append(Issue("clipped", FATAL, f"{clipped * 100:.2f}% des échantillons saturés"))

    # -- shape of the delivery ---------------------------------------------
    gap = longest_internal_silence_sec(wav, sr, relative_db=thresholds.silence_relative_db)
    if gap > thresholds.max_internal_silence_sec:
        issues.append(Issue("gap", SUSPECT, f"silence interne de {gap:.1f}s"))

    if wav.size and ends_abruptly(
        wav, sr, margin_db=thresholds.abrupt_end_margin_db, window_ms=thresholds.edge_window_ms
    ):
        issues.append(Issue("abrupt_end", SUSPECT, "se termine au niveau de parole, coupé net"))

    repetition = envelope_repetition(wav, sr, min_lag_sec=thresholds.min_loop_lag_sec)
    if repetition > thresholds.loop_correlation:
        issues.append(Issue("looped", SUSPECT, f"enveloppe répétitive (corrélation {repetition:.2f})"))

    return SegmentReport(
        duration_sec=duration,
        characters=characters,
        chars_per_second=rate,
        rms_db=float(rms),
        peak_db=float(peak),
        longest_silence_sec=gap,
        issues=tuple(issues),
        expected_chars_per_second=thresholds.expected_chars_per_second,
    )


# --------------------------------------------------------------------------
# The recording a cloned voice is built from
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceThresholds:
    """Where a cloning recording stops being usable.

    The rate bounds are deliberately not :class:`QualityThresholds`': those
    judge *generated* speech against the fourteen preset voices, while this
    judges a human take, measured over the speech alone rather than the whole
    file. Measured on the three recordings available here — one voice, one
    session — the under-transcribed one comes back at 10.4 characters per second
    of speech against 15.9 and 19.0 for the two whose transcripts are exact.
    The lower bound sits between them, nearer the bad case, because the cost is
    asymmetric: this only ever prints a warning, so missing a mild mismatch is
    cheaper than crying wolf at a deliberate speaker.

    Three recordings of one speaker is a thin basis, and the honest reading of
    these numbers is "far enough outside plausible narration to be worth a
    look", not "measured to two significant figures". Widen them rather than
    argue with them if a real take is ever flagged.
    """

    #: Below this, there is more speech in the recording than the transcript accounts for.
    min_chars_per_second: float = 12.0
    #: Above this, the transcript claims words the recording does not contain.
    max_chars_per_second: float = 30.0
    #: Typical rate, used only to phrase the report in seconds.
    expected_chars_per_second: float = 17.0
    #: Below this there is too little voice to clone from.
    min_speech_sec: float = 3.0
    #: Past this the recording is only costing tokens; it clones no better.
    max_speech_sec: float = 30.0
    silence_peak_db: float = -50.0
    clipping_sample_ratio: float = 0.0005
    clipping_threshold: float = 0.999


@dataclass(frozen=True)
class ReferenceReport:
    """What a cloning recording measures, and what is wrong with it."""

    duration_sec: float
    speech_sec: float
    characters: int
    chars_per_second: float
    peak_db: float
    issues: Tuple[Issue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def fatal(self) -> bool:
        return any(issue.severity == FATAL for issue in self.issues)

    @property
    def severity(self) -> Optional[str]:
        if not self.issues:
            return None
        return max((issue.severity for issue in self.issues), key=lambda s: _SEVERITY_RANK.get(s, 0))

    @property
    def codes(self) -> Tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    def describe(self) -> str:
        if self.ok:
            return f"ok ({self.speech_sec:.1f}s de parole, {self.chars_per_second:.0f} car/s)"
        return ", ".join(f"{i.code} — {i.detail}" for i in self.issues)


def inspect_reference(
    wav: np.ndarray,
    sr: int,
    text: str,
    thresholds: ReferenceThresholds = ReferenceThresholds(),
) -> ReferenceReport:
    """Measure a cloning recording against the transcript that goes with it.

    Runs before a single segment is generated, which is the entire point: a
    mismatch here is silent — the recording sounds perfectly fine on its own —
    and only shows up as truncated narration minutes of CPU later, where the
    cause is nowhere near the symptom.

    The comparison is against *speech* seconds, not the file's length, so
    trailing silence and a speaker's pauses do not read as a mismatch. Nothing
    is refused: these bounds are heuristics on a thin sample, and the caller is
    better placed than they are to decide that an unusual take is deliberate.
    """
    wav = audio_tools.as_float_mono(wav)
    duration = float(wav.size) / sr if sr > 0 else 0.0
    speech = audio_tools.speech_seconds(wav, sr) if wav.size and sr > 0 else 0.0
    characters = len((text or "").strip())
    rate = characters / speech if speech > 0 else 0.0
    peak = audio_tools.peak_db(wav)

    issues: List[Issue] = []

    if wav.size == 0 or speech <= 0.0 or peak < thresholds.silence_peak_db:
        issues.append(Issue("silent", FATAL, f"aucune parole dans l'enregistrement ({duration:.1f}s)"))
        return ReferenceReport(duration, speech, characters, rate, float(peak), tuple(issues))

    if not characters:
        # Supported by the engine, and markedly worse: without the words, the
        # timbre is copied but the prosody is guessed.
        issues.append(
            Issue("no_transcript", SUSPECT, "enregistrement sans transcription, le clonage sera moins fidèle")
        )
    elif rate < thresholds.min_chars_per_second:
        accounted = characters / thresholds.expected_chars_per_second
        issues.append(
            Issue(
                "undertranscribed",
                FATAL,
                f"{speech:.1f}s de parole pour {characters} caractères "
                f"(~{accounted:.1f}s attendues) — la transcription ne couvre pas tout "
                f"l'enregistrement, la narration sera tronquée",
            )
        )
    elif rate > thresholds.max_chars_per_second:
        accounted = characters / thresholds.expected_chars_per_second
        issues.append(
            Issue(
                "overtranscribed",
                FATAL,
                f"{characters} caractères pour {speech:.1f}s de parole "
                f"(~{accounted:.1f}s attendues) — la transcription contient des mots "
                f"qui ne sont pas dans l'enregistrement",
            )
        )

    if speech < thresholds.min_speech_sec:
        issues.append(
            Issue("too_short", SUSPECT, f"{speech:.1f}s de parole, peu pour caractériser une voix")
        )
    elif speech > thresholds.max_speech_sec:
        issues.append(
            Issue("too_long", SUSPECT, f"{speech:.1f}s de parole, sans bénéfice pour le clonage")
        )

    clipped = _clipped_ratio(wav, thresholds.clipping_threshold)
    if clipped > thresholds.clipping_sample_ratio:
        issues.append(Issue("clipped", SUSPECT, f"{clipped * 100:.2f}% des échantillons saturés"))

    return ReferenceReport(duration, speech, characters, rate, float(peak), tuple(issues))


# --------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------


def retry_seed(base_seed: Optional[int], attempt: int, text: str = "") -> Optional[int]:
    """A different but reproducible seed for re-generating one segment.

    Derived rather than random so that a repaired book stays reproducible: the
    same book re-run from scratch repairs the same segment with the same seed
    and gets the same audio. ``None`` stays ``None`` — the engine is already
    picking its own seed, so asking again is enough.
    """
    if base_seed is None:
        return None
    digest = hashlib.sha256(f"{base_seed}:{attempt}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**32 - 1)


@dataclass
class RenderResult:
    """The audio finally kept for a segment, and how it was arrived at."""

    sample_rate: int
    wav: np.ndarray
    report: SegmentReport
    attempts: int
    seed: Optional[int]
    #: Reports of every rejected attempt, oldest first.
    rejected: List[SegmentReport] = field(default_factory=list)

    @property
    def repaired(self) -> bool:
        """True when a re-roll was needed and produced something acceptable."""
        return self.attempts > 1 and not self.report.fatal

    @property
    def unrepairable(self) -> bool:
        """True when every attempt came back defective."""
        return self.report.fatal


def render_checked(
    text: str,
    render: Callable[[Optional[int]], Tuple[int, np.ndarray]],
    *,
    base_seed: Optional[int] = None,
    max_attempts: int = 2,
    thresholds: QualityThresholds = QualityThresholds(),
    on_attempt: Optional[Callable[[int, SegmentReport], None]] = None,
) -> RenderResult:
    """Generate a segment, and re-roll it while it comes back fatally defective.

    ``render`` takes a seed and returns ``(sample_rate, audio)``; keeping the
    engine behind that callable is what lets this be tested without a model.

    Only fatal defects trigger a re-roll — a suspect one costs minutes of CPU to
    chase and is often the text's own doing. The best attempt is always
    returned, never the last: a second roll can be worse than the first, and
    silently keeping the worse one would make the repair pass harmful.
    """
    attempts = max(1, int(max_attempts))
    best: Optional[RenderResult] = None
    rejected: List[SegmentReport] = []
    last_error: Optional[Exception] = None
    calls = 0

    for attempt in range(attempts):
        seed = base_seed if attempt == 0 else retry_seed(base_seed, attempt, text)
        calls += 1
        try:
            sample_rate, wav = render(seed)
        except Exception as error:  # noqa: BLE001 - retried below, re-raised if terminal
            # A crash on one seed is itself a failure mode worth re-rolling: the
            # engine occasionally dies on a specific seed/text pair.
            last_error = error
            continue

        wav = audio_tools.as_float_mono(wav)
        report = inspect_segment(wav, sample_rate, text, thresholds)
        if on_attempt is not None:
            on_attempt(attempt, report)

        candidate = RenderResult(
            sample_rate=sample_rate, wav=wav, report=report, attempts=calls, seed=seed
        )
        if best is None or report.penalty < best.report.penalty:
            if best is not None:
                rejected.append(best.report)
            best = candidate
        else:
            rejected.append(report)

        if not report.fatal:
            break

    if best is None:
        raise last_error if last_error is not None else RuntimeError(
            "render produced nothing and raised nothing"
        )

    best.attempts = calls
    best.rejected = rejected
    return best


def summarize(reports: Sequence[Tuple[str, SegmentReport]]) -> dict:
    """Aggregate per-segment reports into something worth printing at the end."""
    flagged = [(label, report) for label, report in reports if not report.ok]
    counts: dict = {}
    for _, report in flagged:
        for issue in report.issues:
            counts[issue.code] = counts.get(issue.code, 0) + 1
    return {
        "segments": len(reports),
        "flagged": len(flagged),
        "fatal": sum(1 for _, report in flagged if report.fatal),
        "by_code": counts,
        "details": [
            {
                "segment": label,
                "severity": report.severity,
                "duration_sec": round(report.duration_sec, 2),
                "chars_per_second": round(report.chars_per_second, 1),
                "issues": [{"code": i.code, "severity": i.severity, "detail": i.detail} for i in report.issues],
            }
            for label, report in flagged
        ],
    }
