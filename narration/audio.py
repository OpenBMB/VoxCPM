"""Master and stitch generated speech into audiobook-grade audio.

Raw TTS output is not publishable as-is. Each generated segment carries a little
silence at its edges, its level drifts from one segment to the next, and butting
segments together end-to-end produces audible clicks and a robotic, pause-less
read. This module fixes all three, and measures the result against the levels
audiobook distributors actually check.

The reference target is the ACX specification, which every major audiobook
platform mirrors: RMS between -23 and -18 dBFS, peak no higher than -3 dBFS, and
a noise floor below -60 dBFS. The same specification also governs the *shape* of
a file — 0.5 to 1 second of room tone before the first word, 1 to 5 after the
last, and no file longer than 120 minutes — so :func:`acx_report` measures those
too. A chapter that passes on level and fails on room tone is rejected on upload
just the same.

Pure ``numpy`` on purpose — no resampling library, no loudness package. Frame
energies are computed from a cumulative sum rather than a sliding window so that
a one-hour chapter costs O(n) memory instead of tens of gigabytes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ACX_HEAD_ROOM_MAX_SEC",
    "ACX_HEAD_ROOM_MIN_SEC",
    "ACX_MAX_FILE_SEC",
    "ACX_PEAK_CEILING_DB",
    "ACX_RMS_MAX_DB",
    "ACX_RMS_MIN_DB",
    "ACX_NOISE_FLOOR_DB",
    "ACX_TAIL_ROOM_MAX_SEC",
    "ACX_TAIL_ROOM_MIN_SEC",
    "MasteringSettings",
    "acx_report",
    "as_float_mono",
    "fade_edges",
    "frame_rms_db",
    "master_segment",
    "noise_floor_db",
    "normalize_level",
    "peak_db",
    "remove_dc",
    "silence",
    "speech_bounds",
    "speech_rms_db",
    "speech_seconds",
    "stitch",
    "trim_silence",
]

#: ACX / audiobook distribution limits, in dBFS.
ACX_RMS_MIN_DB = -23.0
ACX_RMS_MAX_DB = -18.0
ACX_PEAK_CEILING_DB = -3.0
ACX_NOISE_FLOOR_DB = -60.0

#: Room tone ACX expects around the speech of every delivered file, in seconds.
#: Not decoration: a file starting on the first syllable, or ending on it, is
#: rejected at quality review however clean its levels are.
ACX_HEAD_ROOM_MIN_SEC = 0.5
ACX_HEAD_ROOM_MAX_SEC = 1.0
ACX_TAIL_ROOM_MIN_SEC = 1.0
ACX_TAIL_ROOM_MAX_SEC = 5.0
#: No single delivered file may run longer than two hours.
ACX_MAX_FILE_SEC = 120 * 60

_EPS = 1e-12
#: Below this peak level a signal carries no usable level to correct.
_SILENCE_FLOOR_DB = -120.0
#: Below this, a frame is silence for any purpose. Mirrors the BS.1770 absolute gate.
_ABSOLUTE_GATE_DB = -70.0
#: Frames quieter than the ungated mean by this much do not count as speech.
_RELATIVE_GATE_DB = 10.0


@dataclass(frozen=True)
class MasteringSettings:
    """How a segment and a finished chapter should be treated.

    Defaults aim at the middle of the ACX window (-20 dBFS RMS), which leaves
    room on both sides for the level drift between separately generated segments.
    """

    target_rms_db: float = -20.0
    peak_ceiling_db: float = ACX_PEAK_CEILING_DB
    trim_silence: bool = True
    #: A segment edge is considered silent this far below its own speech level.
    trim_relative_db: float = 25.0
    #: Silence deliberately kept at each edge, so words never start abruptly.
    trim_keep_ms: float = 60.0
    #: Click-free ramp applied to every segment edge.
    fade_ms: float = 8.0
    #: Room tone before the first word and after the last one, in seconds.
    #: Sits inside the ACX windows (0.5–1 s and 1–5 s) rather than at their
    #: edges, so a chapter stays compliant even if trimming leaves a little
    #: silence of its own.
    lead_sec: float = 0.75
    tail_sec: float = 2.0
    #: Run the studio chain — high-pass, de-esser, compressor — over the
    #: assembled chapter before its level is set. A plain bool rather than the
    #: settings themselves, so a saved plan stays a flat JSON object; the
    #: amounts are passed to :func:`stitch` separately when they need changing.
    polish: bool = True


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def as_float_mono(wav: np.ndarray) -> np.ndarray:
    """Return the signal as 1-D float32, averaging channels.

    Conversion only — no level or offset is touched here, so that a measurement
    reports what is actually in the file.
    """
    data = np.asarray(wav)
    if data.ndim > 1:
        # soundfile hands back (frames, channels); anything else is already flat.
        data = data.mean(axis=1) if data.shape[0] >= data.shape[-1] else data.mean(axis=0)
    return data.astype(np.float32, copy=False)


def remove_dc(wav: np.ndarray) -> np.ndarray:
    """Centre the waveform on zero.

    A constant offset eats headroom and makes every join click, but it is not
    audible in itself, so it is easy to ship by accident.
    """
    data = as_float_mono(wav)
    if data.size == 0:
        return data
    return data - np.float32(data.mean())


def _to_db(amplitude: float) -> float:
    return 20.0 * float(np.log10(max(float(amplitude), _EPS)))


def _frame_power(wav: np.ndarray, sr: int, frame_ms: float, hop_ms: float) -> np.ndarray:
    """Mean square of every frame, computed from a cumulative sum.

    A sliding-window view would allocate frame_length x n_frames floats — over
    100 GB for a long chapter — so the energies come from prefix sums instead.
    """
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if wav.size < frame:
        return np.array([float(np.mean(np.square(wav, dtype=np.float64)))]) if wav.size else np.zeros(0)

    cumulative = np.concatenate(([0.0], np.cumsum(np.square(wav, dtype=np.float64))))
    starts = np.arange(0, wav.size - frame + 1, hop)
    return (cumulative[starts + frame] - cumulative[starts]) / frame


def speech_rms_db(wav: np.ndarray, sr: int, frame_ms: float = 400.0, hop_ms: float = 100.0) -> float:
    """RMS level in dBFS, measured over speech only.

    Silence between sentences must not count: a chapter with generous pauses
    would otherwise measure several dB quieter than it sounds, and normalising
    against that figure would push the actual speech above the peak ceiling.
    """
    wav = as_float_mono(wav)
    if wav.size == 0:
        return -np.inf

    power = _frame_power(wav, sr, frame_ms, hop_ms)
    if power.size == 0:
        return -np.inf

    absolute_gate = 10.0 ** (_ABSOLUTE_GATE_DB / 10.0)
    kept = power[power > absolute_gate]
    if kept.size == 0:
        return _to_db(float(np.sqrt(power.mean())))

    # Second, relative pass: drop everything well below the ungated average.
    relative_gate = kept.mean() * 10.0 ** (-_RELATIVE_GATE_DB / 10.0)
    speech = kept[kept > relative_gate]
    if speech.size == 0:
        speech = kept
    return _to_db(float(np.sqrt(speech.mean())))


def frame_rms_db(
    wav: np.ndarray, sr: int, frame_ms: float = 50.0, hop_ms: float = 25.0
) -> np.ndarray:
    """Level of every frame in dBFS, as a 1-D array.

    The shape of this curve over time is what tells a dropped sentence or a
    segment cut off mid-word apart from a clean one, so :mod:`narration.quality`
    reads it rather than re-deriving the framing itself.
    """
    wav = as_float_mono(wav)
    if wav.size == 0:
        return np.zeros(0, dtype=np.float32)
    power = _frame_power(wav, sr, frame_ms, hop_ms)
    if power.size == 0:
        return np.zeros(0, dtype=np.float32)
    return (10.0 * np.log10(np.maximum(power, _EPS))).astype(np.float32)


def peak_db(wav: np.ndarray) -> float:
    """Sample peak in dBFS."""
    wav = as_float_mono(wav)
    if wav.size == 0:
        return -np.inf
    return _to_db(float(np.max(np.abs(wav))))


def noise_floor_db(wav: np.ndarray, sr: int, percentile: float = 10.0) -> float:
    """Level of the quietest part of the signal, in dBFS.

    Taken as a low percentile of short-frame energies, so a single clean gap is
    enough to characterise the floor without a silence detector.
    """
    wav = as_float_mono(wav)
    if wav.size == 0:
        return -np.inf
    power = _frame_power(wav, sr, frame_ms=50.0, hop_ms=25.0)
    if power.size == 0:
        return -np.inf
    return _to_db(float(np.sqrt(max(np.percentile(power, percentile), 0.0))))


def room_tone_sec(wav: np.ndarray, sr: int) -> Tuple[float, float]:
    """Seconds of silence before the first word and after the last one.

    A file with no speech at all reports its whole length as head silence and
    nothing as tail, which is what an all-silence chapter deserves to be told.
    """
    wav = as_float_mono(wav)
    if wav.size == 0 or not sr:
        return 0.0, 0.0
    bounds = speech_bounds(wav, sr)
    if bounds is None:
        return float(wav.size) / sr, 0.0
    start, end = bounds
    return float(start) / sr, float(max(0, wav.size - end)) / sr


def acx_report(wav: np.ndarray, sr: int) -> dict:
    """Measure a chapter against the ACX limits and say which ones it meets.

    Level *and* shape: a chapter can sit perfectly in the loudness window and
    still be rejected for opening on its first syllable or running past two
    hours, so both are reported side by side.
    """
    from . import polish as polish_tools  # circular at module scope, see stitch

    rms = speech_rms_db(wav, sr)
    peak = peak_db(wav)
    floor = noise_floor_db(wav, sr)
    head, tail = room_tone_sec(wav, sr)
    duration = float(np.asarray(wav).shape[0]) / sr if sr else 0.0
    checks = {
        "rms_ok": ACX_RMS_MIN_DB <= rms <= ACX_RMS_MAX_DB,
        "peak_ok": peak <= ACX_PEAK_CEILING_DB,
        "noise_floor_ok": floor <= ACX_NOISE_FLOOR_DB,
        "head_room_ok": ACX_HEAD_ROOM_MIN_SEC <= head <= ACX_HEAD_ROOM_MAX_SEC,
        "tail_room_ok": ACX_TAIL_ROOM_MIN_SEC <= tail <= ACX_TAIL_ROOM_MAX_SEC,
        "duration_ok": duration <= ACX_MAX_FILE_SEC,
    }
    return {
        "rms_db": rms,
        "peak_db": peak,
        "noise_floor_db": floor,
        # Reported, never gated on: ACX states its limits in RMS, while the
        # streaming platforms normalise in LUFS and disagree on the target.
        # Inventing a pass/fail no standard states would be worse than a number.
        "lufs": polish_tools.loudness_lufs(wav, sr),
        "head_room_sec": head,
        "tail_room_sec": tail,
        "duration_sec": duration,
        **checks,
        "compliant": all(checks.values()),
    }


# --------------------------------------------------------------------------
# Processing
# --------------------------------------------------------------------------


def silence(sr: int, seconds: float) -> np.ndarray:
    """A block of digital silence."""
    return np.zeros(max(0, int(round(sr * max(0.0, seconds)))), dtype=np.float32)


def trim_silence(
    wav: np.ndarray,
    sr: int,
    *,
    relative_db: float = 25.0,
    keep_ms: float = 60.0,
    frame_ms: float = 20.0,
) -> np.ndarray:
    """Cut leading and trailing silence, keeping a short margin.

    The threshold is relative to the segment's own speech level rather than an
    absolute dBFS value, because segments arrive un-normalised and a fixed
    threshold would either clip the start of a quiet segment or trim nothing at
    all from a loud one.
    """
    wav = as_float_mono(wav)
    bounds = speech_bounds(wav, sr, relative_db=relative_db, frame_ms=frame_ms)
    if bounds is None:
        return wav

    start, end = bounds
    margin = int(sr * max(0.0, keep_ms) / 1000.0)
    return wav[max(0, start - margin) : min(wav.size, end + margin)]


def speech_bounds(
    wav: np.ndarray,
    sr: int,
    *,
    relative_db: float = 25.0,
    frame_ms: float = 20.0,
) -> Optional[Tuple[int, int]]:
    """First and last sample carrying speech, or ``None`` if none does.

    The threshold is relative to the signal's own speech level rather than an
    absolute dBFS value, because segments arrive un-normalised and a fixed
    threshold would either clip the start of a quiet one or trim nothing at all
    from a loud one.
    """
    wav = as_float_mono(wav)
    if wav.size == 0:
        return None

    hop_ms = frame_ms / 2.0
    power = _frame_power(wav, sr, frame_ms, hop_ms)
    if power.size == 0:
        return None

    threshold = 10.0 ** ((speech_rms_db(wav, sr) - relative_db) / 10.0)
    loud = np.flatnonzero(power > threshold)
    if loud.size == 0:
        return None

    hop = max(1, int(sr * hop_ms / 1000.0))
    frame = max(1, int(sr * frame_ms / 1000.0))
    return int(loud[0]) * hop, min(wav.size, int(loud[-1]) * hop + frame)


def speech_seconds(
    wav: np.ndarray,
    sr: int,
    *,
    relative_db: float = 25.0,
    frame_ms: float = 20.0,
) -> float:
    """Seconds of the signal that actually carry speech.

    Not the file's length, and not the span between its first and last word
    either: the silence *inside* that span is excluded too. That is what makes
    the number comparable with a transcript — a pause carries no characters, so
    counting it would make a speaker who breathes measure as a slower speaker.

    Same relative threshold as :func:`speech_bounds`, for the same reason: a
    recording arrives at whatever level it was made at.
    """
    wav = as_float_mono(wav)
    if wav.size == 0 or not sr:
        return 0.0

    hop_ms = frame_ms / 2.0
    power = _frame_power(wav, sr, frame_ms, hop_ms)
    if power.size == 0:
        return 0.0

    level = speech_rms_db(wav, sr)
    if not np.isfinite(level):
        return 0.0

    threshold = 10.0 ** ((level - relative_db) / 10.0)
    loud = int(np.count_nonzero(power > threshold))
    # Frames overlap, so each one stands for a hop's worth of signal.
    return min(float(wav.size) / sr, loud * hop_ms / 1000.0)


def fade_edges(wav: np.ndarray, sr: int, fade_ms: float = 8.0) -> np.ndarray:
    """Ramp the first and last few milliseconds so joins do not click.

    Cutting a waveform at a non-zero sample leaves a step discontinuity, which is
    exactly the click heard at every segment boundary in naive concatenation.
    """
    wav = as_float_mono(wav).copy()
    length = int(sr * max(0.0, fade_ms) / 1000.0)
    if length <= 0 or wav.size == 0:
        return wav
    length = min(length, wav.size // 2)
    if length <= 0:
        return wav
    ramp = np.linspace(0.0, 1.0, length, dtype=np.float32)
    wav[:length] *= ramp
    wav[-length:] *= ramp[::-1]
    return wav


def normalize_level(
    wav: np.ndarray,
    sr: int,
    *,
    target_rms_db: float = -20.0,
    peak_ceiling_db: float = ACX_PEAK_CEILING_DB,
) -> Tuple[np.ndarray, float]:
    """Scale to the target speech RMS without breaching the peak ceiling.

    Returns ``(audio, applied_gain_db)``. When the RMS target would push peaks
    above the ceiling the gain is reduced to respect the ceiling instead: a
    breached ceiling is a hard rejection at distribution, a slightly quiet
    chapter is not.
    """
    wav = as_float_mono(wav)
    if wav.size == 0:
        return wav, 0.0

    current_peak = peak_db(wav)
    current_rms = speech_rms_db(wav, sr)
    # Silence carries no level to correct. Without this guard, an all-zero
    # segment measures around -240 dBFS and asks for 220 dB of gain — harmless
    # on true digital silence, but it would explode any dither or DC residue.
    if not np.isfinite(current_rms) or current_peak < _SILENCE_FLOOR_DB:
        return wav, 0.0

    gain_db = target_rms_db - current_rms
    if np.isfinite(current_peak):
        gain_db = min(gain_db, peak_ceiling_db - current_peak)

    gain = float(10.0 ** (gain_db / 20.0))
    return (wav * np.float32(gain)).astype(np.float32), float(gain_db)


def master_segment(
    wav: np.ndarray,
    sr: int,
    settings: MasteringSettings = MasteringSettings(),
) -> np.ndarray:
    """Trim and de-click a single generated segment.

    Level is deliberately *not* set here. Normalising each segment separately
    would flatten the natural dynamics between a whispered line and a shouted
    one; the chapter is normalised once, as a whole, in :func:`stitch`.
    """
    wav = remove_dc(wav)
    if wav.size == 0:
        return wav
    if settings.trim_silence:
        wav = trim_silence(
            wav,
            sr,
            relative_db=settings.trim_relative_db,
            keep_ms=settings.trim_keep_ms,
        )
    return fade_edges(wav, sr, settings.fade_ms)


def stitch(
    segments: Sequence[Tuple[np.ndarray, float]],
    sr: int,
    settings: MasteringSettings = MasteringSettings(),
    *,
    normalize: bool = True,
    polish_settings=None,
) -> np.ndarray:
    """Assemble ``(audio, pause_after_seconds)`` pairs into one mastered chapter.

    Each segment is trimmed and faded, the requested pause is inserted after it,
    the studio chain runs over the whole chapter, and only then is it normalised
    — once, so the level is consistent from the first word to the last, and so
    that nothing after the levelling can move it again.
    """
    if not segments:
        return np.zeros(0, dtype=np.float32)

    pieces: List[np.ndarray] = []
    if settings.lead_sec > 0:
        pieces.append(silence(sr, settings.lead_sec))

    last = len(segments) - 1
    for index, (wav, pause_after) in enumerate(segments):
        processed = master_segment(wav, sr, settings)
        if processed.size:
            pieces.append(processed)
        if index != last and pause_after > 0:
            pieces.append(silence(sr, pause_after))

    if settings.tail_sec > 0:
        pieces.append(silence(sr, settings.tail_sec))

    chapter = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    if settings.polish and chapter.size:
        # Imported here rather than at module scope: polish builds on this
        # module, and importing it at the top would close the circle.
        from . import polish as polish_tools

        chapter = polish_tools.polish(
            chapter, sr, polish_settings or polish_tools.PolishSettings()
        )

    if normalize and chapter.size:
        chapter, _ = normalize_level(
            chapter,
            sr,
            target_rms_db=settings.target_rms_db,
            peak_ceiling_db=settings.peak_ceiling_db,
        )
    return chapter


def concatenate(
    parts: Iterable[np.ndarray],
    sr: int,
    gap_sec: float = 0.0,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """Join already-mastered blocks with an optional gap between them."""
    blocks: List[np.ndarray] = []
    for index, part in enumerate(parts):
        data = as_float_mono(part)
        if index and gap_sec > 0:
            blocks.append(silence(sr, gap_sec))
        blocks.append(data)
    if not blocks:
        return np.zeros(0, dtype=np.float32)
    joined = np.concatenate(blocks)
    return joined.astype(dtype) if dtype is not None else joined
