"""The processing that separates correct levels from a produced sound.

:mod:`narration.audio` gets a chapter to the right loudness, the right peak and
the right silences — which is what a distributor *checks*. It is not what a
listener hears. Three things are missing from it, and every audiobook studio
does all three:

* **A high-pass.** Below 80 Hz there is nothing a voice needs and plenty a room
  produces: a rumble that is inaudible on a laptop, eats headroom, and wears the
  ear down over an hour in headphones.
* **De-essing.** Sibilance is the first thing that gives synthetic French away —
  the ``s`` of *ses histoires* arriving several decibels above the vowels around
  it. It is a narrow, high band, and it only needs pulling down when it spikes.
* **Compression.** An audiobook is listened to while walking, driving, falling
  asleep. The gap between a murmured line and an emphatic one has to close, or
  half the sentences are lost under the road noise.
* **Limiting.** Compression alone made two of the three test chapters *worse*:
  it pulls sustained speech down without touching the short peaks, the crest
  factor grows, and the -3 dBFS ceiling then drags the whole chapter quiet to
  make room for a handful of samples — 1.3 LU lost, measured. Holding the peaks
  is what lets the rest sit where it belongs.

And one measurement is missing. ACX reasons in RMS, which this pipeline already
reports; Spotify, Apple Books and YouTube normalise in **LUFS** (ITU-R BS.1770 /
EBU R128), which weights the spectrum the way an ear does. A chapter sitting
perfectly at -20 dBFS RMS can still arrive too loud or too quiet on those
platforms, and nothing in the ACX report would have said so.

Order matters and is not negotiable: correct first (high-pass), then control
dynamics (de-ess, compress, limit), then set the level. Normalising before
compressing would undo the level; compressing before the high-pass would make
the compressor duck on rumble nobody can hear; limiting before compressing would
leave the limiter working on peaks the compressor is about to move anyway.

Gains are computed at a **control rate** of one point per millisecond and
interpolated back, rather than per sample. That is how hardware does it, it is
inaudible at these time constants, and it keeps an hour-long chapter to a couple
of seconds of work instead of minutes of Python loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import signal

from . import audio as audio_tools

__all__ = [
    "PolishSettings",
    "compress",
    "expand_down",
    "deess",
    "highpass",
    "limit",
    "loudness_lufs",
    "polish",
]

#: Gain is computed this often, then interpolated back to the sample rate.
_CONTROL_MS = 1.0

#: ITU-R BS.1770-4 K-weighting, stage 1: a high shelf standing in for the head.
_SHELF_F0 = 1681.974450955533
_SHELF_GAIN_DB = 3.999843853973347
_SHELF_Q = 0.7071752369554196
#: Stage 2: the RLB high-pass that discards what the ear barely weighs.
_RLB_F0 = 38.13547087602444
_RLB_Q = 0.5003270373238773
#: The offset in the BS.1770 loudness equation.
_LUFS_OFFSET = -0.691
#: Gating, in LUFS and LU: silence never counts, and neither do the quiet parts.
_ABSOLUTE_GATE_LUFS = -70.0
_RELATIVE_GATE_LU = -10.0
_BLOCK_SEC = 0.400
_BLOCK_OVERLAP = 0.75


@dataclass(frozen=True)
class PolishSettings:
    """How much of each treatment. Defaults are deliberately conservative.

    A narrator's voice is the product; processing it heavily is how an audiobook
    starts sounding like a radio advert. These values remove the defects and
    stop there.
    """

    #: Everything below this is rumble, not voice.
    highpass_hz: float = 80.0

    deess: bool = True
    #: Sibilance lives above this. French ``s`` and ``ch`` sit around 5-8 kHz.
    deess_band_hz: float = 5000.0
    #: How far above the band's own average a peak has to be to be pulled down.
    deess_threshold_db: float = 6.0
    #: Ratio applied to the excess. 3:1 tames without lisping.
    deess_ratio: float = 3.0
    #: Never pull the band down by more than this, whatever the excess.
    deess_max_reduction_db: float = 8.0

    #: Baisser le fond pendant les silences. Une voix clonée hérite du bruit de
    #: sa référence : Alex Somerset rend un plancher à -58 dBFS là où Aurore est
    #: à -70, et l'ACX refuse tout ce qui dépasse -60. Le bruit se concentre
    #: dans 80-150 Hz, qui est aussi le fondamental d'une voix masculine, donc
    #: un passe-haut plus haut amaigrirait la voix : il faut agir dans le temps,
    #: pas en fréquence.
    expand: bool = True
    #: Seuil, relatif au niveau de parole du chapitre — comme la compression,
    #: parce qu'un seuil absolu écraserait une prise forte et raterait une prise
    #: faible.
    expand_threshold_db: float = -30.0
    expand_ratio: float = 2.5
    #: Plafonner la réduction : un silence poussé à -100 dB s'entend comme un
    #: trou, ce qui est un défaut d'un autre genre.
    expand_max_reduction_db: float = 16.0
    expand_attack_ms: float = 5.0
    expand_release_ms: float = 220.0

    compress: bool = True
    #: Relative to the signal's own speech level, not an absolute dBFS value:
    #: the chapter arrives un-normalised and a fixed threshold would either do
    #: nothing or crush it.
    compress_threshold_db: float = -6.0
    compress_ratio: float = 2.5
    compress_attack_ms: float = 25.0
    compress_release_ms: float = 300.0

    limit: bool = True
    #: How far the loudest instants may stand above the speech level. A
    #: narrated chapter naturally sits around 14 dB; letting it run to 19 costs
    #: real loudness, because the -3 dBFS ceiling then forces the whole chapter
    #: down to make room for a handful of samples.
    limit_crest_db: float = 14.0
    limit_release_ms: float = 80.0

    @property
    def enabled(self) -> bool:
        return bool(self.highpass_hz or self.expand or self.deess or self.compress or self.limit)


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------


def _shelf_biquad(sr: int) -> np.ndarray:
    """BS.1770 stage 1, from the audio EQ cookbook so it follows the rate."""
    amplitude = 10.0 ** (_SHELF_GAIN_DB / 40.0)
    omega = 2.0 * np.pi * _SHELF_F0 / sr
    alpha = np.sin(omega) / (2.0 * _SHELF_Q)
    cosine = np.cos(omega)
    root = 2.0 * np.sqrt(amplitude) * alpha

    b0 = amplitude * ((amplitude + 1) + (amplitude - 1) * cosine + root)
    b1 = -2.0 * amplitude * ((amplitude - 1) + (amplitude + 1) * cosine)
    b2 = amplitude * ((amplitude + 1) + (amplitude - 1) * cosine - root)
    a0 = (amplitude + 1) - (amplitude - 1) * cosine + root
    a1 = 2.0 * ((amplitude - 1) - (amplitude + 1) * cosine)
    a2 = (amplitude + 1) - (amplitude - 1) * cosine - root
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])


def _rlb_biquad(sr: int) -> np.ndarray:
    """BS.1770 stage 2: a plain high-pass."""
    omega = 2.0 * np.pi * _RLB_F0 / sr
    alpha = np.sin(omega) / (2.0 * _RLB_Q)
    cosine = np.cos(omega)

    b0 = (1.0 + cosine) / 2.0
    b1 = -(1.0 + cosine)
    b2 = (1.0 + cosine) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cosine
    a2 = 1.0 - alpha
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])


def _bands(wav: np.ndarray, sr: int, cutoff_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """Split into (below cutoff, above cutoff), summing back to the original."""
    high = highpass(wav, sr, cutoff_hz)
    return wav - high, high


def highpass(wav: np.ndarray, sr: int, cutoff_hz: float = 80.0, order: int = 2) -> np.ndarray:
    """Remove everything below ``cutoff_hz``, without phase smear.

    Zero-phase (forward then backward), because this runs offline on a finished
    chapter and there is no reason to accept the group delay a live filter would
    impose on the transients of the consonants.
    """
    wav = audio_tools.as_float_mono(wav)
    if wav.size == 0 or cutoff_hz <= 0 or cutoff_hz >= sr / 2:
        return wav
    sos = signal.butter(order, cutoff_hz / (sr / 2.0), btype="highpass", output="sos")
    # filtfilt needs a few times the filter length to work on; a very short
    # segment is left alone rather than raising.
    if wav.size <= 3 * (sos.shape[0] * 2 + 1):
        return wav
    return signal.sosfiltfilt(sos, wav).astype(np.float32)


# --------------------------------------------------------------------------
# Dynamics
# --------------------------------------------------------------------------


def _control_envelope_db(
    wav: np.ndarray, sr: int, window_ms: float = 30.0
) -> Tuple[np.ndarray, int]:
    """RMS in dB over a sliding window, one point per control period.

    The window is what makes this a *level* detector rather than a peak
    detector. Measured over a single millisecond, the envelope follows every
    glottal pulse: the compressor then lets short transients through and squashes
    sustained vowels, which raises the crest factor instead of lowering it —
    measured at +3 dB on a real chapter before this window existed. Thirty
    milliseconds is a syllable, which is the scale speech levelling works at.
    """
    hop = max(1, int(sr * _CONTROL_MS / 1000.0))
    window = max(hop, int(sr * window_ms / 1000.0))
    if wav.size < window:
        return np.zeros(0, dtype=np.float64), hop

    cumulative = np.concatenate(([0.0], np.cumsum(np.square(wav, dtype=np.float64))))
    starts = np.arange(0, wav.size - window + 1, hop)
    power = (cumulative[starts + window] - cumulative[starts]) / window
    return 10.0 * np.log10(np.maximum(power, 1e-20)), hop


def _smooth_gain(gain_db: np.ndarray, attack_ms: float, release_ms: float) -> np.ndarray:
    """Attack/release smoothing of a gain curve, at control rate.

    Two different time constants mean a branch per point, so this is the one
    genuine loop in the module — over milliseconds, not samples, which is what
    makes it affordable.
    """
    if gain_db.size == 0:
        return gain_db
    attack = np.exp(-_CONTROL_MS / max(attack_ms, 1e-6))
    release = np.exp(-_CONTROL_MS / max(release_ms, 1e-6))
    out = np.empty_like(gain_db)
    current = gain_db[0]
    for index, target in enumerate(gain_db):
        # Going down (more reduction) is the attack; coming back is the release.
        coefficient = attack if target < current else release
        current = coefficient * current + (1.0 - coefficient) * target
        out[index] = current
    return out


def _apply_control_gain(wav: np.ndarray, gain_db: np.ndarray, hop: int) -> np.ndarray:
    """Interpolate a control-rate gain back onto the samples and apply it."""
    if gain_db.size == 0:
        return wav
    positions = np.arange(gain_db.size) * hop + hop / 2.0
    gain = np.interp(np.arange(wav.size), positions, gain_db)
    return (wav * (10.0 ** (gain / 20.0))).astype(np.float32)


def deess(
    wav: np.ndarray,
    sr: int,
    settings: PolishSettings = PolishSettings(),
) -> np.ndarray:
    """Pull down sibilance when it spikes, and only then.

    The high band is measured against **its own average**, not against the whole
    signal: what makes an ``s`` harsh is that it stands out from the other
    ``s`` sounds and from the vowels, and that comparison has to be made in the
    band where it happens.

    Only the high band is attenuated — the rest of the voice passes untouched,
    which is what keeps this from sounding like a blanket over the narrator.
    """
    wav = audio_tools.as_float_mono(wav)
    if wav.size == 0 or not settings.deess:
        return wav

    low, high = _bands(wav, sr, settings.deess_band_hz)
    # A sibilant lasts a fraction of a syllable, so it is watched over a shorter
    # window than the levelling uses, or it would be averaged away.
    envelope_db, hop = _control_envelope_db(high, sr, window_ms=12.0)
    voice_db, _ = _control_envelope_db(wav, sr, window_ms=12.0)
    if envelope_db.size == 0 or voice_db.size == 0:
        return wav

    # The reference is where the band normally sits *while someone is talking*.
    # Measuring it over frames where the band itself is loud would compare
    # sibilance to sibilance — a lone harsh `s` would then be its own reference
    # and never exceed it. Gating on the voice instead makes vowels the
    # baseline, which is what an `s` actually stands out from. The median, not
    # the mean, so a handful of spikes cannot lift the very threshold meant to
    # catch them.
    voiced = voice_db > voice_db.max() - 40.0
    band_while_voiced = envelope_db[: voiced.size][voiced[: envelope_db.size]]
    if band_while_voiced.size == 0:
        return wav
    threshold = float(np.median(band_while_voiced)) + settings.deess_threshold_db

    excess = np.maximum(0.0, envelope_db - threshold)
    reduction = -excess * (1.0 - 1.0 / max(settings.deess_ratio, 1.0))
    reduction = np.maximum(reduction, -abs(settings.deess_max_reduction_db))
    reduction = _smooth_gain(reduction, attack_ms=2.0, release_ms=40.0)

    return (low + _apply_control_gain(high, reduction, hop)).astype(np.float32)


def expand_down(
    wav: np.ndarray,
    sr: int,
    settings: PolishSettings = PolishSettings(),
) -> np.ndarray:
    """Pousser le fond vers le bas pendant les silences, sans toucher la voix.

    Le seuil est relatif au niveau de parole du chapitre, comme pour la
    compression. En dessous, le gain descend selon le rapport, plafonné : un
    silence creusé à l'excès s'entend comme un trou, et un trou est un défaut
    au même titre qu'un souffle.
    """
    wav = audio_tools.as_float_mono(wav)
    if wav.size == 0 or not settings.expand:
        return wav

    envelope_db, hop = _control_envelope_db(wav, sr)
    if envelope_db.size == 0:
        return wav

    speech_db = audio_tools.speech_rms_db(wav, sr)
    if not np.isfinite(speech_db):
        return wav
    threshold = speech_db + settings.expand_threshold_db

    deficit = np.maximum(0.0, threshold - envelope_db)
    reduction = -deficit * (max(settings.expand_ratio, 1.0) - 1.0)
    reduction = np.maximum(reduction, -abs(settings.expand_max_reduction_db))
    reduction = _smooth_gain(
        reduction, settings.expand_attack_ms, settings.expand_release_ms
    )
    return _apply_control_gain(wav, reduction, hop)


def compress(
    wav: np.ndarray,
    sr: int,
    settings: PolishSettings = PolishSettings(),
) -> np.ndarray:
    """Close the gap between the quiet lines and the loud ones.

    The threshold is relative to the chapter's own speech level, because a
    chapter arrives here un-normalised: an absolute dBFS threshold would crush
    a loud take and leave a quiet one untouched, which is the opposite of what
    consistency means.

    Make-up gain is deliberately *not* applied. The level is set once, later,
    by the normalisation — adding gain here would only move the target.
    """
    wav = audio_tools.as_float_mono(wav)
    if wav.size == 0 or not settings.compress:
        return wav

    envelope_db, hop = _control_envelope_db(wav, sr)
    if envelope_db.size == 0:
        return wav

    speech_db = audio_tools.speech_rms_db(wav, sr)
    if not np.isfinite(speech_db):
        return wav
    threshold = speech_db + settings.compress_threshold_db

    excess = np.maximum(0.0, envelope_db - threshold)
    reduction = -excess * (1.0 - 1.0 / max(settings.compress_ratio, 1.0))
    reduction = _smooth_gain(
        reduction, settings.compress_attack_ms, settings.compress_release_ms
    )
    return _apply_control_gain(wav, reduction, hop)


def _peak_envelope_db(wav: np.ndarray, sr: int, window_ms: float = 3.0) -> Tuple[np.ndarray, int]:
    """Highest sample in a short window, per control period.

    A limiter has to see the sample that will breach the ceiling, so this looks
    at the peak rather than the RMS the compressor uses.
    """
    hop = max(1, int(sr * _CONTROL_MS / 1000.0))
    window = max(hop, int(sr * window_ms / 1000.0))
    if wav.size < window:
        return np.zeros(0, dtype=np.float64), hop
    starts = np.arange(0, wav.size - window + 1, hop)
    # A strided view costs no copy: window is a few dozen samples.
    frames = np.lib.stride_tricks.sliding_window_view(np.abs(wav), window)[starts]
    return 20.0 * np.log10(np.maximum(frames.max(axis=1), 1e-10)), hop


def limit(
    wav: np.ndarray,
    sr: int,
    settings: PolishSettings = PolishSettings(),
) -> np.ndarray:
    """Hold the loudest instants down so the whole chapter can sit louder.

    Without this the compressor makes things *worse* on some chapters: it pulls
    sustained speech down without touching short peaks, the crest factor grows,
    and the -3 dBFS ceiling then drags the entire chapter quiet to accommodate a
    few samples. Measured on a real chapter: crest 17.7 dB before, 19.0 after
    compression alone, and 1.3 LU of loudness lost to it.

    Attack is immediate by construction — the gain is computed from a peak
    envelope, so a breach is caught in the millisecond it happens — and only the
    release is smoothed, which is what keeps it from pumping.
    """
    wav = audio_tools.as_float_mono(wav)
    if wav.size == 0 or not settings.limit:
        return wav

    speech_db = audio_tools.speech_rms_db(wav, sr)
    if not np.isfinite(speech_db):
        return wav

    envelope_db, hop = _peak_envelope_db(wav, sr)
    if envelope_db.size == 0:
        return wav

    threshold = speech_db + settings.limit_crest_db
    reduction = np.minimum(0.0, threshold - envelope_db)
    # Attack of zero: never let a peak through. Release smoothed, or the gain
    # would step back up inside a syllable and pump audibly.
    reduction = _smooth_gain(reduction, attack_ms=0.01, release_ms=settings.limit_release_ms)
    return _apply_control_gain(wav, reduction, hop)


# --------------------------------------------------------------------------
# Loudness
# --------------------------------------------------------------------------


def loudness_lufs(wav: np.ndarray, sr: int) -> float:
    """Integrated loudness in LUFS, per ITU-R BS.1770-4 / EBU R128.

    Two K-weighting biquads, 400 ms blocks overlapping by three quarters, then
    the two gates: everything below -70 LUFS is silence and never counts, and
    everything more than 10 LU below the ungated average is the quiet part of
    the programme and does not count either.

    Returns ``-inf`` for a signal with nothing in it, which is the honest answer
    rather than a number.
    """
    wav = audio_tools.as_float_mono(wav)
    if wav.size == 0 or sr <= 0:
        return float("-inf")

    weighted = signal.sosfilt(
        np.vstack([_shelf_biquad(sr), _rlb_biquad(sr)]), wav.astype(np.float64)
    )

    block = int(round(_BLOCK_SEC * sr))
    step = max(1, int(round(block * (1.0 - _BLOCK_OVERLAP))))
    if weighted.size < block:
        return float("-inf")

    starts = np.arange(0, weighted.size - block + 1, step)
    cumulative = np.concatenate(([0.0], np.cumsum(np.square(weighted))))
    power = (cumulative[starts + block] - cumulative[starts]) / block

    loudness = _LUFS_OFFSET + 10.0 * np.log10(np.maximum(power, 1e-20))
    above_absolute = power[loudness > _ABSOLUTE_GATE_LUFS]
    if above_absolute.size == 0:
        return float("-inf")

    ungated = _LUFS_OFFSET + 10.0 * np.log10(np.mean(above_absolute))
    relative_gate = ungated + _RELATIVE_GATE_LU
    kept = above_absolute[
        _LUFS_OFFSET + 10.0 * np.log10(np.maximum(above_absolute, 1e-20)) > relative_gate
    ]
    if kept.size == 0:
        return float(ungated)
    return float(_LUFS_OFFSET + 10.0 * np.log10(np.mean(kept)))


# --------------------------------------------------------------------------
# The chain
# --------------------------------------------------------------------------


def polish(
    wav: np.ndarray,
    sr: int,
    settings: PolishSettings = PolishSettings(),
) -> np.ndarray:
    """Correct, then control, then hand back for levelling.

    Deliberately does not set the level: :func:`narration.audio.stitch`
    normalises the chapter once, after this, and doing it in both places would
    mean neither is in charge.
    """
    wav = audio_tools.as_float_mono(wav)
    if wav.size == 0 or not settings.enabled:
        return wav
    if settings.highpass_hz:
        wav = highpass(wav, sr, settings.highpass_hz)
    wav = expand_down(wav, sr, settings)
    wav = deess(wav, sr, settings)
    wav = compress(wav, sr, settings)
    return limit(wav, sr, settings)
