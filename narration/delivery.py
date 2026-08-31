"""Turn finished chapters into the folder a distributor actually accepts.

An M4B is what you listen to. It is not what you upload. ACX — and every
platform that mirrors it — takes **one file per chapter**, encoded to a fixed
specification, plus a short retail sample, and rejects the lot over details that
have nothing to do with how the narration sounds:

* **192 kbps CBR MP3 at 44.1 kHz.** Not 96 kbps, not variable bitrate, whatever
  the synthesis sample rate was. The resampling happens at encode time.
* **No file longer than 120 minutes, and none larger than 170 MB.** At 192 kbps
  constant bitrate those two limits are close enough to swap places — two hours
  comes to about 173 MB — so which one binds is computed rather than assumed,
  and the split follows whichever is tighter.
* **A retail sample of 1 to 5 minutes**, taken from the book itself.

This module prepares all of that. What it does *not* do is encode: that needs
ffmpeg, which may not be installed, and the hours of synthesis behind a book
must not be held hostage to a missing binary. Every function here works on
audio and paths and hands back the exact command to run, exactly as
:mod:`narration.assemble` does.

The cut points matter more than they look. A chapter split at a fixed offset
lands mid-word; a sample that ends at its target second stops mid-sentence. Both
are cut at the quietest moment in a window around the target instead, which is
where the narrator was drawing breath.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import audio as audio_tools

__all__ = [
    "ACX_MAX_FILE_BYTES",
    "ACX_PROFILE",
    "DeliveryProfile",
    "SAMPLE_MAX_SEC",
    "SAMPLE_MIN_SEC",
    "check_delivered",
    "encode_command",
    "max_seconds_for",
    "retail_sample",
    "split_for_delivery",
]

#: Largest file a distributor accepts. ACX writes "170 MB" without saying
#: whether it means 170 x 1000^2 or 170 x 1024^2 bytes; the smaller reading is
#: taken, because being under a limit that turns out to be larger costs one
#: extra file and being over one costs a rejected submission.
ACX_MAX_FILE_BYTES = 170 * 1000 * 1000
#: Retail sample bounds, in seconds.
SAMPLE_MIN_SEC = 60.0
SAMPLE_MAX_SEC = 300.0
#: What a sample aims for when nothing else is asked: long enough to judge the
#: voice, short enough that nobody stops it halfway.
SAMPLE_TARGET_SEC = 150.0

#: How far back from a limit the cut may wander to land in a pause, in seconds.
_CUT_WINDOW_SEC = 20.0
#: Frame resolution used to hunt for that pause.
_CUT_FRAME_MS = 50.0
#: A frame this far below the loudest of the window is a pause, not a dip.
_PAUSE_DEPTH_DB = 25.0
#: The cut never reaches further back than this fraction of the limit, so a
#: chapter with no pause at all still advances instead of emitting empty files.
_CUT_FLOOR_FRACTION = 0.6


@dataclass(frozen=True)
class DeliveryProfile:
    """An encoding a distributor will accept, and the limits that come with it."""

    name: str
    codec: str
    bitrate_kbps: int
    sample_rate: int
    channels: int
    suffix: str
    max_seconds: float
    max_bytes: int

    @property
    def bytes_per_second(self) -> float:
        """Constant bitrate makes file size a straight function of duration."""
        return self.bitrate_kbps * 1000.0 / 8.0


#: The ACX specification, which Amazon, Apple Books, Kobo and Google Play follow.
ACX_PROFILE = DeliveryProfile(
    name="ACX",
    codec="libmp3lame",
    bitrate_kbps=192,
    sample_rate=44100,
    channels=1,
    suffix=".mp3",
    max_seconds=float(audio_tools.ACX_MAX_FILE_SEC),
    max_bytes=ACX_MAX_FILE_BYTES,
)


def max_seconds_for(profile: DeliveryProfile = ACX_PROFILE) -> float:
    """Longest a single delivered file may run under *both* limits.

    The clock says 120 minutes and the file size says about 118 at 192 kbps.
    Whichever binds first is the one that decides, so it is computed rather
    than assumed.
    """
    from_size = profile.max_bytes / profile.bytes_per_second
    return min(profile.max_seconds, from_size)


def _pause_before(wav: np.ndarray, sr: int, limit: int, window: int) -> int:
    """The last pause at or before ``limit``, in samples.

    Backwards only, never past the limit: this places a cut that must not
    exceed a duration or a file size, and a pause two seconds after the limit
    is a pause that puts the file over it.

    The *last* silent frame, not the quietest one. A chapter's quietest moment
    can be a dip inside a sentence twenty seconds earlier; taking it would throw
    away capacity for no reason. Only when no frame in the window is properly
    silent does the quietest one stand in.

    The search never reaches back past :data:`_CUT_FLOOR_FRACTION` of the limit,
    which is what guarantees the caller makes progress: a cut at sample one
    would loop forever producing empty files.
    """
    limit = int(min(max(limit, 0), wav.size))
    start = max(int(limit * _CUT_FLOOR_FRACTION), limit - window)
    if limit - start <= 1:
        return limit

    hop_ms = _CUT_FRAME_MS / 2.0
    levels = audio_tools.frame_rms_db(
        wav[start:limit], sr, frame_ms=_CUT_FRAME_MS, hop_ms=hop_ms
    )
    if levels.size == 0:
        return limit

    quiet = np.flatnonzero(levels < float(levels.max()) - _PAUSE_DEPTH_DB)
    index = int(quiet[-1]) if quiet.size else int(np.argmin(levels))
    hop = max(1, int(sr * hop_ms / 1000.0))
    return int(min(start + index * hop, limit))


def split_for_delivery(
    wav: np.ndarray,
    sr: int,
    profile: DeliveryProfile = ACX_PROFILE,
    *,
    max_seconds: Optional[float] = None,
) -> List[np.ndarray]:
    """Cut a chapter into parts no delivered file limit can reject.

    A chapter that already fits comes back as a single part, untouched — the
    common case, and it must stay bit-for-bit what the mastering produced.

    Each part is cut in a pause and given the room tone ACX expects at the two
    ends, since each part becomes a file in its own right.
    """
    wav = audio_tools.as_float_mono(wav)
    limit = float(max_seconds if max_seconds is not None else max_seconds_for(profile))
    if wav.size == 0 or limit <= 0 or wav.size / sr <= limit:
        return [wav]

    # Each part becomes a file of its own and gets room tone at both ends, so
    # the budget for actual narration is the limit minus that tone. Without
    # this the parts come out fitting the limit and the files do not.
    settings = audio_tools.MasteringSettings()
    room = settings.lead_sec + settings.tail_sec
    budget = limit - room
    if budget <= 0:
        # A limit shorter than its own room tone cannot be honoured; keep the
        # audio whole rather than emit a pile of near-empty files.
        return [wav]

    parts: List[np.ndarray] = []
    remaining = wav
    window = int(sr * _CUT_WINDOW_SEC)
    while remaining.size / sr > budget:
        cut = _pause_before(remaining, sr, int(sr * budget), window)
        cut = min(max(cut, 1), remaining.size - 1)
        parts.append(remaining[:cut])
        remaining = remaining[cut:]
    parts.append(remaining)

    return [
        np.concatenate(
            [
                audio_tools.silence(sr, settings.lead_sec),
                audio_tools.trim_silence(part, sr),
                audio_tools.silence(sr, settings.tail_sec),
            ]
        )
        for part in parts
    ]


def retail_sample(
    wav: np.ndarray,
    sr: int,
    *,
    target_seconds: float = SAMPLE_TARGET_SEC,
    start_seconds: float = 0.0,
) -> np.ndarray:
    """A 1-to-5 minute excerpt, ending where the narrator paused.

    Taken from the book rather than from the credits: a sample is what a buyer
    listens to before deciding, and nobody decides on hearing the title read
    out. The caller picks the chapter; this picks where to stop inside it.
    """
    wav = audio_tools.as_float_mono(wav)
    if wav.size == 0:
        return wav

    target = float(np.clip(target_seconds, SAMPLE_MIN_SEC, SAMPLE_MAX_SEC))
    begin = int(max(0.0, start_seconds) * sr)
    if begin >= wav.size:
        begin = 0

    body = wav[begin:]
    if body.size / sr > target:
        window = int(sr * _CUT_WINDOW_SEC)
        end = _pause_before(body, sr, int(sr * target), window)
        body = body[: max(1, end)]

    settings = audio_tools.MasteringSettings()
    return np.concatenate(
        [
            audio_tools.silence(sr, settings.lead_sec),
            audio_tools.fade_edges(audio_tools.trim_silence(body, sr), sr),
            audio_tools.silence(sr, settings.tail_sec),
        ]
    )


def encode_command(
    wav_path: str | Path,
    out_path: str | Path,
    profile: DeliveryProfile = ACX_PROFILE,
) -> List[str]:
    """The ffmpeg invocation producing one delivery-ready file.

    ``-b:a`` without any quality flag is what makes libmp3lame constant-bitrate;
    a variable-bitrate file is rejected however good it sounds.
    """
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(wav_path),
        "-c:a",
        profile.codec,
        "-b:a",
        f"{profile.bitrate_kbps}k",
        "-ar",
        str(profile.sample_rate),
        "-ac",
        str(profile.channels),
        # Neither a VBR header nor an encoder tag belongs in a delivered file.
        "-write_xing",
        "0",
        "-map_metadata",
        "-1",
        str(out_path),
    ]


def check_delivered(
    wav: np.ndarray,
    sr: int,
    profile: DeliveryProfile = ACX_PROFILE,
    *,
    encoded_bytes: Optional[int] = None,
) -> dict:
    """Everything a distributor checks about one file, in one report.

    Levels and room tone come from :func:`narration.audio.acx_report`; what is
    added here is the pair of limits that only exist once the file is a file —
    its duration against the clock, and its size against the 170 MB cap. The
    size is predicted from the constant bitrate when the file is not encoded
    yet, which is the whole point of checking before spending the encode.
    """
    report = dict(audio_tools.acx_report(wav, sr))
    duration = report["duration_sec"]
    size = (
        float(encoded_bytes)
        if encoded_bytes is not None
        else duration * profile.bytes_per_second
    )
    report.update(
        {
            "profile": profile.name,
            "encoded_bytes": size,
            "encoded_estimated": encoded_bytes is None,
            "duration_ok": duration <= profile.max_seconds,
            "size_ok": size <= profile.max_bytes,
        }
    )
    report["compliant"] = all(
        value for key, value in report.items() if key.endswith("_ok")
    )
    return report


def failures(report: dict) -> List[str]:
    """Readable reasons a file would be sent back, in the order they matter."""
    reasons = {
        "rms_ok": "sonie hors de la fenêtre -23..-18 dBFS",
        "peak_ok": "crête au-dessus de -3 dBFS",
        "noise_floor_ok": "bruit de fond au-dessus de -60 dBFS",
        "head_room_ok": "silence de tête hors de 0,5-1 s",
        "tail_room_ok": "silence de queue hors de 1-5 s",
        "duration_ok": "fichier de plus de 120 min",
        "size_ok": "fichier de plus de 170 Mo",
    }
    return [text for key, text in reasons.items() if report.get(key) is False]
