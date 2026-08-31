"""Join per-chapter WAVs into one deliverable audiobook file.

A directory of forty WAV files is not an audiobook. A listener expects a single
file that remembers where they stopped, with chapters they can skip between —
which means chapter markers, and in practice M4B or a chaptered MP3.

Two responsibilities, deliberately split:

* Concatenation is done here, streaming, with ``soundfile`` alone. It always
  works, it never loads a ten-hour book into memory, and it writes 16-bit PCM
  because that is both the delivery format and a quarter the size of float32.
* Encoding to MP3/M4B needs ``ffmpeg``, which may not be installed. When it is
  missing the concatenated WAV and the chapter-marker file are still produced,
  and the exact command to run later is returned — a missing encoder must not
  cost the hours of synthesis that went into the audio.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import soundfile as sf

__all__ = [
    "Chapter",
    "AssemblyResult",
    "assemble",
    "build_ffmetadata",
    "concat_chapters",
    "ffmpeg_command",
    "find_ffmpeg",
    "normalize_bitrate",
]

#: Encoder settings per container.
#:
#: 64 kbps AAC mono is not a compromise here: it is at or above what Audible
#: itself streams for a finished audiobook (its enhanced format sits around
#: that figure), and speech gains very little above it. MP3 needs more to
#: sound the same, hence 128. Both are overridable — a book that will be
#: re-encoded downstream, or archived, is worth more.
_ENCODERS: Dict[str, Dict[str, str]] = {
    "m4b": {"codec": "aac", "bitrate": "64k"},
    "m4a": {"codec": "aac", "bitrate": "64k"},
    "mp3": {"codec": "libmp3lame", "bitrate": "128k"},
}


def normalize_bitrate(bitrate: str | int | None) -> Optional[str]:
    """``128``, ``"128"`` and ``"128k"`` all mean the same thing to a user.

    Returns None for anything empty, which the callers read as "keep the
    default for this container".
    """
    if bitrate is None:
        return None
    text = str(bitrate).strip().lower()
    if not text:
        return None
    if text.endswith("k"):
        text = text[:-1]
    try:
        value = int(float(text))
    except ValueError as error:
        raise ValueError(f"Not a bitrate: {bitrate!r}") from error
    if value <= 0:
        raise ValueError(f"Not a bitrate: {bitrate!r}")
    return f"{value}k"

#: Silence inserted between chapters in the concatenated file, in seconds.
DEFAULT_CHAPTER_GAP_SEC = 1.5

_WRITE_BLOCK_FRAMES = 1 << 16


@dataclass
class Chapter:
    """A source file plus where it lands in the finished book."""

    path: Path
    title: str
    start_sec: float = 0.0
    duration_sec: float = 0.0

    @property
    def end_sec(self) -> float:
        return self.start_sec + self.duration_sec


@dataclass
class AssemblyResult:
    """What assembly produced, and what is left to do by hand."""

    wav_path: Path
    metadata_path: Optional[Path]
    output_path: Optional[Path]
    chapters: List[Chapter] = field(default_factory=list)
    duration_sec: float = 0.0
    sample_rate: int = 0
    #: Set when ffmpeg was unavailable or failed — run this to finish the job.
    pending_command: Optional[List[str]] = None
    message: str = ""


def find_ffmpeg() -> Optional[str]:
    """Path to an ffmpeg binary, or None if it is not installed."""
    return shutil.which("ffmpeg")


def _title_from_path(path: Path, index: int) -> str:
    """Readable chapter title from a filename like ``chapitre_003.wav``."""
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    return stem[:1].upper() + stem[1:] if stem else f"Chapitre {index}"


def concat_chapters(
    chapter_paths: Sequence[str | Path],
    out_wav: str | Path,
    *,
    gap_sec: float = DEFAULT_CHAPTER_GAP_SEC,
    titles: Optional[Sequence[str]] = None,
    subtype: str = "PCM_16",
) -> tuple[List[Chapter], int]:
    """Stream chapter WAVs into a single file and record chapter boundaries.

    Streaming rather than concatenating arrays keeps memory flat regardless of
    book length. Returns the chapters with their start times filled in, and the
    sample rate.
    """
    paths = [Path(p) for p in chapter_paths]
    if not paths:
        raise ValueError("No chapter files to assemble.")
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Chapter file(s) not found: {', '.join(str(p) for p in missing)}")

    infos = [sf.info(str(p)) for p in paths]
    rates = {info.samplerate for info in infos}
    if len(rates) > 1:
        raise ValueError(
            f"Chapters have different sample rates ({sorted(rates)}); "
            "they must all be generated with the same model settings."
        )
    sample_rate = infos[0].samplerate

    out_path = Path(out_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gap_frames = max(0, int(round(sample_rate * max(0.0, gap_sec))))

    chapters: List[Chapter] = []
    cursor = 0  # frames written so far
    with sf.SoundFile(
        str(out_path), mode="w", samplerate=sample_rate, channels=1, subtype=subtype
    ) as sink:
        import numpy as np  # local: only needed for the inter-chapter silence

        gap = np.zeros(gap_frames, dtype="float32")
        for index, (path, info) in enumerate(zip(paths, infos), start=1):
            if index > 1 and gap_frames:
                sink.write(gap)
                cursor += gap_frames
            start_frames = cursor
            with sf.SoundFile(str(path)) as source:
                while True:
                    block = source.read(_WRITE_BLOCK_FRAMES, dtype="float32", always_2d=False)
                    if not len(block):
                        break
                    if block.ndim > 1:
                        block = block.mean(axis=1)
                    sink.write(block)
                    cursor += len(block)
            title = titles[index - 1] if titles and index <= len(titles) else _title_from_path(path, index)
            chapters.append(
                Chapter(
                    path=path,
                    title=title,
                    start_sec=start_frames / sample_rate,
                    duration_sec=(cursor - start_frames) / sample_rate,
                )
            )
    return chapters, sample_rate


def build_ffmetadata(
    chapters: Sequence[Chapter],
    *,
    title: str = "",
    author: str = "",
    album: str = "",
    year: str = "",
    genre: str = "Audiobook",
) -> str:
    """Render an ffmpeg FFMETADATA document carrying the chapter markers."""
    lines = [";FFMETADATA1"]
    for key, value in (
        ("title", title),
        ("artist", author),
        ("album", album or title),
        ("date", year),
        ("genre", genre),
    ):
        if value:
            lines.append(f"{key}={_escape_metadata(value)}")

    # Each marker runs up to the next chapter's start rather than to the end of
    # its own audio, so the inter-chapter silence belongs to the chapter that
    # precedes it. Leaving those gaps unclaimed makes players show "no chapter"
    # while the book is still playing.
    for index, chapter in enumerate(chapters):
        is_last = index == len(chapters) - 1
        end_sec = chapter.end_sec if is_last else chapters[index + 1].start_sec
        lines += [
            "",
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={int(round(chapter.start_sec * 1000))}",
            f"END={int(round(end_sec * 1000))}",
            f"title={_escape_metadata(chapter.title)}",
        ]
    return "\n".join(lines) + "\n"


def _escape_metadata(value: str) -> str:
    """FFMETADATA treats ``= ; # \\`` and newlines as syntax."""
    for char in ("\\", "=", ";", "#"):
        value = value.replace(char, f"\\{char}")
    return value.replace("\n", " ").strip()


def ffmpeg_command(
    wav_path: str | Path,
    metadata_path: str | Path,
    out_path: str | Path,
    *,
    cover_path: Optional[str | Path] = None,
    bitrate: str | int | None = None,
) -> List[str]:
    """The ffmpeg invocation that turns the WAV into the final chaptered file."""
    out = Path(out_path)
    encoder = _ENCODERS.get(out.suffix.lstrip(".").lower(), _ENCODERS["m4b"])
    chosen = normalize_bitrate(bitrate) or encoder["bitrate"]

    command = ["ffmpeg", "-y", "-i", str(wav_path), "-i", str(metadata_path)]
    if cover_path:
        command += ["-i", str(cover_path)]
    command += ["-map", "0:a", "-map_metadata", "1"]
    if cover_path:
        command += ["-map", "2:v", "-disposition:v", "attached_pic", "-c:v", "copy"]
    command += ["-c:a", encoder["codec"], "-b:a", chosen, "-ac", "1"]
    if out.suffix.lower() in (".m4b", ".m4a"):
        command += ["-movflags", "+faststart"]
    command.append(str(out))
    return command


def assemble(
    chapter_paths: Sequence[str | Path],
    out_path: str | Path,
    *,
    title: str = "",
    author: str = "",
    titles: Optional[Sequence[str]] = None,
    gap_sec: float = DEFAULT_CHAPTER_GAP_SEC,
    cover_path: Optional[str | Path] = None,
    keep_wav: bool = True,
    bitrate: str | int | None = None,
) -> AssemblyResult:
    """Build a single chaptered audiobook from per-chapter WAVs.

    The concatenated WAV and the chapter-marker file are written first and
    unconditionally, so that an absent or failing ffmpeg costs only the encoding
    step — never the synthesis.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wav_path = out.with_suffix(".wav") if out.suffix.lower() != ".wav" else out

    chapters, sample_rate = concat_chapters(
        chapter_paths, wav_path, gap_sec=gap_sec, titles=titles
    )
    duration = chapters[-1].end_sec if chapters else 0.0

    metadata_path = out.with_suffix(".chapters.txt")
    metadata_path.write_text(
        build_ffmetadata(chapters, title=title or out.stem, author=author),
        encoding="utf-8",
    )

    result = AssemblyResult(
        wav_path=wav_path,
        metadata_path=metadata_path,
        output_path=None,
        chapters=chapters,
        duration_sec=duration,
        sample_rate=sample_rate,
    )

    if wav_path == out:
        result.output_path = out
        result.message = "Assembled to WAV (chapter markers written alongside)."
        return result

    command = ffmpeg_command(
        wav_path, metadata_path, out, cover_path=cover_path, bitrate=bitrate
    )
    if not find_ffmpeg():
        result.pending_command = command
        result.message = (
            "ffmpeg not found — the concatenated WAV and chapter markers are ready. "
            "Install ffmpeg and run the reported command to produce "
            f"{out.name} with chapter markers."
        )
        return result

    # ffmpeg echoes the chapter titles back on stderr, so its output carries
    # whatever the book is called. `text=True` alone decodes with the locale's
    # preferred encoding, and a server shell without LANG resolves that to
    # ASCII — so a French title raises UnicodeDecodeError and loses a book that
    # was already fully narrated. Name the encoding rather than inherit it.
    completed = subprocess.run(command, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        result.pending_command = command
        tail = (completed.stderr or "").strip().splitlines()[-3:]
        result.message = "ffmpeg failed: " + " / ".join(tail)
        return result

    result.output_path = out
    result.message = f"Wrote {out.name} with {len(chapters)} chapter marker(s)."
    if not keep_wav and wav_path.is_file():
        wav_path.unlink(missing_ok=True)
        result.wav_path = out
    return result
