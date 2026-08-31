"""Fix one bad segment without re-narrating the chapter it sits in.

The quality pass says which segment came out wrong. Acting on that used to mean
regenerating the whole chapter — hours on a CPU to replace three seconds — or
hand-deleting a cache entry whose name is a hash. What was missing is not the
audio, which the cache still holds, but the *recipe*: how the chapter was cut
into segments, and with which voice. That is thrown away when a run ends.

So a narration writes ``plan.json`` beside its chapters. With it, a repair is
cheap and entirely offline except for the one segment being re-rolled: read the
plan, generate that segment again with a fresh derived seed, drop it into the
cache under the same key, and stitch the chapter back together from cache
entries. The other segments are never touched, and never re-synthesized.

Re-rolls are derived, not random, so a repair is reproducible; the attempt
number is remembered in the cache sidecar, so asking twice gives two different
takes rather than the same one again.

Torch-free like the rest of the package: the engine arrives as a callable, which
is what lets the whole repair path be tested in milliseconds.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

from . import audio as audio_tools
from . import cache as cache_tools
from . import quality

__all__ = [
    "PLAN_FILENAME",
    "BookPlan",
    "PlannedChapter",
    "PlannedSegment",
    "RepairResult",
    "chapter_path",
    "flagged_segments",
    "inspect_book",
    "rebuild_chapter",
    "reroll_segment",
    "segment_label",
]

PLAN_FILENAME = "plan.json"
#: Bumped when a plan written by an older version can no longer be read.
PLAN_VERSION = 1


@dataclass(frozen=True)
class PlannedSegment:
    text: str
    pause_after: float = 0.0


@dataclass(frozen=True)
class PlannedChapter:
    index: int
    title: str = ""
    segments: Tuple[PlannedSegment, ...] = ()


@dataclass
class BookPlan:
    """Everything needed to rebuild a book's audio from its cache.

    The voice is stored as the fields of :class:`~narration.cache.VoiceSpec`
    rather than the spec itself, because the cache key is derived from it: a
    plan that could not reproduce the exact key would point at nothing.
    """

    voice: dict = field(default_factory=dict)
    mastering: dict = field(default_factory=dict)
    chapters: Tuple[PlannedChapter, ...] = ()
    version: int = PLAN_VERSION

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "voice": dict(self.voice),
            "mastering": dict(self.mastering),
            "chapters": [
                {
                    "index": chapter.index,
                    "title": chapter.title,
                    "segments": [asdict(segment) for segment in chapter.segments],
                }
                for chapter in self.chapters
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BookPlan":
        version = int(payload.get("version", PLAN_VERSION))
        if version > PLAN_VERSION:
            raise ValueError(
                f"plan.json is version {version}, this build reads up to {PLAN_VERSION}"
            )
        chapters = []
        for entry in payload.get("chapters", []):
            chapters.append(
                PlannedChapter(
                    index=int(entry["index"]),
                    title=entry.get("title", ""),
                    segments=tuple(
                        PlannedSegment(
                            text=segment.get("text", ""),
                            pause_after=float(segment.get("pause_after", 0.0)),
                        )
                        for segment in entry.get("segments", [])
                    ),
                )
            )
        return cls(
            voice=dict(payload.get("voice", {})),
            mastering=dict(payload.get("mastering", {})),
            chapters=tuple(chapters),
            version=version,
        )

    def save(self, outdir: str | Path) -> Path:
        target = Path(outdir) / PLAN_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        # Written whole then moved: a plan truncated by a killed process would
        # make every later repair impossible, which is worse than having none.
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, outdir: str | Path) -> "BookPlan":
        source = Path(outdir)
        if source.is_dir():
            source = source / PLAN_FILENAME
        if not source.is_file():
            raise FileNotFoundError(
                f"No {PLAN_FILENAME} in {Path(outdir)} — that book was narrated "
                "before plans were recorded, or by another tool. Re-run the "
                "narration to write one; cached segments are reused, so it is cheap."
            )
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))

    # -- access ------------------------------------------------------------

    def voice_spec(self) -> cache_tools.VoiceSpec:
        known = {f for f in cache_tools.VoiceSpec.__dataclass_fields__}
        return cache_tools.VoiceSpec(**{k: v for k, v in self.voice.items() if k in known})

    def mastering_settings(self) -> audio_tools.MasteringSettings:
        known = {f for f in audio_tools.MasteringSettings.__dataclass_fields__}
        return audio_tools.MasteringSettings(
            **{k: v for k, v in self.mastering.items() if k in known}
        )

    def chapter(self, index: int) -> PlannedChapter:
        for chapter in self.chapters:
            if chapter.index == index:
                return chapter
        raise KeyError(f"No chapter {index} in this plan")

    def segment(self, chapter_index: int, position: int) -> PlannedSegment:
        """``position`` is 1-based, matching the labels in a quality report."""
        segments = self.chapter(chapter_index).segments
        if not 1 <= position <= len(segments):
            raise KeyError(
                f"Chapter {chapter_index} has {len(segments)} segment(s), asked for {position}"
            )
        return segments[position - 1]


def segment_label(chapter_index: int, position: int) -> str:
    """The identifier used in quality reports and repair menus."""
    return f"ch{chapter_index:03d}/seg{position:03d}"


def parse_label(label: str) -> Tuple[int, int]:
    """Inverse of :func:`segment_label`."""
    try:
        chapter_part, segment_part = label.split("/")
        return int(chapter_part.removeprefix("ch")), int(segment_part.removeprefix("seg"))
    except (ValueError, AttributeError) as error:
        raise ValueError(f"Not a segment label: {label!r}") from error


def chapter_path(outdir: str | Path, index: int) -> Path:
    return Path(outdir) / f"chapitre_{index:03d}.wav"


# --------------------------------------------------------------------------
# Inspecting a finished book
# --------------------------------------------------------------------------


def inspect_book(
    plan: BookPlan,
    cache: cache_tools.ChunkCache,
    thresholds: quality.QualityThresholds = quality.QualityThresholds(),
) -> List[Tuple[str, quality.SegmentReport]]:
    """Re-run the quality checks over every cached segment of a book.

    Reads the cache rather than the finished chapters, because a defect has to
    be located at the segment to be repaired at the segment — and because this
    then works on a book narrated before the quality pass existed.
    """
    spec = plan.voice_spec()
    reports: List[Tuple[str, quality.SegmentReport]] = []
    for chapter in plan.chapters:
        for position, segment in enumerate(chapter.segments, 1):
            entry = cache.get(cache.key(segment.text, spec))
            if entry is None:
                continue
            sample_rate, wav = entry
            reports.append(
                (
                    segment_label(chapter.index, position),
                    quality.inspect_segment(wav, sample_rate, segment.text, thresholds),
                )
            )
    return reports


def flagged_segments(
    reports: Sequence[Tuple[str, quality.SegmentReport]], fatal_only: bool = False
) -> List[Tuple[str, quality.SegmentReport]]:
    """The segments worth a human's attention, worst first."""
    picked = [
        (label, report)
        for label, report in reports
        if (report.fatal if fatal_only else not report.ok)
    ]
    picked.sort(key=lambda pair: (not pair[1].fatal, pair[0]))
    return picked


# --------------------------------------------------------------------------
# Repairing
# --------------------------------------------------------------------------


@dataclass
class RepairResult:
    """Outcome of re-rolling one segment."""

    label: str
    sample_rate: int
    wav: np.ndarray
    report: quality.SegmentReport
    seed: Optional[int]
    attempt: int
    previous: Optional[quality.SegmentReport] = None

    @property
    def improved(self) -> bool:
        """True when the new take is no worse than the one it replaces."""
        if self.previous is None:
            return self.report.ok
        return self.report.penalty <= self.previous.penalty


def reroll_segment(
    plan: BookPlan,
    chapter_index: int,
    position: int,
    cache: cache_tools.ChunkCache,
    render: Callable[[Optional[int]], Tuple[int, np.ndarray]],
    *,
    attempt: Optional[int] = None,
    thresholds: quality.QualityThresholds = quality.QualityThresholds(),
    keep_worse: bool = False,
) -> RepairResult:
    """Generate one segment again and put the new take in the cache.

    The seed is derived from the voice's own seed and an attempt number, so the
    same repair always yields the same audio, while asking again yields a
    different take. When ``attempt`` is not given it continues from whatever the
    cache last recorded.

    A worse take is discarded unless ``keep_worse`` is set: the point of a
    repair is to improve the segment, and a re-roll can come back worse than
    what it replaces.
    """
    segment = plan.segment(chapter_index, position)
    spec = plan.voice_spec()
    key = cache.key(segment.text, spec)
    label = segment_label(chapter_index, position)

    existing = cache.get(key)
    previous_report = (
        quality.inspect_segment(existing[1], existing[0], segment.text, thresholds)
        if existing is not None
        else None
    )

    if attempt is None:
        attempt = cache.attempt_of(key) + 1
    seed = quality.retry_seed(spec.seed, attempt, segment.text)

    sample_rate, wav = render(seed)
    wav = audio_tools.as_float_mono(wav)
    report = quality.inspect_segment(wav, sample_rate, segment.text, thresholds)

    result = RepairResult(
        label=label,
        sample_rate=sample_rate,
        wav=wav,
        report=report,
        seed=seed,
        attempt=attempt,
        previous=previous_report,
    )
    if result.improved or keep_worse:
        cache.put(key, sample_rate, wav, text=segment.text, attempt=attempt)
    return result


@dataclass
class RebuildResult:
    path: Optional[Path]
    sample_rate: int
    duration_sec: float
    missing: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.path is not None and not self.missing


def rebuild_chapter(
    plan: BookPlan,
    chapter_index: int,
    cache: cache_tools.ChunkCache,
    outdir: str | Path,
) -> RebuildResult:
    """Stitch a chapter back together from cached segments and write it.

    Nothing is synthesized here: every segment comes from the cache. A chapter
    with a missing segment is reported rather than written, because a silently
    shortened chapter is far worse than one that failed to rebuild.
    """
    chapter = plan.chapter(chapter_index)
    spec = plan.voice_spec()
    rendered: List[Tuple[np.ndarray, float]] = []
    sample_rate: Optional[int] = None
    missing: List[str] = []

    for position, segment in enumerate(chapter.segments, 1):
        entry = cache.get(cache.key(segment.text, spec))
        if entry is None:
            missing.append(segment_label(chapter_index, position))
            continue
        sample_rate, wav = entry
        rendered.append((wav, segment.pause_after))

    if missing or not rendered or sample_rate is None:
        return RebuildResult(None, sample_rate or 0, 0.0, tuple(missing))

    audio = audio_tools.stitch(rendered, sample_rate, plan.mastering_settings())
    target = chapter_path(outdir, chapter_index)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), audio, sample_rate, subtype="PCM_16")
    return RebuildResult(target, sample_rate, len(audio) / float(sample_rate), ())
