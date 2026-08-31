"""Content-addressed store of generated segments, so a run resumes where it died.

On a CPU-only machine a single chapter takes hours, and any interruption — a
closed laptop, a killed shell, a crash on one bad segment — used to throw away
every segment of that chapter. This cache makes the unit of lost work a single
segment instead of a whole chapter.

The key is a hash of everything that determines the audio: the text itself and
the full voice specification. Two consequences follow, and both are the point:
re-running an unchanged book regenerates nothing, and editing one paragraph
invalidates only the segments of that paragraph.

Writes go through a temporary file and an atomic replace. A half-written WAV
left behind by a process killed mid-write would otherwise be indistinguishable
from a valid cache hit on the next run, and would be silently stitched into the
finished chapter.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

__all__ = ["CacheStats", "ChunkCache", "VoiceSpec"]

#: Bumped when a change to generation would make existing entries wrong.
CACHE_VERSION = 1


@dataclass(frozen=True)
class VoiceSpec:
    """Everything that determines how a piece of text will sound.

    Anything that changes the audio belongs here; anything applied afterwards
    (pauses, trimming, level) deliberately does not, so that re-mastering a book
    does not force it to be re-synthesized.
    """

    description: str = ""
    seed: Optional[int] = None
    cfg: float = 2.0
    steps: int = 10
    normalize: bool = True
    model_id: str = ""
    #: Identifies the reference recording a cloned voice was built from — a
    #: hash of its *contents*, not its path, because the same path can hold a
    #: different take tomorrow and the same take can be moved. Empty for a voice
    #: described in words. It belongs here for the same reason the seed does:
    #: without it, a chapter narrated in a cloned voice would collide in the
    #: cache with the same sentence narrated from a description.
    reference: str = ""
    #: The transcript given alongside that recording, which also changes the
    #: result.
    reference_text: str = ""

    @staticmethod
    def hash_reference(path) -> str:
        """Content hash of a reference recording, or "" when there is none.

        Hashing the bytes rather than the name is what makes the cache honest:
        re-recording into the same filename must not silently reuse the old
        voice, and moving the file must not throw the cache away.
        """
        if not path:
            return ""
        file_path = Path(path)
        if not file_path.is_file():
            return ""
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()[:16]

    def fingerprint(self) -> str:
        payload = {"version": CACHE_VERSION, **asdict(self)}
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    def describe(self) -> str:
        if not self.total:
            return "cache: unused"
        return f"cache: {self.hits}/{self.total} hits, {self.writes} written"


class ChunkCache:
    """A directory of generated segments, addressed by content.

    Set ``enabled=False`` to bypass it entirely without the calling code needing
    to branch on every lookup.
    """

    def __init__(self, root: str | Path, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.stats = CacheStats()
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    # -- addressing --------------------------------------------------------

    def key(self, text: str, voice: VoiceSpec, parent: Optional[str] = None) -> str:
        """Address of a segment.

        ``parent`` chains a segment to the one it was continued from. Under
        continuity mode a segment's audio depends on its predecessor, so without
        that link the cache would serve an entry generated from a different
        starting point.
        """
        digest = hashlib.sha256()
        digest.update(voice.fingerprint().encode("utf-8"))
        digest.update(b"\x00")
        if parent:
            digest.update(parent.encode("utf-8"))
            digest.update(b"\x00")
        digest.update((text or "").encode("utf-8"))
        return digest.hexdigest()[:24]

    def path(self, key: str) -> Path:
        return self.root / f"{key}.wav"

    # -- access ------------------------------------------------------------

    def get(self, key: str) -> Optional[Tuple[int, np.ndarray]]:
        """Return ``(sample_rate, audio)`` for a cached segment, or None."""
        if not self.enabled:
            return None
        target = self.path(key)
        if not target.is_file():
            self.stats.misses += 1
            return None
        try:
            data, sample_rate = sf.read(str(target), dtype="float32", always_2d=False)
        except (RuntimeError, OSError):
            # A corrupt entry is a miss, not a crash: drop it and regenerate.
            target.unlink(missing_ok=True)
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return int(sample_rate), np.asarray(data, dtype=np.float32)

    def attempt_of(self, key: str) -> int:
        """Which re-roll produced the stored entry — 0 when it is the first take.

        Recorded so that repairing the same segment twice gives two different
        takes: the seed of a re-roll is derived from the attempt number, so
        without this the second repair would reproduce the first one exactly.
        """
        sidecar = self.path(key).with_suffix(".json")
        if not sidecar.is_file():
            return 0
        try:
            return int(json.loads(sidecar.read_text(encoding="utf-8")).get("attempt", 0))
        except (OSError, ValueError, TypeError):
            return 0

    def put(
        self,
        key: str,
        sample_rate: int,
        wav: np.ndarray,
        text: str = "",
        attempt: int = 0,
    ) -> Optional[Path]:
        """Store a generated segment. Returns the path, or None when disabled."""
        if not self.enabled:
            return None
        target = self.path(key)
        temporary = target.with_suffix(".wav.tmp")
        try:
            # The temporary name ends in ".tmp", so the container has to be
            # stated explicitly — soundfile otherwise infers it from the suffix.
            sf.write(
                str(temporary),
                np.asarray(wav, dtype=np.float32),
                int(sample_rate),
                format="WAV",
            )
            os.replace(temporary, target)
        except (RuntimeError, OSError):
            temporary.unlink(missing_ok=True)
            return None
        if text or attempt:
            self._write_sidecar(key, text, sample_rate, wav, attempt)
        self.stats.writes += 1
        return target

    def _write_sidecar(
        self, key: str, text: str, sample_rate: int, wav: np.ndarray, attempt: int = 0
    ) -> None:
        """Record what a cache file contains, so the directory stays readable.

        Never fatal: losing a debugging aid must not lose the audio it describes.
        """
        try:
            self.path(key).with_suffix(".json").write_text(
                json.dumps(
                    {
                        "text": text,
                        "duration_sec": round(len(wav) / float(sample_rate or 1), 3),
                        "sample_rate": int(sample_rate),
                        "attempt": int(attempt),
                    },
                    ensure_ascii=False,
                    indent=1,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    # -- maintenance -------------------------------------------------------

    def clear(self) -> int:
        """Delete every entry. Returns how many audio files were removed."""
        if not self.root.is_dir():
            return 0
        removed = 0
        for entry in self.root.iterdir():
            if entry.suffix in (".wav", ".json", ".tmp"):
                try:
                    entry.unlink()
                    if entry.suffix == ".wav":
                        removed += 1
                except OSError:
                    pass
        return removed

    def size_bytes(self) -> int:
        if not self.root.is_dir():
            return 0
        return sum(f.stat().st_size for f in self.root.glob("*.wav") if f.is_file())
