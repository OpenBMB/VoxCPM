"""Cut prepared text into engine-sized segments, with a pause plan.

The engine cannot synthesize an arbitrarily long passage — it errors out above
roughly 8192 tokens — so long-form text must be segmented no matter what. That
constraint turns out to be an opportunity: the boundary between two segments is
exactly where a narrator would draw breath, so each segment carries how long the
silence after it should be.

A uniform gap between segments is what makes machine narration sound mechanical.
Here the pause follows the punctuation that caused the split: a paragraph break
breathes longer than a full stop, which breathes longer than a comma.

Segments never span a paragraph boundary, which keeps the pacing honest and
gives the resume cache stable keys — reflowing one paragraph does not invalidate
the segments of every paragraph after it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

__all__ = [
    "DEFAULT_MAX_CHARS",
    "PauseProfile",
    "Segment",
    "split_chapters",
    "split_into_segments",
    "split_text_into_chunks",
]

#: Characters per segment. Well under the engine limit: shorter segments also
#: fail less often and cost less to regenerate when one comes out badly.
DEFAULT_MAX_CHARS = 300

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")

# Sentence boundaries, in two parts:
#   * end punctuation followed by whitespace — including when a closing quote or
#     bracket sits between them, as in `Il dit "oui." Puis...`, where a bare
#     lookbehind sees `"` rather than `.` and finds no boundary at all;
#   * any line break, even without surrounding spaces, so hard-wrapped prose,
#     verse and dialogue lines split where they visibly break.
# Each lookbehind alternative is separately fixed-width, which is what Python's
# re module requires.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?:(?<=[.!?…。！？])|(?<=[.!?…。！？][\"'»)\]]))\s+"
    r"|\s*\n\s*"
)
_SENTENCE_END_RE = re.compile(r"[.!?…。！？][\"'»)\]]*$")
_CHAPTER_SPLIT_RE = r"(?m)^\s*---\s*$"


@dataclass(frozen=True)
class PauseProfile:
    """Silence inserted after a segment, by the reason the split happened.

    Values are seconds. The defaults are on the generous side of natural speech
    because listeners forgive a slow narrator far more readily than a breathless
    one, and because audiobooks are usually heard at increased playback speed.
    """

    #: Split inside a sentence — the segment hit the character limit.
    clause: float = 0.25
    #: Segment ends on a full stop, question or exclamation mark.
    sentence: float = 0.45
    #: Segment ends a paragraph.
    paragraph: float = 0.9

    def for_segment(self, text: str, ends_paragraph: bool) -> float:
        if ends_paragraph:
            return self.paragraph
        return self.sentence if _SENTENCE_END_RE.search(text.rstrip()) else self.clause


@dataclass(frozen=True)
class Segment:
    """One unit of synthesis, plus the silence that should follow it."""

    text: str
    pause_after: float
    #: Index of the source paragraph, kept for progress reporting and debugging.
    paragraph: int = 0


#: Where a sentence too long to send whole may be cut, best first. A colon or a
#: semicolon already carries a pause in the reading; a comma carries a lighter
#: one; a dash lighter still. All of them are places a narrator breathes.
_CLAUSE_BOUNDARIES = ("; ", " : ", ", ", " — ", " – ")


def _split_long_sentence(sentence: str, max_chars: int) -> List[str]:
    """Cut an over-long sentence at the places a narrator would breathe.

    This used to hand the sentence over whole, on the reasoning that a cut
    mid-clause is more audible than a slightly long segment. Measurement says
    otherwise: an over-long segment is not read slightly long, it is *truncated*
    by the engine. Across one book, the defect rate was 4-8% below three hundred
    characters, 20% between three and four hundred, and 100% on the single
    679-character segment — which came back as 679 characters in 16.2s where
    34s were needed, i.e. half the sentence simply missing.

    Half a sentence lost is worse than a comma turned into a breath.
    """
    if len(sentence) <= max_chars:
        return [sentence]

    for boundary in _CLAUSE_BOUNDARIES:
        if boundary not in sentence:
            continue
        # Split so the separator stays attached to the clause it closes:
        # `"a, b".split(", ")` would drop the comma, and a comma dropped is a
        # breath the narrator no longer takes.
        parts = re.split(f"({re.escape(boundary)})", sentence)
        tokens = [
            (parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")).strip()
            for i in range(0, len(parts), 2)
        ]
        tokens = [t for t in tokens if t]

        pieces, current = [], ""
        for token in tokens:
            candidate = f"{current} {token}" if current else token
            if current and len(candidate) > max_chars:
                pieces.append(current)
                current = token
            else:
                current = candidate
        if current:
            pieces.append(current)
        # Only accept a boundary that actually solved the problem; a sentence
        # whose commas all sit in the first ten words is not helped by them.
        if pieces and all(len(p) <= max_chars for p in pieces):
            return pieces

    # No usable boundary. Sending it whole loses half of it, so fall back to
    # word boundaries: audible, but every word survives.
    words, pieces, current = sentence.split(), [], ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces or [sentence]


def _pack_sentences(text: str, max_chars: int) -> List[str]:
    """Greedily pack whole sentences into chunks no longer than ``max_chars``.

    A sentence longer than the limit is cut at clause boundaries rather than
    sent whole — see ``_split_long_sentence`` for why that trade was reversed.
    """
    text = (text or "").strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_sentence(sentence, max_chars))
        elif current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        chunks.append(current)
    return chunks


def split_text_into_chunks(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> List[str]:
    """Plain list of segment texts, without the pause plan.

    Kept for callers that only need the segmentation (the single-shot UI path
    and anything written against the original helper in ``app.py``).
    """
    return _pack_sentences(text, max_chars)


#: A segment carrying fewer speakable characters than this is not a sentence —
#: it is debris. Two is the smallest useful sentence in French ("Si.", "Va.")
#: once punctuation is discounted, so anything under three letters or digits
#: is a fragment that arrived from the source rather than from the prose.
_MIN_SPEAKABLE = 3


def _speakable(text: str) -> int:
    return sum(1 for c in text if c.isalnum())


def _absorb_fragments(segments: List[Segment]) -> List[Segment]:
    """Fold debris into its neighbour instead of sending it to the engine.

    An EPUB chapter can end on a stray ``-e``: a hyphenated word cut by the
    file boundary, a stripped tag, a footnote marker. Alone, it is two
    characters, and the engine given two characters does not fall silent — it
    babbles for six times the expected duration, which the quality pass then
    reports as a fatal defect. Regenerating never helps, because the fault is
    the fragment, not the take: measured on one book, a re-roll turned 0.6s of
    noise into 1.6s of it.

    Merging costs nothing — the words are read in the same order either way —
    and it removes the whole class of defect rather than one instance.
    """
    if len(segments) < 2:
        return segments

    out: List[Segment] = []
    for seg in segments:
        if _speakable(seg.text) < _MIN_SPEAKABLE and out:
            previous = out[-1]
            out[-1] = Segment(
                text=f"{previous.text} {seg.text}".strip(),
                # The fragment is now the tail, so the silence that followed it
                # is the silence that follows the whole.
                pause_after=seg.pause_after,
                paragraph=previous.paragraph,
            )
        else:
            out.append(seg)

    # A leading fragment has no predecessor to join; give it its successor.
    if len(out) > 1 and _speakable(out[0].text) < _MIN_SPEAKABLE:
        head, following = out[0], out[1]
        out[1] = Segment(
            text=f"{head.text} {following.text}".strip(),
            pause_after=following.pause_after,
            paragraph=following.paragraph,
        )
        out = out[1:]
    return out


def split_into_segments(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    profile: PauseProfile = PauseProfile(),
) -> List[Segment]:
    """Segment a chapter and decide how long the silence after each part is."""
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    segments: List[Segment] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        chunks = _pack_sentences(paragraph, max_chars)
        for chunk_index, chunk in enumerate(chunks):
            ends_paragraph = chunk_index == len(chunks) - 1
            segments.append(
                Segment(
                    text=chunk,
                    pause_after=profile.for_segment(chunk, ends_paragraph),
                    paragraph=paragraph_index,
                )
            )
    return _absorb_fragments(segments)


def split_chapters(text: str, pattern: Optional[str] = None) -> List[str]:
    """Split a book into chapters on a separator line (``---`` by default).

    Text with no separator at all is a single chapter rather than an error — a
    one-chapter book is a perfectly ordinary thing to narrate.
    """
    parts = re.split(pattern or _CHAPTER_SPLIT_RE, text or "")
    chapters = [p.strip() for p in parts if p and p.strip()]
    return chapters or ([text.strip()] if (text or "").strip() else [])


def total_characters(segments: Sequence[Segment]) -> int:
    """Characters that will actually be sent to the engine."""
    return sum(len(segment.text) for segment in segments)
