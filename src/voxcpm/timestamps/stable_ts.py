from __future__ import annotations

from typing import Any

from .base import TimestampItem, TimestampLevel, TimestampResult

# stable-ts sometimes returns segments with empty or missing word lists.
# Segment start/end/text is still usable timing, so word and char requests
# degrade to that span instead of discarding the region.
_MISSING_WORD_TIMING_WARNING = (
    "No per-word timings were available for one or more segments; "
    "those timestamps were derived from segment start, end, and text."
)


class StableTSAligner:
    def __init__(
        self,
        model_name: str = "base",
        device: str | None = None,
        language: str | None = None,
    ) -> None:
        try:
            import stable_whisper
        except ImportError as exc:
            raise ImportError(
                "stable-ts is required for timestamp alignment. " 'Install with: pip install "voxcpm[timestamps]"'
            ) from exc

        self.model = stable_whisper.load_model(model_name, device=device)
        self.model_name = model_name
        self.device = device
        self.language = language

    def align(
        self,
        *,
        audio_path: str,
        text: str,
        sample_rate: int | None = None,
        level: TimestampLevel = "word",
    ) -> TimestampResult:
        result = self.model.align(audio_path, text, language=self.language)
        items, warning = _extract_timestamp_items(result, level)
        return TimestampResult(
            audio_path=audio_path,
            sample_rate=sample_rate,
            backend="stable-ts",
            level=level,
            text=text,
            items=items,
            warning=warning,
        )


def extract_timestamp_items(result: Any, level: TimestampLevel) -> list[TimestampItem]:
    items, _warning = _extract_timestamp_items(result, level)
    return items


def _extract_timestamp_items(result: Any, level: TimestampLevel) -> tuple[list[TimestampItem], str | None]:
    segments = _get_value(result, "segments", []) or []
    if level == "segment":
        return _items_from_segments(segments, "segment"), None

    words: list[TimestampItem] = []
    fell_back = False
    for segment in segments:
        segment_words = _items_from_words(segment)
        if segment_words:
            words.extend(segment_words)
            continue
        derived = _item_from_segment(segment, "word")
        if derived is None:
            continue
        words.append(derived)
        fell_back = True

    warning = _MISSING_WORD_TIMING_WARNING if fell_back else None
    if level == "char":
        return split_word_items_to_chars(words), warning
    return words, warning


def _items_from_segments(segments: list[Any], level: str) -> list[TimestampItem]:
    items = []
    for segment in segments:
        item = _item_from_segment(segment, level)
        if item is not None:
            items.append(item)
    return items


def _item_from_segment(segment: Any, level: str) -> TimestampItem | None:
    text = str(_get_value(segment, "text", "")).strip()
    if not text:
        return None
    return TimestampItem(
        text=text,
        start=float(_get_value(segment, "start", 0.0) or 0.0),
        end=float(_get_value(segment, "end", 0.0) or 0.0),
        level=level,
    )


def _items_from_words(segment: Any) -> list[TimestampItem]:
    words = []
    for word in _get_value(segment, "words", []) or []:
        text = str(_get_value(word, "word", _get_value(word, "text", ""))).strip()
        if not text:
            continue
        words.append(
            TimestampItem(
                text=text,
                start=float(_get_value(word, "start", 0.0) or 0.0),
                end=float(_get_value(word, "end", 0.0) or 0.0),
                level="word",
            )
        )
    return words


def split_word_items_to_chars(words: list[TimestampItem]) -> list[TimestampItem]:
    chars = []
    for word in words:
        text = word.text.strip()
        if not text:
            continue

        duration = max(word.end - word.start, 0.0)
        step = duration / len(text)
        for idx, char in enumerate(text):
            chars.append(
                TimestampItem(
                    text=char,
                    start=word.start + idx * step,
                    end=word.start + (idx + 1) * step,
                    level="char",
                )
            )
    return chars


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
