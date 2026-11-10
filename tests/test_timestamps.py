from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("voxcpm")
pkg.__path__ = [str(ROOT / "src" / "voxcpm")]
sys.modules.setdefault("voxcpm", pkg)

from voxcpm.timestamps.base import TimestampItem
from voxcpm.timestamps.stable_ts import extract_timestamp_items, split_word_items_to_chars


def test_split_word_items_to_chars_evenly_distributes_word_duration():
    chars = split_word_items_to_chars([TimestampItem(text="欢迎", start=0.5, end=0.9, level="word")])

    assert [item.text for item in chars] == ["欢", "迎"]
    assert chars[0].start == 0.5
    assert chars[0].end == 0.7
    assert chars[1].start == 0.7
    assert chars[1].end == 0.9
    assert all(item.level == "char" for item in chars)


def test_split_word_items_to_chars_skips_empty_text():
    chars = split_word_items_to_chars(
        [
            TimestampItem(text=" ", start=0.0, end=0.2, level="word"),
            TimestampItem(text="你", start=0.2, end=0.4, level="word"),
        ]
    )

    assert len(chars) == 1
    assert chars[0].text == "你"


def test_extract_timestamp_items_word_level_falls_back_when_words_missing():
    result = {
        "segments": [
            {
                "text": "你好世界",
                "start": 0.0,
                "end": 1.0,
                "words": [
                    {"word": "  ", "start": 0.0, "end": 0.05},
                    {"word": "你好", "start": 0.05, "end": 0.4},
                    {"text": "世界", "start": 0.4, "end": 1.0},
                ],
            },
            {"text": "欢迎", "start": 1.0, "end": 1.4},
            {"text": "使用", "start": 1.4, "end": 1.8, "words": []},
            {"text": "  ", "start": 1.8, "end": 2.0},
        ]
    }

    words = extract_timestamp_items(result, "word")

    assert [(item.text, item.start, item.end, item.level) for item in words] == [
        ("你好", 0.05, 0.4, "word"),
        ("世界", 0.4, 1.0, "word"),
        ("欢迎", 1.0, 1.4, "word"),
        ("使用", 1.4, 1.8, "word"),
    ]


def test_extract_timestamp_items_char_level_splits_segments_without_words():
    result = {
        "segments": [
            {"text": "欢迎", "start": 0.5, "end": 0.9, "words": []},
            {"text": "你", "start": 0.9, "end": 1.1},
        ]
    }

    chars = extract_timestamp_items(result, "char")

    assert [(item.text, item.start, item.end, item.level) for item in chars] == [
        ("欢", 0.5, 0.7, "char"),
        ("迎", 0.7, 0.9, "char"),
        ("你", 0.9, 1.1, "char"),
    ]
