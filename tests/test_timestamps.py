import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("voxcpm")
pkg.__path__ = [str(ROOT / "src" / "voxcpm")]
sys.modules.setdefault("voxcpm", pkg)

from voxcpm.timestamps.base import TimestampItem
from voxcpm.timestamps.stable_ts import split_word_items_to_chars


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
