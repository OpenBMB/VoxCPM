from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

regex = types.ModuleType("regex")
regex.compile = __import__("re").compile
regex.UNICODE = 0
inflect = types.ModuleType("inflect")
inflect.engine = lambda: None
wetext = types.ModuleType("wetext")


class _Normalizer:
    def __init__(self, *args, **kwargs):
        pass


wetext.Normalizer = _Normalizer
sys.modules["regex"] = regex
sys.modules["inflect"] = inflect
sys.modules["wetext"] = wetext

spec = importlib.util.spec_from_file_location(
    "voxcpm.utils.text_normalize",
    ROOT / "src" / "voxcpm" / "utils" / "text_normalize.py",
)
text_normalize = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(text_normalize)


def test_comparison_is_not_treated_as_html():
    cleaned = text_normalize.clean_markdown("if a < b and c > d")
    assert cleaned == "if a < b and c > d"


def test_real_html_tags_are_removed():
    assert text_normalize.clean_markdown("see <br>next") == "see next"
    assert text_normalize.clean_markdown("hello</div>") == "hello"


def test_bold_markers_are_removed_and_words_stay():
    assert text_normalize.clean_markdown("say **careful** now") == "say careful now"
    assert text_normalize.clean_markdown("2 * 3") == "2 * 3"


def test_bullet_markers_are_removed():
    assert text_normalize.clean_markdown("* item\n- two") == "item\ntwo"


def test_greater_than_is_spoken_in_chinese():
    assert text_normalize.replace_corner_mark("a > b") == "a 大于 b"
    assert text_normalize.replace_corner_mark("a < b") == "a 小于 b"
