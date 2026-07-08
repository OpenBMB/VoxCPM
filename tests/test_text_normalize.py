from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_NORMALIZE_PATH = ROOT / "src" / "voxcpm" / "utils" / "text_normalize.py"

# Stub heavy/third-party imports so the module loads without them. We only
# exercise ``replace_blank``, which depends on nothing beyond the stdlib.
for _name in ("regex", "inflect"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

_wetext_stub = types.ModuleType("wetext")
_wetext_stub.Normalizer = object
sys.modules.setdefault("wetext", _wetext_stub)

spec = importlib.util.spec_from_file_location("voxcpm.utils.text_normalize", TEXT_NORMALIZE_PATH)
text_normalize = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(text_normalize)

replace_blank = text_normalize.replace_blank


def test_replace_blank_handles_trailing_space():
    # A space at the end of the string has no right-hand neighbour. The old
    # implementation indexed text[i + 1] unconditionally and raised
    # IndexError. The trailing blank should simply be dropped.
    assert replace_blank("hello ") == "hello"
    assert replace_blank("\u4e2d\u6587 ") == "\u4e2d\u6587"
    assert replace_blank("a b ") == "a b"


def test_replace_blank_handles_leading_space():
    # A space at the start has no left-hand neighbour. The old implementation
    # let text[i - 1] wrap around to text[-1] (the last character), which
    # could spuriously keep the leading blank. It should be dropped.
    assert replace_blank(" ab") == "ab"
    assert replace_blank(" a") == "a"


def test_replace_blank_keeps_space_between_ascii():
    # The documented behaviour: keep a blank only when it sits between two
    # ASCII non-space characters.
    assert replace_blank("a b") == "a b"
    assert replace_blank("x 1") == "x 1"
    assert replace_blank("hello world") == "hello world"


def test_replace_blank_drops_space_around_cjk():
    assert replace_blank("\u4e2d \u6587") == "\u4e2d\u6587"
    assert replace_blank("\u4f60\u597d world ok") == "\u4f60\u597dworld ok"


def test_replace_blank_empty_string():
    assert replace_blank("") == ""
