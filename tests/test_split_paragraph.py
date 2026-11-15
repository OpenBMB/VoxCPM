from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "voxcpm" / "utils" / "text_normalize.py"

for name in ("regex", "inflect", "wetext"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["wetext"].Normalizer = object
sys.modules["inflect"].engine = lambda: None

spec = importlib.util.spec_from_file_location("voxcpm.utils.text_normalize", MODULE_PATH)
text_normalize = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(text_normalize)

split_paragraph = text_normalize.split_paragraph


def _tokens(text: str):
    return text.split()


def test_keeps_text_after_the_last_punctuation():
    parts = split_paragraph("Hello world. This has no period", _tokens, lang="en", token_max_n=80, token_min_n=60, merge_len=1)
    assert any("This has no period" in part for part in parts)
    assert any("Hello world." in part for part in parts)


def test_no_punctuation_still_returns_the_whole_paragraph():
    parts = split_paragraph("Just one line", _tokens, lang="en")
    assert parts == ["Just one line."]


def test_closing_quote_stays_on_the_sentence():
    parts = split_paragraph('He said "hi." Then left.', _tokens, lang="en", token_max_n=80, token_min_n=60, merge_len=1)
    assert any(part.strip().startswith('He said "hi."') or 'hi."' in part for part in parts)


def test_leading_punctuation_does_not_raise():
    parts = split_paragraph('."Hello there.', _tokens, lang="en", token_max_n=80, token_min_n=60, merge_len=1)
    assert parts
    assert any("Hello there." in part for part in parts)
