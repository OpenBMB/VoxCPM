from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import inflect
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "voxcpm" / "utils" / "text_normalize.py"

wetext_stub = types.ModuleType("wetext")


class _Normalizer:
    def __init__(self, *args, **kwargs):
        pass


wetext_stub.Normalizer = _Normalizer
sys.modules.setdefault("wetext", wetext_stub)

spec = importlib.util.spec_from_file_location("voxcpm.utils.text_normalize", MODULE_PATH)
text_normalize = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(text_normalize)

PARSER = inflect.engine()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("section3", "section three"),
        ("3dogs", "three dogs"),
        ("a3b", "a three b"),
        ("H2O", "H two O"),
        ("room21", "room twenty-one"),
        ("section3.", "section three."),
    ],
)
def test_spell_out_number_inserts_word_boundaries(text, expected):
    assert text_normalize.spell_out_number(text, PARSER) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("3rd", "third"),
        ("1st", "first"),
        ("2nd", "second"),
        ("5th", "fifth"),
        ("12th", "twelfth"),
        ("21st", "twenty-first"),
        ("the 3rd", "the third"),
        ("May3rd", "May third"),
        ("3rd place", "third place"),
        ("3rd.", "third."),
        ("3rdogs", "three rdogs"),
        ("3rd4", "third four"),
    ],
)
def test_spell_out_number_spells_bounded_ordinals(text, expected):
    assert text_normalize.spell_out_number(text, PARSER) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4th", "fourth"),
        ("11th", "eleventh"),
        ("There are 3.", "There are three."),
        ("(3)", "(three)"),
        ("section 3", "section three"),
        ("3 dogs", "three dogs"),
        ("3", "three"),
    ],
)
def test_spell_out_number_preserves_spacing_and_punctuation(text, expected):
    assert text_normalize.spell_out_number(text, PARSER) == expected
