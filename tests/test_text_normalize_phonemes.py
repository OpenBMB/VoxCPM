from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("wetext")
pytest.importorskip("inflect")
pytest.importorskip("regex")

ROOT = Path(__file__).resolve().parents[1]
TEXT_NORMALIZE_PATH = ROOT / "src" / "voxcpm" / "utils" / "text_normalize.py"

spec = importlib.util.spec_from_file_location("voxcpm_text_normalize", TEXT_NORMALIZE_PATH)
text_normalize = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(text_normalize)


@pytest.fixture(scope="module")
def normalizer():
    return text_normalize.TextNormalizer()


def test_chinese_pinyin_tags_keep_tone_digits(normalizer):
    text = "这几日{zuo1}坊里多了个小客人，今天是2025年"
    assert normalizer.normalize(text) == "这几日{zuo1}坊里多了个小客人，今天是二零二五年"


def test_english_phoneme_tags_keep_stress_digits(normalizer):
    text = "Say {HH AH0 L OW1} to 3 people"
    assert normalizer.normalize(text) == "Say {HH AH0 L OW1} to three people"


def test_consecutive_tags_keep_their_order(normalizer):
    assert normalizer.normalize("{ni3}{hao3}，价格是5元") == "{ni3}{hao3}，价格是五元"


def test_text_without_tags_is_unchanged_by_the_protection(normalizer):
    assert normalizer.normalize("今天是2025年") == "今天是二零二五年"
