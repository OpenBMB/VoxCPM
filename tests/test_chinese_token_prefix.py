from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

torch = types.ModuleType("torch")
torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
torch.cuda = types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda seed: None)
sys.modules["torch"] = torch

transformers = types.ModuleType("transformers")
transformers.PreTrainedTokenizer = object
sys.modules["transformers"] = transformers

spec = importlib.util.spec_from_file_location(
    "voxcpm.model.utils",
    ROOT / "src" / "voxcpm" / "model" / "utils.py",
)
utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(utils)


class _Tokenizer:
    def __init__(self, vocab: dict[str, int], pieces: list[str]):
        self.vocab = vocab
        self.pieces = pieces

    def tokenize(self, text: str, **kwargs) -> list[str]:
        return list(self.pieces)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.vocab[token] for token in tokens]


def test_sentencepiece_prefix_is_split_into_characters():
    tokenizer = _Tokenizer(
        {"▁你好": 7, "你": 1, "好": 2},
        ["▁你好"],
    )
    wrapped = utils.mask_multichar_chinese_tokens(tokenizer)

    assert wrapped.tokenize("你好") == ["你", "好"]
    assert wrapped("你好") == [1, 2]


def test_wordpiece_prefix_is_split_into_characters():
    tokenizer = _Tokenizer(
        {"##你好": 7, "你": 1, "好": 2},
        ["##你好"],
    )
    wrapped = utils.mask_multichar_chinese_tokens(tokenizer)

    assert wrapped.tokenize("你好") == ["你", "好"]


def test_plain_multichar_token_still_splits():
    tokenizer = _Tokenizer({"你好": 7, "你": 1, "好": 2}, ["你好"])
    wrapped = utils.mask_multichar_chinese_tokens(tokenizer)

    assert wrapped.tokenize("你好") == ["你", "好"]


def test_non_chinese_token_is_left_intact():
    tokenizer = _Tokenizer({"hello": 3}, ["hello"])
    wrapped = utils.mask_multichar_chinese_tokens(tokenizer)

    assert wrapped.tokenize("hello") == ["hello"]
