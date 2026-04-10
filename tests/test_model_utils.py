from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = ROOT / "src" / "voxcpm" / "model" / "utils.py"

transformers_stub = types.ModuleType("transformers")
transformers_stub.PreTrainedTokenizer = object
sys.modules.setdefault("transformers", transformers_stub)

spec = importlib.util.spec_from_file_location("voxcpm.model.utils", UTILS_PATH)
utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(utils)


def test_resolve_runtime_device_keeps_cpu_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)

    assert utils.resolve_runtime_device("cuda") == "cpu"


def test_resolve_runtime_device_keeps_cuda_when_available(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)

    assert utils.resolve_runtime_device("cuda") == "cuda"
