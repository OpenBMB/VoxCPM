from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.modules.setdefault("argbind", types.ModuleType("argbind"))


def _safe_load(stream):
    text = stream.read() if hasattr(stream, "read") else stream
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and lines[0].lstrip().startswith("- "):
        return [line.split("- ", 1)[1].strip() for line in lines]
    loaded = {}
    for line in lines:
        key, value = line.split(":", 1)
        value = value.strip()
        loaded[key.strip()] = int(value) if value.isdigit() else value
    return loaded


yaml_stub = types.ModuleType("yaml")
yaml_stub.safe_load = _safe_load
sys.modules["yaml"] = yaml_stub

spec = importlib.util.spec_from_file_location(
    "voxcpm.training.config",
    ROOT / "src" / "voxcpm" / "training" / "config.py",
)
config = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(config)


def test_load_yaml_config_expands_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "finetune.yaml").write_text("max_len: 4096\nseed: 7\n", encoding="utf-8")

    loaded = config.load_yaml_config("~/finetune.yaml")

    assert loaded == {"max_len": 4096, "seed": 7}


def test_load_yaml_config_rejects_a_list(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    try:
        config.load_yaml_config(path)
    except ValueError as exc:
        assert "top-level mapping" in str(exc)
    else:
        raise AssertionError("expected ValueError")
