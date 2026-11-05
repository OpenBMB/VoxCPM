from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_core():
    pkg = types.ModuleType("voxcpm")
    pkg.__path__ = [str(ROOT / "src" / "voxcpm")]
    pkg.__package__ = "voxcpm"
    model = types.ModuleType("voxcpm.model")
    model.__path__ = [str(ROOT / "src" / "voxcpm" / "model")]
    model.__package__ = "voxcpm.model"
    v1 = types.ModuleType("voxcpm.model.voxcpm")
    v1.VoxCPMModel = object
    v1.LoRAConfig = object
    v2 = types.ModuleType("voxcpm.model.voxcpm2")
    v2.VoxCPM2Model = object
    utils = types.ModuleType("voxcpm.model.utils")
    utils.next_and_close = lambda gen: None
    hub = types.ModuleType("huggingface_hub")
    hub.calls = []

    def snapshot_download(**kwargs):
        hub.calls.append(kwargs)
        raise AssertionError("snapshot_download should not run for a local directory")

    hub.snapshot_download = snapshot_download
    numpy = types.ModuleType("numpy")
    numpy.ndarray = object
    sys.modules["voxcpm"] = pkg
    sys.modules["voxcpm.model"] = model
    sys.modules["voxcpm.model.voxcpm"] = v1
    sys.modules["voxcpm.model.voxcpm2"] = v2
    sys.modules["voxcpm.model.utils"] = utils
    sys.modules["huggingface_hub"] = hub
    sys.modules["numpy"] = numpy
    spec = importlib.util.spec_from_file_location("voxcpm.core", ROOT / "src" / "voxcpm" / "core.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules["voxcpm.core"] = core
    assert spec.loader is not None
    spec.loader.exec_module(core)
    return core, hub


def test_resolve_model_path_expands_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    model_dir = tmp_path / "VoxCPM2"
    model_dir.mkdir()
    core, hub = _load_core()

    resolved = core.resolve_model_path("~/VoxCPM2")

    assert resolved == os.path.abspath(model_dir)
    assert hub.calls == []
