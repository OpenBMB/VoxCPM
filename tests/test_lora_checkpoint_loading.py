from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap_repo_modules(monkeypatch):
    for name, path in [
        ("voxcpm", SRC / "voxcpm"),
        ("voxcpm.model", SRC / "voxcpm" / "model"),
        ("voxcpm.modules", SRC / "voxcpm" / "modules"),
    ]:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, pkg)

    hh = types.ModuleType("huggingface_hub")
    hh.snapshot_download = lambda *a, **k: "/tmp/fake"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hh)

    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        @classmethod
        def model_rebuild(cls):
            return None

        @classmethod
        def model_validate_json(cls, s):
            return cls()

        def model_dump(self):
            return {}

    pydantic.BaseModel = BaseModel
    monkeypatch.setitem(sys.modules, "pydantic", pydantic)

    torchaudio = types.ModuleType("torchaudio")
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)

    librosa = types.ModuleType("librosa")
    librosa.effects = types.SimpleNamespace(trim=lambda *a, **k: (None, (0, 0)))
    monkeypatch.setitem(sys.modules, "librosa", librosa)

    einops = types.ModuleType("einops")
    einops.rearrange = lambda x, *a, **k: x
    monkeypatch.setitem(sys.modules, "einops", einops)

    tqdm_pkg = types.ModuleType("tqdm")
    tqdm_pkg.__path__ = ["/nonexistent"]
    tqdm_pkg.tqdm = lambda x, *a, **k: x
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_pkg)

    tqdm_auto = types.ModuleType("tqdm.auto")
    tqdm_auto.tqdm = lambda x, *a, **k: x
    monkeypatch.setitem(sys.modules, "tqdm.auto", tqdm_auto)

    transformers = types.ModuleType("transformers")

    class LlamaTokenizerFast:
        pass

    class PreTrainedTokenizer:
        pass

    transformers.LlamaTokenizerFast = LlamaTokenizerFast
    transformers.PreTrainedTokenizer = PreTrainedTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    internal_mods = {
        "voxcpm.modules.audiovae": ["AudioVAE", "AudioVAEConfig", "AudioVAEV2", "AudioVAEConfigV2"],
        "voxcpm.modules.layers": ["ScalarQuantizationLayer"],
        "voxcpm.modules.locdit": ["CfmConfig", "UnifiedCFM", "VoxCPMLocDiT", "VoxCPMLocDiTV2"],
        "voxcpm.modules.locenc": ["VoxCPMLocEnc"],
        "voxcpm.modules.minicpm4": ["MiniCPM4Config", "MiniCPMModel"],
        "voxcpm.modules.layers.lora": ["apply_lora_to_named_linear_modules", "LoRALinear"],
    }
    for modname, names in internal_mods.items():
        module = types.ModuleType(modname)
        for name in names:
            if name == "apply_lora_to_named_linear_modules":
                setattr(module, name, lambda *a, **k: None)
            else:
                setattr(module, name, type(name, (), {}))
        monkeypatch.setitem(sys.modules, modname, module)

    _load_module("voxcpm.model.utils", SRC / "voxcpm" / "model" / "utils.py")
    voxcpm = _load_module("voxcpm.model.voxcpm", SRC / "voxcpm" / "model" / "voxcpm.py")
    voxcpm2 = _load_module("voxcpm.model.voxcpm2", SRC / "voxcpm" / "model" / "voxcpm2.py")
    return voxcpm.VoxCPMModel, voxcpm2.VoxCPM2Model


class DummyModel:
    device = "cpu"

    def named_parameters(self):
        return []


@pytest.mark.parametrize("module_name", ["v1", "v2"])
def test_load_lora_weights_accepts_tensor_only_legacy_checkpoints(monkeypatch, tmp_path, module_name):
    VoxCPMModel, VoxCPM2Model = bootstrap_repo_modules(monkeypatch)
    cls = VoxCPMModel if module_name == "v1" else VoxCPM2Model

    ckpt_path = tmp_path / "lora_weights.ckpt"
    torch.save({"state_dict": {"fake": torch.zeros(1)}}, ckpt_path)

    loaded, skipped = cls.load_lora_weights(DummyModel(), str(ckpt_path), device="cpu")

    assert loaded == []
    assert skipped == ["fake"]


@pytest.mark.parametrize("module_name", ["v1", "v2"])
def test_load_lora_weights_rejects_malicious_pickle_payloads(monkeypatch, tmp_path, module_name):
    VoxCPMModel, VoxCPM2Model = bootstrap_repo_modules(monkeypatch)
    cls = VoxCPMModel if module_name == "v1" else VoxCPM2Model

    ckpt_path = tmp_path / "lora_weights.ckpt"
    marker_path = tmp_path / f"{module_name}-marker.txt"

    class Exploit:
        def __reduce__(self):
            import pathlib

            return (pathlib.Path.write_text, (marker_path, f"{module_name} executed\n"))

    torch.save({"state_dict": {"fake": torch.zeros(1)}, "boom": Exploit()}, ckpt_path)

    with pytest.raises(Exception, match="Weights only load failed"):
        cls.load_lora_weights(DummyModel(), str(ckpt_path), device="cpu")

    assert not marker_path.exists()
