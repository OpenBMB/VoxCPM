from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

torchaudio = types.ModuleType("torchaudio")
sys.modules["torchaudio"] = torchaudio

pipelines = types.ModuleType("modelscope.pipelines")
pipelines.pipeline = lambda *args, **kwargs: (lambda *a, **k: None)
sys.modules["modelscope.pipelines"] = pipelines
utils = types.ModuleType("modelscope.utils")
constant = types.ModuleType("modelscope.utils.constant")
constant.Tasks = types.SimpleNamespace(acoustic_noise_suppression="ans")
sys.modules["modelscope.utils"] = utils
sys.modules["modelscope.utils.constant"] = constant
modelscope = types.ModuleType("modelscope")
modelscope.pipelines = pipelines
modelscope.utils = utils
sys.modules["modelscope"] = modelscope

spec = importlib.util.spec_from_file_location(
    "voxcpm.zipenhancer",
    ROOT / "src" / "voxcpm" / "zipenhancer.py",
)
zipenhancer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(zipenhancer)


def _enhancer():
    enhancer = zipenhancer.ZipEnhancer.__new__(zipenhancer.ZipEnhancer)
    enhancer._pipeline = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    enhancer._normalize_loudness = lambda path: None
    return enhancer


def test_failure_keeps_caller_output(tmp_path):
    source = tmp_path / "in.wav"
    source.write_bytes(b"wav")
    output = tmp_path / "out.wav"
    output.write_bytes(b"keep-me")
    enhancer = _enhancer()

    try:
        enhancer.enhance(str(source), str(output), normalize_loudness=False)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")

    assert output.read_bytes() == b"keep-me"


def test_failure_removes_temp_output(tmp_path):
    source = tmp_path / "in.wav"
    source.write_bytes(b"wav")
    created = []

    real_ntf = zipenhancer.tempfile.NamedTemporaryFile

    def tracking(*args, **kwargs):
        handle = real_ntf(*args, **kwargs)
        created.append(handle.name)
        return handle

    zipenhancer.tempfile.NamedTemporaryFile = tracking
    try:
        enhancer = _enhancer()
        try:
            enhancer.enhance(str(source), None, normalize_loudness=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        zipenhancer.tempfile.NamedTemporaryFile = real_ntf

    assert created
    assert not Path(created[0]).exists()
