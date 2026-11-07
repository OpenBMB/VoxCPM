"""Integration tests for the ``torch.compile`` eager-mode fallback.

These lock in the regression from issue #419: when no C compiler is usable,
``VoxCPM2Model.optimize()`` / ``VoxCPMModel.optimize()`` must fall back to
eager mode *without ever calling* ``torch.compile`` (which is lazy and would
only fail later, during the first ``generate()``), and must emit a warning.

Review follow-up for OpenBMB/VoxCPM#440: the tests assert explicitly that
``torch.compile`` is never called and check the warning via ``pytest.warns``
(and the kept stderr line via ``capsys``).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _collect_stub_modules():
    """Remove stub modules other tests inject into ``sys.modules``.

    Several stdlib-only test modules in this directory stub out heavy packages
    (``transformers``, ``voxcpm.*``, ...) at import time so they can run
    without the real dependencies. Those stubs leak into the shared pytest
    session and would shadow the real modules under test here, so remove any
    entry that is not backed by a real module spec. Returns the removed stubs
    so they can be restored if the real import is not possible.
    """
    saved = {}
    for name in list(sys.modules):
        mod = sys.modules[name]
        if getattr(mod, "__spec__", None) is None and getattr(mod, "__file__", None) is None:
            if name == "transformers" or name == "voxcpm" or name.startswith("voxcpm."):
                saved[name] = mod
                del sys.modules[name]
    return saved


_saved_stubs = _collect_stub_modules()

torch = pytest.importorskip("torch")

try:
    import voxcpm.model.voxcpm as voxcpm_module
    import voxcpm.model.voxcpm2 as voxcpm2_module
except ImportError as exc:  # missing optional heavy deps in the test env
    for _name, _mod in _saved_stubs.items():
        sys.modules[_name] = _mod  # leave sibling tests untouched
    pytest.skip(f"voxcpm model modules not importable: {exc}", allow_module_level=True)

# Restore the stubs sibling tests installed: this module keeps direct
# references to the real modules (and patches through them), so the shared
# sys.modules view must be returned exactly as the other tests expect.
for _name, _mod in _saved_stubs.items():
    sys.modules[_name] = _mod

MODELS = [
    pytest.param(voxcpm2_module, voxcpm2_module.VoxCPM2Model, id="VoxCPM2Model"),
    pytest.param(voxcpm_module, voxcpm_module.VoxCPMModel, id="VoxCPMModel"),
]


def _make_instance(cls):
    """Build a bare model instance without running the heavy ``__init__``."""
    obj = object.__new__(cls)
    obj.device = "cuda"
    obj.base_lm = types.SimpleNamespace(forward_step=mock.Mock(name="base_lm.forward_step"))
    obj.residual_lm = types.SimpleNamespace(forward_step=mock.Mock(name="residual_lm.forward_step"))
    obj.feat_encoder = mock.Mock(name="feat_encoder")
    obj.feat_decoder = types.SimpleNamespace(estimator=mock.Mock(name="feat_decoder.estimator"))
    return obj


@pytest.mark.parametrize("module,cls", MODELS)
def test_optimize_falls_back_to_eager_without_calling_torch_compile(
    module, cls, monkeypatch, capsys
):
    """No compiler -> eager fallback: torch.compile is NEVER called, warning emitted."""
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
    monkeypatch.setattr(module, "compiler_available", lambda: False)

    obj = _make_instance(cls)
    base_step = obj.base_lm.forward_step
    residual_step = obj.residual_lm.forward_step

    with mock.patch.object(module.torch, "compile") as compile_mock:
        with pytest.warns(RuntimeWarning, match="torch.compile disabled"):
            result = cls.optimize(obj)

    # The regression from #419: torch.compile must not be touched at all.
    compile_mock.assert_not_called()
    # Eager fallback keeps the original callables and returns the instance.
    assert result is obj
    assert obj.base_lm.forward_step is base_step
    assert obj.residual_lm.forward_step is residual_step
    # The stderr line is kept for CLI visibility.
    err = capsys.readouterr().err
    assert "torch.compile disabled" in err
    assert "no C compiler found" in err


@pytest.mark.parametrize("module,cls", MODELS)
def test_optimize_calls_torch_compile_when_compiler_available(module, cls, monkeypatch):
    """Positive control: with a compiler present, torch.compile IS called.

    Proves the mock wiring in the fallback test is real (i.e. the fallback's
    ``assert_not_called`` would fail if torch.compile were reached).
    """
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
    monkeypatch.setattr(module, "compiler_available", lambda: True)

    obj = _make_instance(cls)
    base_step = obj.base_lm.forward_step
    raw_feat_encoder = obj.feat_encoder
    compiled_marker = object()

    with mock.patch.object(module.torch, "compile", return_value=compiled_marker) as compile_mock:
        result = cls.optimize(obj)

    assert compile_mock.call_count == 4  # base_lm, residual_lm, feat_encoder, estimator
    assert result is obj
    assert obj.base_lm.forward_step is compiled_marker
    assert obj.residual_lm.forward_step is compiled_marker
    assert obj.feat_encoder is compiled_marker
    assert obj.feat_decoder.estimator is compiled_marker
    assert obj._feat_encoder_raw is raw_feat_encoder  # raw encoder saved for fallback use
    assert base_step is not obj.base_lm.forward_step  # replaced by compiled version
