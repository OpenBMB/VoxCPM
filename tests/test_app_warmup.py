"""Regression test for issue #367: the web demo must warm up the model at
server startup, not lazily on the first request.

Loading the multi-GB model inside the first click blocks that request for
minutes and surfaces any load/download failure as an opaque connection error.
``run_demo`` must therefore call ``get_or_load_voxcpm()`` *before* the Gradio
server is launched. This test records the call order and fails if the warm-up
is removed.

Heavy deps (gradio/funasr/voxcpm/torch) are not importable in the test env, so
they are stubbed before importing ``app`` — same approach as test_cli.py.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"


def _install_stub_deps() -> None:
    # --- fake gradio (only the symbols app.py touches at import time) ---
    gr = types.ModuleType("gradio")

    def _passthrough(*args, **kwargs):
        return mock.MagicMock()

    gr.I18n = _passthrough
    themes = types.SimpleNamespace(Soft=_passthrough, GoogleFont=_passthrough)
    gr.themes = themes
    sys.modules["gradio"] = gr

    # --- fake funasr ---
    funasr = types.ModuleType("funasr")
    funasr.AutoModel = mock.MagicMock()
    sys.modules["funasr"] = funasr

    # --- fake voxcpm + voxcpm.model.utils ---
    voxcpm = types.ModuleType("voxcpm")
    voxcpm.VoxCPM = mock.MagicMock()
    model_pkg = types.ModuleType("voxcpm.model")
    utils = types.ModuleType("voxcpm.model.utils")
    utils.resolve_runtime_device = lambda device, fallback: "cpu"
    model_pkg.utils = utils
    voxcpm.model = model_pkg
    sys.modules["voxcpm"] = voxcpm
    sys.modules["voxcpm.model"] = model_pkg
    sys.modules["voxcpm.model.utils"] = utils


def _load_app_module():
    _install_stub_deps()
    spec = importlib.util.spec_from_file_location("voxcpm_app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_is_warmed_up_before_launch():
    app = _load_app_module()

    calls: list[str] = []

    class _FakeInterface:
        def queue(self, *args, **kwargs):
            return self

        def launch(self, *args, **kwargs):
            calls.append("launch")

    with mock.patch.object(app, "create_demo_interface", return_value=_FakeInterface()), \
         mock.patch.object(app.VoxCPMDemo, "get_or_load_voxcpm",
                           autospec=True,
                           side_effect=lambda self: calls.append("warmup")):
        app.run_demo(device="cpu")

    print(f"call order: {calls}")
    assert "warmup" in calls, "model was never warmed up (loads lazily on first request)"
    assert calls.index("warmup") < calls.index("launch"), (
        f"model must load before launch, got order: {calls}"
    )
