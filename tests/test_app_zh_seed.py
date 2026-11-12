from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    class _Gradio(types.ModuleType):
        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            child = _Gradio(name)
            setattr(self, name, child)
            return child

    gradio = _Gradio("gradio")

    class I18n:
        def __init__(self, **kwargs):
            self.catalogs = kwargs

    gradio.I18n = I18n
    sys.modules["gradio"] = gradio

    funasr = types.ModuleType("funasr")
    funasr.AutoModel = object
    sys.modules["funasr"] = funasr

    numpy = types.ModuleType("numpy")
    numpy.ndarray = object
    sys.modules["numpy"] = numpy

    voxcpm = _Gradio("voxcpm")
    model = _Gradio("voxcpm.model")
    utils = _Gradio("voxcpm.model.utils")
    utils.resolve_runtime_device = lambda *args, **kwargs: "cpu"
    sys.modules["voxcpm"] = voxcpm
    sys.modules["voxcpm.model"] = model
    sys.modules["voxcpm.model.utils"] = utils

    spec = importlib.util.spec_from_file_location("voxcpm_app", ROOT / "app.py")
    app = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(app)
    return app


def test_chinese_seed_controls_are_translated():
    app = _load()
    zh = app._I18N_TRANSLATIONS["zh-CN"]
    en = app._I18N_TRANSLATIONS["en"]
    for key in ("seed_label", "seed_info", "random_seed_label", "random_seed_info"):
        assert zh[key] != en[key]
        assert any("\u4e00" <= char <= "\u9fff" for char in zh[key])
