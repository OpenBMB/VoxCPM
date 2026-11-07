from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "voxcpm" / "model" / "compile_fallback.py"

spec = importlib.util.spec_from_file_location("voxcpm.model.compile_fallback", MODULE_PATH)
compile_fallback = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(compile_fallback)


def test_compiler_found_via_path(monkeypatch):
    monkeypatch.setattr(compile_fallback.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.delenv("CC", raising=False)
    assert compile_fallback.compiler_available() is True


def test_compiler_found_via_cc_env(monkeypatch):
    monkeypatch.setattr(
        compile_fallback.shutil,
        "which",
        lambda name: "/opt/my-cc" if name == "my-cc" else None,
    )
    monkeypatch.setenv("CC", "my-cc --flag")
    assert compile_fallback.compiler_available() is True


def test_compiler_found_via_triton_knobs_impl(monkeypatch):
    monkeypatch.setattr(compile_fallback.shutil, "which", lambda name: None)
    monkeypatch.delenv("CC", raising=False)
    triton_stub = types.ModuleType("triton")
    knobs = types.SimpleNamespace(build=types.SimpleNamespace(impl="custom-cc -v"))
    triton_stub.knobs = knobs
    monkeypatch.setitem(sys.modules, "triton", triton_stub)
    monkeypatch.setattr(compile_fallback.shutil, "which", lambda name: "/x/custom-cc" if name == "custom-cc" else None)
    assert compile_fallback.compiler_available() is True


def test_no_compiler_anywhere(monkeypatch):
    monkeypatch.setattr(compile_fallback.shutil, "which", lambda name: None)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
    assert compile_fallback.compiler_available() is False


def test_missing_triton_import_is_tolerated(monkeypatch):
    monkeypatch.setattr(compile_fallback.shutil, "which", lambda name: "/usr/bin/gcc" if name == "gcc" else None)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setitem(sys.modules, "triton", None)  # import triton -> ImportError
    assert compile_fallback.compiler_available() is True


@pytest.mark.parametrize("compiler_name", ["cc", "gcc", "clang", "cl", "clang-cl"])
def test_each_compiler_name_is_probed_and_found(monkeypatch, compiler_name):
    """Every supported driver name - POSIX and MSVC - is probed on PATH.

    Regression guard for Windows: ``cl`` (MSVC) and ``clang-cl`` must be part
    of the probe list, otherwise ``torch.compile`` would be silently disabled
    on Windows machines that only have the MSVC toolchain installed.
    """
    probed = []

    def fake_which(name):
        probed.append(name)
        return f"C:/toolchain/{name}.exe" if name == compiler_name else None

    monkeypatch.setattr(compile_fallback.shutil, "which", fake_which)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))  # no knobs
    assert compile_fallback.compiler_available() is True
    assert compiler_name in probed, f"{compiler_name} was never probed via shutil.which"


def test_msvc_toolchain_alone_satisfies_probe(monkeypatch):
    """A Windows box with only MSVC drivers on PATH must still be detected.

    POSIX names are probed first and come up empty; the probe then continues
    into the MSVC names (``any`` may stop at the first hit, e.g. ``cl``, so we
    only require that an MSVC name is what finally satisfies the probe).
    """
    probed = []

    def fake_which_msvc_only(name):
        probed.append(name)
        return f"C:/toolchain/{name}.exe" if name in ("cl", "clang-cl") else None

    monkeypatch.setattr(compile_fallback.shutil, "which", fake_which_msvc_only)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
    assert compile_fallback.compiler_available() is True
    assert probed[:3] == ["cc", "gcc", "clang"]  # POSIX names probed and missed
    assert probed[3] in ("cl", "clang-cl")  # an MSVC name satisfies the probe
