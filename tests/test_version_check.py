"""Tests for the version reporting and PyPI update check helpers."""

from __future__ import annotations

import json
import sys
import types
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Stub the package so the module can be imported without the heavy runtime deps.
pkg = types.ModuleType("voxcpm")
pkg.__path__ = [str(ROOT / "src" / "voxcpm")]
sys.modules.setdefault("voxcpm", pkg)

from voxcpm import version_check  # noqa: E402
from voxcpm.version_check import (  # noqa: E402
    STATUS_AHEAD,
    STATUS_DEVELOPMENT,
    STATUS_OUTDATED,
    STATUS_UNKNOWN,
    STATUS_UP_TO_DATE,
    UpdateCheckError,
    check_for_updates,
    compare_versions,
    fetch_latest_version,
    get_installed_version,
    is_development_version,
    normalize_version,
)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_normalize_version():
    assert normalize_version(" v2.0.3 ") == "2.0.3"
    assert normalize_version("") == ""


@pytest.mark.parametrize(
    "value",
    ["0.0.post1+g4b8dd4440", "2.0.3+local", "2.0.0.dev1", "unknown", ""],
)
def test_development_versions(value):
    assert is_development_version(value)


@pytest.mark.parametrize("value", ["2.0.3", "v2.0.3", "2.0.3rc1", "2.0.3.post1"])
def test_release_versions_are_not_development(value):
    assert not is_development_version(value)


def test_compare_versions():
    assert compare_versions("2.0.2", "2.0.3") == STATUS_OUTDATED
    assert compare_versions("2.0.3", "2.0.3") == STATUS_UP_TO_DATE
    assert compare_versions("2.0.4", "2.0.3") == STATUS_AHEAD
    assert compare_versions("0.0.post1+g4b8dd4440", "2.0.3") == STATUS_DEVELOPMENT
    assert compare_versions("not-a-version", "2.0.3") == STATUS_UNKNOWN


def test_get_installed_version(monkeypatch):
    monkeypatch.setattr(version_check, "version", lambda name: "2.0.3")
    assert get_installed_version() == "2.0.3"


def test_get_installed_version_from_source_tree(monkeypatch):
    def missing(name):
        raise version_check.PackageNotFoundError()

    monkeypatch.setattr(version_check, "version", missing)
    assert get_installed_version() == "unknown"


def test_fetch_latest_version(monkeypatch):
    def fake_urlopen(request, timeout):
        assert timeout == 5.0
        return FakeResponse({"info": {"version": "2.0.4"}})

    monkeypatch.setattr(version_check.urllib.request, "urlopen", fake_urlopen)
    assert fetch_latest_version() == "2.0.4"


def test_fetch_latest_version_rejects_missing_version(monkeypatch):
    monkeypatch.setattr(
        version_check.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse({"info": {}}),
    )
    with pytest.raises(UpdateCheckError, match="info.version"):
        fetch_latest_version()


def test_fetch_latest_version_rejects_non_string_version(monkeypatch):
    monkeypatch.setattr(
        version_check.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse({"info": {"version": None}}),
    )
    with pytest.raises(UpdateCheckError, match="info.version"):
        fetch_latest_version()


def test_fetch_latest_version_wraps_network_errors(monkeypatch):
    def fake_urlopen(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(version_check.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(UpdateCheckError, match="could not reach PyPI"):
        fetch_latest_version()


def test_check_for_updates(monkeypatch):
    monkeypatch.setattr(version_check, "get_installed_version", lambda: "2.0.2")
    monkeypatch.setattr(version_check, "fetch_latest_version", lambda timeout=5.0: "2.0.3")
    assert check_for_updates() == {
        "installed": "2.0.2",
        "latest": "2.0.3",
        "status": STATUS_OUTDATED,
        "update_available": True,
    }


def test_check_for_updates_development_build(monkeypatch):
    monkeypatch.setattr(version_check, "get_installed_version", lambda: "0.0.post1+g4b8dd4440")
    monkeypatch.setattr(version_check, "fetch_latest_version", lambda timeout=5.0: "2.0.3")
    result = check_for_updates()
    assert result["status"] == STATUS_DEVELOPMENT
    assert result["update_available"] is False
