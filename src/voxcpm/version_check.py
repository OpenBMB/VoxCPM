"""Version reporting and optional PyPI update checks for VoxCPM."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from packaging.version import InvalidVersion, Version

PYPI_PROJECT = "voxcpm"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PYPI_PROJECT}/json"
DEFAULT_TIMEOUT = 5.0
USER_AGENT = "voxcpm-version-check (+https://github.com/OpenBMB/VoxCPM)"

STATUS_UP_TO_DATE = "up-to-date"
STATUS_OUTDATED = "outdated"
STATUS_AHEAD = "ahead"
STATUS_DEVELOPMENT = "development"
STATUS_UNKNOWN = "unknown"


class UpdateCheckError(RuntimeError):
    """Raised when the latest release cannot be determined."""


def get_installed_version() -> str:
    """Return the installed ``voxcpm`` version, or ``"unknown"`` from a source tree."""
    try:
        return version(PYPI_PROJECT)
    except PackageNotFoundError:
        return "unknown"


def normalize_version(value: str) -> str:
    """Strip whitespace and a leading ``v`` from a version string."""
    return (value or "").strip().lstrip("vV")


def is_development_version(value: str) -> bool:
    """Return True for local/dev builds that cannot be compared to a release."""
    normalized = normalize_version(value).lower()
    if not normalized or normalized == "unknown":
        return True
    try:
        parsed = Version(normalized)
    except InvalidVersion:
        return False
    return parsed.is_devrelease or parsed.local is not None


def compare_versions(installed: str, latest: str) -> str:
    """Compare two versions and return one of the ``STATUS_*`` constants."""
    installed = normalize_version(installed)
    latest = normalize_version(latest)
    if is_development_version(installed):
        return STATUS_DEVELOPMENT
    try:
        installed_version = Version(installed)
        latest_version = Version(latest)
    except InvalidVersion:
        return STATUS_UNKNOWN
    if installed_version < latest_version:
        return STATUS_OUTDATED
    if installed_version > latest_version:
        return STATUS_AHEAD
    return STATUS_UP_TO_DATE


def fetch_latest_version(timeout: float = DEFAULT_TIMEOUT) -> str:
    """Return the latest ``voxcpm`` version published on PyPI."""
    request = urllib.request.Request(
        PYPI_JSON_URL,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise UpdateCheckError(f"could not reach PyPI: {exc}") from exc

    if not isinstance(payload, dict):
        raise UpdateCheckError("PyPI returned an unexpected response")
    info = payload.get("info")
    if not isinstance(info, dict):
        raise UpdateCheckError("PyPI response did not include an info object")
    raw_version = info.get("version")
    if not isinstance(raw_version, str):
        raise UpdateCheckError("PyPI response did not include info.version")
    latest = normalize_version(raw_version)
    if not latest:
        raise UpdateCheckError("PyPI response did not include info.version")
    return latest


def check_for_updates(timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Check PyPI and return a small machine-readable result."""
    installed = get_installed_version()
    latest = fetch_latest_version(timeout=timeout)
    status = compare_versions(installed, latest)
    return {
        "installed": installed,
        "latest": latest,
        "status": status,
        "update_available": status == STATUS_OUTDATED,
    }
