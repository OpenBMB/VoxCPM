"""
Preset management utilities for the VoxCPM WebUI.

A preset stores a reusable configuration including the reference audio, the
cloning-mode toggle and transcript, the control instruction, the target text,
and the advanced generation parameters. Each preset lives in its own directory
under ``presets/<name>/``.
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional


PRESET_VERSION = "1.0"
PRESETS_DIRNAME = "presets"


def _project_root() -> Path:
    """Return the directory that holds this module (the app root)."""
    return Path(__file__).parent.resolve()


def get_presets_dir() -> Path:
    """Return the directory where presets are stored, creating it if needed."""
    path = _project_root() / PRESETS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_preset_name(name: str) -> str:
    """
    Sanitize a preset name so it can be used as a directory name.

    Strips surrounding whitespace, replaces filesystem-unfriendly characters
    with underscores, and prevents empty or purely-special names.
    """
    name = name.strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("._")
    return name or "untitled"


def list_presets() -> List[str]:
    """Return a sorted list of existing preset names."""
    presets_dir = get_presets_dir()
    if not presets_dir.exists():
        return []
    names = [
        p.name
        for p in presets_dir.iterdir()
        if p.is_dir() and (p / "preset.json").is_file()
    ]
    return sorted(names)


def _preset_dir(name: str) -> Path:
    return get_presets_dir() / safe_preset_name(name)


def _copy_audio(src: Optional[str], dst_dir: Path, dst_name: str) -> Optional[str]:
    """
    Copy an uploaded audio file into the preset directory.

    Returns the relative filename written, or None if *src* is empty/missing.
    """
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    # Keep the original suffix so non-wav reference audio still plays back.
    suffix = src_path.suffix or ".wav"
    dst_name = f"{dst_name}{suffix}"
    dst_path = dst_dir / dst_name
    shutil.copy2(src_path, dst_path)
    return dst_name


def save_preset(
    name: str,
    data: Dict,
    reference_audio: Optional[str] = None,
) -> None:
    """
    Save a preset to disk.

    Parameters
    ----------
    name:
        Preset display name.
    data:
        Dictionary containing all non-audio preset fields. See ``load_preset``
        for the expected structure.
    reference_audio:
        Path to the uploaded reference audio, if any.
    """
    name = safe_preset_name(name)
    preset_dir = _preset_dir(name)
    preset_dir.mkdir(parents=True, exist_ok=True)

    # Copy the reference audio and store the relative filename.
    ref_rel = _copy_audio(reference_audio, preset_dir, "reference")

    payload = {
        "version": PRESET_VERSION,
        **data,
        "reference_audio": ref_rel or "",
    }

    preset_file = preset_dir / "preset.json"
    with open(preset_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_preset(name: str) -> Optional[Dict]:
    """
    Load a preset from disk.

    Returns None if the preset does not exist. Relative audio paths are
    resolved to absolute paths so the UI can load them directly.
    """
    preset_dir = _preset_dir(name)
    preset_file = preset_dir / "preset.json"
    if not preset_file.exists():
        return None

    with open(preset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("reference_audio"):
        data["reference_audio"] = str(preset_dir / data["reference_audio"])

    return data


def delete_preset(name: str) -> bool:
    """
    Delete a preset directory.

    Returns True if the preset existed and was removed, False otherwise.
    """
    preset_dir = _preset_dir(name)
    if not preset_dir.exists():
        return False
    shutil.rmtree(preset_dir)
    return True


def preset_exists(name: str) -> bool:
    """Return whether a preset with the given name already exists."""
    return _preset_dir(name).exists()
