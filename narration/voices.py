"""The catalogue of narration voices, read from configuration.

A voice is answered for in one of two ways, and the rest of the pipeline should
not have to care which:

* **Designed** — a description in words and a seed. The pair is reproducible:
  the same two values always yield the same voice.
* **Cloned** — a recording, and ideally the exact words spoken in it. The
  recording *is* the description, so a cloned entry needs none.

This lives in :mod:`narration` rather than in the interface because it is data,
not user interface: it loads without ``torch`` or ``gradio``, which is what
makes it testable in milliseconds and usable from a script that never opens a
window.

**Recordings are never committed.** A voice sample is the one asset that lets
anyone impersonate its owner, and this fork's repository is public. The entry
pointing at a recording is configuration and belongs in git; the audio it
points at does not, and ``assets/voices/`` is ignored.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

__all__ = ["CUSTOM_LABEL", "load_presets", "reference_of"]

logger = logging.getLogger(__name__)

#: What the dropdown shows for "no preset — I will describe it myself".
CUSTOM_LABEL = "Personnalisé / manuel"


def _entry(item: dict, root: Path) -> Optional[dict]:
    """One catalogue entry, or None when it is not usable.

    A malformed entry is dropped rather than raised on: a typo in an optional
    configuration file must not cost the whole voice list.
    """
    try:
        name = str(item["name"]).strip()
    except (KeyError, TypeError):
        return None
    if not name:
        return None

    voice = {
        "name": name,
        # A cloned voice needs no description: the recording is the description.
        "description": str(item.get("description", "")),
        "seed": int(item.get("seed", 0)),
        "cfg": float(item.get("cfg", 2.0)),
        "diffusion_steps": int(item.get("diffusion_steps", 10)),
        "normalize": bool(item.get("normalize", True)),
        "lang": str(item.get("lang", "fr")),
        "reference": "",
        "reference_text": str(item.get("reference_text", "")),
    }

    reference = str(item.get("reference", "")).strip()
    if reference:
        path = Path(reference)
        resolved = path if path.is_absolute() else root / path
        if resolved.is_file():
            voice["reference"] = str(resolved)
        else:
            # Said at load, not at generation: a preset pointing at a missing
            # recording would otherwise fail minutes into a chapter, and the
            # reason would be nowhere near the symptom.
            logger.warning(
                "Voice %r references a recording that is not there (%s); it will be "
                "used as a described voice instead.", name, resolved,
            )
    return voice


def load_presets(
    path: str | Path,
    root: str | Path,
    fallback: Sequence[dict] = (),
) -> List[dict]:
    """Read the voice catalogue, falling back to ``fallback`` on any problem.

    ``root`` is what a relative ``reference`` is resolved against — the
    repository, so a catalogue committed on one machine works on another.
    """
    config = Path(path)
    root = Path(root)
    if not config.is_file():
        return [dict(voice) for voice in fallback]

    try:
        data = json.loads(config.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("the voice catalogue must be a list")
        voices = [voice for voice in (_entry(item, root) for item in data) if voice]
        if not voices:
            raise ValueError("no usable voice in the catalogue")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        logger.warning("Could not load %s (%s); using the built-in voices.", config, error)
        return [dict(voice) for voice in fallback]

    logger.info("Loaded %d preset voices from %s", len(voices), config)
    return voices


def reference_of(voice: Optional[Dict]) -> tuple:
    """``(recording, transcript)`` for a cloned voice, ``(None, "")`` otherwise."""
    if not voice or not voice.get("reference"):
        return None, ""
    return voice["reference"], voice.get("reference_text", "")
