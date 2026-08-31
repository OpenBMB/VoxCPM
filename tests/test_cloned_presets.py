"""Tests for cloned voices offered as presets.

A preset used to be a description and a seed. A cloned voice is a recording
instead, and the two must coexist in the same list: the dropdown does not know
which kind it is showing, and neither should the rest of the pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from narration import voices as catalogue  # noqa: E402
from narration.cache import VoiceSpec  # noqa: E402


@pytest.fixture
def voices_file(tmp_path):
    """Build a catalogue this test owns, next to a recording that exists."""
    recording = tmp_path / "voix.wav"
    sf.write(str(recording), np.zeros(24000, dtype=np.float32), 24000)

    def write(entries):
        path = tmp_path / "preset_voices.json"
        path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
        return catalogue.load_presets(path, tmp_path)

    write.recording = recording
    return write


class TestLoading:
    def test_a_cloned_voice_needs_no_description(self, voices_file):
        voices = voices_file([
            {"name": "Edwin", "reference": "voix.wav", "reference_text": "Bonjour."}
        ])
        assert voices[0]["description"] == ""
        assert voices[0]["reference"].endswith("voix.wav")
        assert voices[0]["reference_text"] == "Bonjour."

    def test_a_relative_path_resolves_against_the_repository(self, voices_file):
        voices = voices_file([{"name": "Edwin", "reference": "voix.wav"}])
        assert Path(voices[0]["reference"]).is_absolute()
        assert Path(voices[0]["reference"]).is_file()

    def test_a_described_voice_is_untouched(self, voices_file):
        voices = voices_file([
            {"name": "Narrateur", "description": "voix grave", "seed": 7}
        ])
        assert voices[0]["description"] == "voix grave"
        assert voices[0]["reference"] == ""

    def test_the_two_kinds_live_side_by_side(self, voices_file):
        voices = voices_file([
            {"name": "Narrateur", "description": "voix grave", "seed": 7},
            {"name": "Edwin", "reference": "voix.wav"},
        ])
        assert [v["name"] for v in voices] == ["Narrateur", "Edwin"]

    def test_a_missing_recording_degrades_instead_of_breaking(self, voices_file, caplog):
        """Silence here would fail minutes into a generation instead."""
        voices = voices_file([
            {"name": "Edwin", "description": "de secours", "reference": "absent.wav"}
        ])
        assert voices[0]["reference"] == ""
        assert voices[0]["description"] == "de secours"


class TestFallingBack:
    """An optional configuration file must never cost the whole voice list."""

    FALLBACK = [{"name": "de secours", "description": "voix grave", "seed": 1}]

    def test_a_missing_catalogue_falls_back(self, tmp_path):
        voices = catalogue.load_presets(tmp_path / "absent.json", tmp_path, self.FALLBACK)
        assert [v["name"] for v in voices] == ["de secours"]

    def test_malformed_json_falls_back(self, tmp_path):
        path = tmp_path / "voices.json"
        path.write_text("{ pas du json", encoding="utf-8")
        assert catalogue.load_presets(path, tmp_path, self.FALLBACK)[0]["name"] == "de secours"

    def test_something_that_is_not_a_list_falls_back(self, tmp_path):
        path = tmp_path / "voices.json"
        path.write_text('{"name": "seul"}', encoding="utf-8")
        assert catalogue.load_presets(path, tmp_path, self.FALLBACK)[0]["name"] == "de secours"

    def test_an_entry_without_a_name_is_dropped_not_fatal(self, tmp_path):
        path = tmp_path / "voices.json"
        path.write_text(
            json.dumps([{"description": "sans nom"}, {"name": "Bonne", "seed": 2}]),
            encoding="utf-8",
        )
        voices = catalogue.load_presets(path, tmp_path, self.FALLBACK)
        assert [v["name"] for v in voices] == ["Bonne"]

    def test_the_fallback_is_copied_not_shared(self, tmp_path):
        """A caller mutating what it got back must not corrupt the built-ins."""
        voices = catalogue.load_presets(tmp_path / "absent.json", tmp_path, self.FALLBACK)
        voices[0]["name"] = "modifié"
        assert self.FALLBACK[0]["name"] == "de secours"


class TestTheCacheFollows:
    def test_two_recordings_do_not_share_an_address(self, tmp_path):
        first, second = tmp_path / "a.wav", tmp_path / "b.wav"
        sf.write(str(first), np.zeros(2400, dtype=np.float32), 24000)
        sf.write(str(second), np.ones(2400, dtype=np.float32) * 0.1, 24000)

        a = VoiceSpec(seed=1, reference=VoiceSpec.hash_reference(first))
        b = VoiceSpec(seed=1, reference=VoiceSpec.hash_reference(second))
        assert a.fingerprint() != b.fingerprint()

    def test_a_cloned_voice_never_collides_with_a_described_one(self, tmp_path):
        recording = tmp_path / "a.wav"
        sf.write(str(recording), np.zeros(2400, dtype=np.float32), 24000)
        described = VoiceSpec(description="voix grave", seed=1)
        cloned = VoiceSpec(
            description="voix grave", seed=1, reference=VoiceSpec.hash_reference(recording)
        )
        assert described.fingerprint() != cloned.fingerprint()


class TestTheRepositoryStaysClean:
    def test_recordings_are_never_committed(self):
        """A public repository plus a voice sample is impersonation waiting."""
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "assets/voices/" in ignored

    def test_the_shipped_preset_points_at_a_relative_path(self):
        """An absolute path would only work on the machine that wrote it."""
        entries = json.loads((ROOT / "conf" / "preset_voices.json").read_text(encoding="utf-8"))
        for entry in entries:
            reference = entry.get("reference", "")
            if reference:
                assert not Path(reference).is_absolute(), entry["name"]
