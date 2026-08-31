"""Tests for the segment cache that makes an interrupted narration resumable."""
import numpy as np
import pytest
import soundfile as sf

from narration.cache import ChunkCache, VoiceSpec

SR = 24000


@pytest.fixture
def voice():
    return VoiceSpec(description="Voix grave", seed=42, cfg=2.0, steps=10, model_id="test")


@pytest.fixture
def cache(tmp_path):
    return ChunkCache(tmp_path / "cache")


def audio_block(value=0.1, length=1000):
    return np.full(length, value, dtype=np.float32)


class TestKeying:
    def test_same_input_same_key(self, cache, voice):
        assert cache.key("Bonjour", voice) == cache.key("Bonjour", voice)

    def test_different_text_different_key(self, cache, voice):
        assert cache.key("Bonjour", voice) != cache.key("Bonsoir", voice)

    @pytest.mark.parametrize(
        "field,value",
        [("seed", 43), ("description", "Autre voix"), ("cfg", 3.0), ("steps", 20),
         ("normalize", False), ("model_id", "other")],
    )
    def test_every_voice_parameter_affects_the_key(self, cache, voice, field, value):
        other = VoiceSpec(**{**voice.__dict__, field: value})
        assert cache.key("Bonjour", voice) != cache.key("Bonjour", other)

    def test_continuity_parent_affects_the_key(self, cache, voice):
        # Under continuity a segment's audio depends on its predecessor, so two
        # identical texts continued from different points must not collide.
        assert cache.key("suite", voice, parent="aaa") != cache.key("suite", voice, parent="bbb")
        assert cache.key("suite", voice, parent="aaa") != cache.key("suite", voice)


class TestRoundTrip:
    def test_miss_then_hit(self, cache, voice):
        key = cache.key("Bonjour", voice)
        assert cache.get(key) is None
        cache.put(key, SR, audio_block())

        result = cache.get(key)
        assert result is not None
        sample_rate, data = result
        assert sample_rate == SR
        assert np.allclose(data, 0.1, atol=1e-4)

    def test_statistics_are_tracked(self, cache, voice):
        key = cache.key("Bonjour", voice)
        cache.get(key)          # miss
        cache.put(key, SR, audio_block())
        cache.get(key)          # hit
        assert (cache.stats.hits, cache.stats.misses, cache.stats.writes) == (1, 1, 1)
        assert "1/2 hits" in cache.stats.describe()

    def test_a_sidecar_records_what_the_entry_contains(self, cache, voice):
        key = cache.key("Bonjour", voice)
        cache.put(key, SR, audio_block(), text="Bonjour")
        sidecar = cache.path(key).with_suffix(".json")
        assert sidecar.is_file()
        assert "Bonjour" in sidecar.read_text(encoding="utf-8")

    def test_survives_a_new_cache_object_on_the_same_directory(self, tmp_path, voice):
        first = ChunkCache(tmp_path / "c")
        key = first.key("Bonjour", voice)
        first.put(key, SR, audio_block())

        second = ChunkCache(tmp_path / "c")
        assert second.get(second.key("Bonjour", voice)) is not None


class TestRobustness:
    def test_a_corrupt_entry_is_a_miss_not_a_crash(self, cache, voice):
        # A process killed mid-write must not poison the next run.
        key = cache.key("Bonjour", voice)
        cache.path(key).write_bytes(b"not a wav file at all")
        assert cache.get(key) is None
        assert not cache.path(key).exists()  # dropped so it will be regenerated

    def test_no_temporary_file_is_left_behind(self, cache, voice):
        key = cache.key("Bonjour", voice)
        cache.put(key, SR, audio_block())
        assert list(cache.root.glob("*.tmp")) == []

    def test_writes_are_atomic_enough_to_be_readable(self, cache, voice):
        key = cache.key("Bonjour", voice)
        path = cache.put(key, SR, audio_block(length=48000))
        assert sf.info(str(path)).frames == 48000

    def test_disabled_cache_stores_and_returns_nothing(self, tmp_path, voice):
        disabled = ChunkCache(tmp_path / "off", enabled=False)
        key = disabled.key("Bonjour", voice)
        assert disabled.put(key, SR, audio_block()) is None
        assert disabled.get(key) is None
        assert not (tmp_path / "off").exists()


class TestMaintenance:
    def test_clear_removes_entries_and_counts_them(self, cache, voice):
        for text in ("un", "deux", "trois"):
            cache.put(cache.key(text, voice), SR, audio_block(), text=text)
        assert cache.clear() == 3
        assert list(cache.root.glob("*.wav")) == []

    def test_size_is_reported(self, cache, voice):
        assert cache.size_bytes() == 0
        cache.put(cache.key("un", voice), SR, audio_block(length=48000))
        assert cache.size_bytes() > 0
