"""Tests for preparing the folder a distributor accepts.

No ffmpeg is involved: what is under test is the audio work (where a file is
cut, how long a sample runs) and the exact command that would be handed to the
encoder — which is what has to be right, since the encode itself may happen on
another machine entirely.
"""
import numpy as np
import pytest

from narration import audio, delivery

SR = 22050  # deliberately not 44100: the profile has to resample, not assume


def speech(seconds: float, level: float = 0.2) -> np.ndarray:
    """Signal that reads as speech to the level and pause detectors."""
    samples = max(1, int(SR * seconds))
    rng = np.random.default_rng(int(seconds * 1000) % (2**32))
    syllables = max(2, int(seconds * 5))
    envelope = np.interp(
        np.linspace(0.0, syllables, samples),
        np.arange(syllables + 1),
        rng.uniform(0.4, 1.0, syllables + 1),
    )
    body = (rng.normal(0.0, 1.0, samples) * envelope).astype(np.float32)
    return body * (level / max(float(np.max(np.abs(body))), 1e-9))


def with_pause_at(seconds_before: float, pause: float, seconds_after: float) -> np.ndarray:
    """Speech, a clear pause, then more speech — a cut point that is obvious."""
    return np.concatenate(
        [speech(seconds_before), audio.silence(SR, pause), speech(seconds_after)]
    )


class TestLimits:
    def test_the_binding_limit_is_computed_not_assumed(self):
        """At 192 kbps CBR, 170 MB runs out before the 120-minute clock does."""
        limit = delivery.max_seconds_for(delivery.ACX_PROFILE)
        assert limit < audio.ACX_MAX_FILE_SEC
        assert limit == pytest.approx(
            delivery.ACX_MAX_FILE_BYTES / delivery.ACX_PROFILE.bytes_per_second
        )

    def test_the_profile_is_the_acx_specification(self):
        profile = delivery.ACX_PROFILE
        assert profile.bitrate_kbps >= 192
        assert profile.sample_rate == 44100
        assert profile.suffix == ".mp3"


class TestSplitting:
    def test_a_chapter_that_fits_is_returned_untouched(self):
        chapter = speech(3.0)
        parts = delivery.split_for_delivery(chapter, SR)
        assert len(parts) == 1
        assert np.array_equal(parts[0], chapter)

    def test_an_over_long_chapter_is_cut_into_parts(self):
        chapter = speech(10.0)
        parts = delivery.split_for_delivery(chapter, SR, max_seconds=4.0)
        assert len(parts) >= 3
        assert all(part.size / SR <= 4.0 for part in parts)

    def test_nothing_is_lost_in_the_split(self):
        """Every second of narration has to survive into some part."""
        chapter = speech(10.0)
        parts = delivery.split_for_delivery(chapter, SR, max_seconds=4.0)
        rooms = audio.MasteringSettings()
        room_per_part = (rooms.lead_sec + rooms.tail_sec) * len(parts)
        total = sum(part.size for part in parts) / SR
        assert total >= 10.0 - 1.0
        assert total <= 10.0 + room_per_part + 1.0

    def test_the_cut_lands_in_the_pause(self):
        """Cutting mid-word is the failure this is written to prevent."""
        chapter = with_pause_at(3.0, 1.0, 3.0)
        rooms = audio.MasteringSettings()
        # A limit leaving a 4-second budget, so the 7-second chapter must be
        # split and the pause at 3 s falls inside the backwards search window.
        parts = delivery.split_for_delivery(
            chapter, SR, max_seconds=4.0 + rooms.lead_sec + rooms.tail_sec
        )
        assert len(parts) == 2
        spoken = audio.trim_silence(parts[0], SR).size / SR
        assert 2.5 <= spoken <= 4.3, spoken

    def test_every_part_is_shaped_like_a_delivered_file(self):
        parts = delivery.split_for_delivery(speech(10.0), SR, max_seconds=4.0)
        for part in parts:
            head, tail = audio.room_tone_sec(part, SR)
            assert head >= audio.ACX_HEAD_ROOM_MIN_SEC
            assert tail >= audio.ACX_TAIL_ROOM_MIN_SEC

    def test_empty_audio_does_not_explode(self):
        parts = delivery.split_for_delivery(np.zeros(0, dtype=np.float32), SR)
        assert len(parts) == 1
        assert parts[0].size == 0

    def test_a_limit_shorter_than_its_own_room_tone_keeps_the_audio_whole(self):
        """Better one over-long file than a pile of files made of silence."""
        chapter = speech(10.0)
        parts = delivery.split_for_delivery(chapter, SR, max_seconds=1.0)
        assert len(parts) == 1
        assert np.array_equal(parts[0], chapter)


class TestRetailSample:
    def test_it_lands_inside_the_one_to_five_minute_window(self):
        sample = delivery.retail_sample(speech(400.0), SR, target_seconds=120.0)
        duration = sample.size / SR
        assert delivery.SAMPLE_MIN_SEC <= duration <= delivery.SAMPLE_MAX_SEC

    def test_a_target_outside_the_window_is_pulled_back_into_it(self):
        too_long = delivery.retail_sample(speech(700.0), SR, target_seconds=600.0)
        assert too_long.size / SR <= delivery.SAMPLE_MAX_SEC + 3.0

    def test_a_short_chapter_gives_a_short_sample_rather_than_an_error(self):
        sample = delivery.retail_sample(speech(20.0), SR)
        assert sample.size > 0
        assert sample.size / SR < 30.0

    def test_it_stops_at_a_pause(self):
        """It stops at the last pause before the target, not on the target."""
        chapter = with_pause_at(55.0, 1.5, 60.0)
        sample = delivery.retail_sample(chapter, SR, target_seconds=60.0)
        # Ends in the 55-second pause rather than 5 seconds into the next
        # sentence — and not 10 seconds early either.
        assert 53.0 <= sample.size / SR <= 61.0, sample.size / SR

    def test_it_can_start_further_in(self):
        chapter = speech(300.0)
        late = delivery.retail_sample(chapter, SR, target_seconds=60.0, start_seconds=120.0)
        assert late.size > 0

    def test_it_is_shaped_like_a_delivered_file(self):
        sample = delivery.retail_sample(speech(200.0), SR, target_seconds=90.0)
        head, tail = audio.room_tone_sec(sample, SR)
        assert head >= audio.ACX_HEAD_ROOM_MIN_SEC
        assert tail >= audio.ACX_TAIL_ROOM_MIN_SEC


class TestEncodeCommand:
    def test_it_asks_for_constant_bitrate_mp3(self):
        command = delivery.encode_command("in.wav", "out.mp3")
        assert "libmp3lame" in command
        assert "192k" in command
        # A quality flag would make it variable bitrate, which is rejected.
        assert "-q:a" not in command

    def test_it_resamples_to_the_required_rate(self):
        command = delivery.encode_command("in.wav", "out.mp3")
        assert command[command.index("-ar") + 1] == "44100"

    def test_it_delivers_mono(self):
        command = delivery.encode_command("in.wav", "out.mp3")
        assert command[command.index("-ac") + 1] == "1"

    def test_the_paths_are_where_ffmpeg_expects_them(self):
        command = delivery.encode_command("chapitre.wav", "sortie/001.mp3")
        assert command[command.index("-i") + 1] == "chapitre.wav"
        assert command[-1] == "sortie/001.mp3"


class TestComplianceReport:
    @staticmethod
    def delivered(seconds=5.0):
        settings = audio.MasteringSettings()
        body, _ = audio.normalize_level(speech(seconds), SR, target_rms_db=-20.0)
        return np.concatenate(
            [
                audio.silence(SR, settings.lead_sec),
                body,
                audio.silence(SR, settings.tail_sec),
            ]
        )

    def test_a_well_made_file_passes_everything(self):
        report = delivery.check_delivered(self.delivered(), SR)
        assert report["compliant"], report
        assert delivery.failures(report) == []

    def test_size_is_predicted_before_the_encode(self):
        report = delivery.check_delivered(self.delivered(60.0), SR)
        assert report["encoded_estimated"] is True
        assert report["encoded_bytes"] == pytest.approx(
            report["duration_sec"] * delivery.ACX_PROFILE.bytes_per_second, rel=0.01
        )

    def test_a_measured_size_replaces_the_estimate(self):
        report = delivery.check_delivered(self.delivered(), SR, encoded_bytes=1234)
        assert report["encoded_estimated"] is False
        assert report["encoded_bytes"] == 1234

    def test_an_oversized_file_is_refused(self):
        report = delivery.check_delivered(
            self.delivered(), SR, encoded_bytes=delivery.ACX_MAX_FILE_BYTES + 1
        )
        assert not report["size_ok"]
        assert not report["compliant"]
        assert any("170 Mo" in reason for reason in delivery.failures(report))

    def test_failures_are_named_in_french_for_the_operator(self):
        report = delivery.check_delivered(speech(2.0), SR)  # no room tone at all
        reasons = delivery.failures(report)
        assert reasons
        assert any("silence de tête" in reason for reason in reasons)
