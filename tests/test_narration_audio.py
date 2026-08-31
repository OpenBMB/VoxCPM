"""Tests for measurement and mastering.

Levels are checked against analytically known signals: a sine of amplitude ``a``
has RMS ``a/sqrt(2)``, so the expected dBFS figures are exact rather than
recorded from a previous run.
"""
import numpy as np
import pytest

from narration import audio

SR = 24000


def sine(seconds=1.0, amplitude=0.1, freq=220.0, sr=SR):
    t = np.arange(int(sr * seconds)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestMeasurement:
    def test_rms_of_a_known_sine(self):
        # 0.1 amplitude -> RMS 0.0707 -> -23.01 dBFS
        assert audio.speech_rms_db(sine(amplitude=0.1), SR) == pytest.approx(-23.01, abs=0.05)

    def test_peak(self):
        assert audio.peak_db(sine(amplitude=0.5)) == pytest.approx(-6.02, abs=0.05)

    def test_silence_between_speech_does_not_drag_the_measurement_down(self):
        # The whole point of gating: a chapter with generous pauses must not
        # measure quieter than the speech it contains.
        speech = sine(seconds=2.0, amplitude=0.1)
        padded = np.concatenate([speech, audio.silence(SR, 6.0)])
        assert audio.speech_rms_db(padded, SR) == pytest.approx(
            audio.speech_rms_db(speech, SR), abs=0.5
        )

    def test_empty_input_is_not_a_crash(self):
        empty = np.zeros(0, dtype=np.float32)
        assert audio.speech_rms_db(empty, SR) == -np.inf
        assert audio.peak_db(empty) == -np.inf
        assert audio.noise_floor_db(empty, SR) == -np.inf

    def test_noise_floor_of_digital_silence_is_very_low(self):
        signal = np.concatenate([sine(1.0), audio.silence(SR, 1.0)])
        assert audio.noise_floor_db(signal, SR) < audio.ACX_NOISE_FLOOR_DB

    def test_input_shorter_than_one_analysis_frame(self):
        tiny = sine(seconds=0.01)
        assert np.isfinite(audio.speech_rms_db(tiny, SR))


class TestAcxReport:
    @staticmethod
    def delivered(lead=0.75, tail=2.0):
        """A chapter shaped the way it would leave the mastering stage."""
        speech = np.concatenate([sine(2.0), audio.silence(SR, 1.5), sine(2.0)])
        mastered, _ = audio.normalize_level(speech, SR, target_rms_db=-20.0)
        return np.concatenate(
            [audio.silence(SR, lead), mastered, audio.silence(SR, tail)]
        )

    def test_a_correctly_mastered_signal_passes(self):
        report = audio.acx_report(self.delivered(), SR)
        assert report["compliant"], report

    def test_a_too_loud_signal_is_flagged(self):
        report = audio.acx_report(sine(2.0, amplitude=0.99), SR)
        assert not report["rms_ok"]
        assert not report["peak_ok"]
        assert not report["compliant"]

    def test_duration_is_reported(self):
        assert audio.acx_report(sine(3.0), SR)["duration_sec"] == pytest.approx(3.0, abs=0.01)

    def test_a_file_opening_on_its_first_syllable_is_flagged(self):
        """Correct levels, wrong shape — rejected at review all the same."""
        report = audio.acx_report(self.delivered(lead=0.0), SR)
        assert not report["head_room_ok"]
        assert report["rms_ok"]
        assert not report["compliant"]

    def test_a_file_ending_on_its_last_syllable_is_flagged(self):
        report = audio.acx_report(self.delivered(tail=0.1), SR)
        assert not report["tail_room_ok"]
        assert not report["compliant"]

    def test_too_much_room_tone_is_flagged_too(self):
        """The windows have an upper bound: dead air is a defect as well."""
        assert not audio.acx_report(self.delivered(lead=3.0), SR)["head_room_ok"]
        assert not audio.acx_report(self.delivered(tail=9.0), SR)["tail_room_ok"]

    def test_room_tone_is_measured_in_seconds(self):
        head, tail = audio.room_tone_sec(self.delivered(lead=0.75, tail=2.0), SR)
        assert head == pytest.approx(0.75, abs=0.05)
        assert tail == pytest.approx(2.0, abs=0.05)

    def test_silence_only_is_all_head_room(self):
        head, tail = audio.room_tone_sec(audio.silence(SR, 3.0), SR)
        assert head == pytest.approx(3.0, abs=0.01)
        assert tail == 0.0

    def test_mastering_defaults_land_inside_the_acx_windows(self):
        """The defaults must produce a compliant file without being tuned."""
        settings = audio.MasteringSettings()
        assert audio.ACX_HEAD_ROOM_MIN_SEC <= settings.lead_sec <= audio.ACX_HEAD_ROOM_MAX_SEC
        assert audio.ACX_TAIL_ROOM_MIN_SEC <= settings.tail_sec <= audio.ACX_TAIL_ROOM_MAX_SEC


class TestNormalizeLevel:
    def test_reaches_the_target(self):
        normalized, gain = audio.normalize_level(sine(2.0, amplitude=0.02), SR, target_rms_db=-20.0)
        assert audio.speech_rms_db(normalized, SR) == pytest.approx(-20.0, abs=0.1)
        assert gain > 0

    def test_the_peak_ceiling_wins_over_the_rms_target(self):
        # A spiky signal: reaching -20 dBFS RMS would push the spike above 0 dBFS.
        signal = np.concatenate([sine(2.0, amplitude=0.001), np.array([0.9], dtype=np.float32)])
        normalized, _ = audio.normalize_level(
            signal, SR, target_rms_db=-20.0, peak_ceiling_db=-3.0
        )
        assert audio.peak_db(normalized) <= -3.0 + 0.01
        # ...and the result is therefore quieter than the RMS target asked for.
        assert audio.speech_rms_db(normalized, SR) < -20.0

    def test_digital_silence_is_left_alone_rather_than_amplified(self):
        quiet = np.zeros(SR, dtype=np.float32)
        normalized, gain = audio.normalize_level(quiet, SR)
        assert gain == 0.0
        assert not np.any(normalized)

    def test_empty(self):
        normalized, gain = audio.normalize_level(np.zeros(0, dtype=np.float32), SR)
        assert normalized.size == 0 and gain == 0.0


class TestTrimSilence:
    def test_leading_and_trailing_silence_are_removed(self):
        padded = np.concatenate([audio.silence(SR, 0.8), sine(1.0), audio.silence(SR, 0.8)])
        trimmed = audio.trim_silence(padded, SR, keep_ms=60.0)
        # 1s of speech plus the 60 ms margin kept at each edge.
        assert len(trimmed) / SR == pytest.approx(1.12, abs=0.06)

    def test_a_margin_is_kept_so_words_do_not_start_abruptly(self):
        padded = np.concatenate([audio.silence(SR, 0.5), sine(1.0)])
        trimmed = audio.trim_silence(padded, SR, keep_ms=100.0)
        assert len(trimmed) > SR  # strictly longer than the speech alone

    def test_speech_without_silence_is_left_essentially_untouched(self):
        speech = sine(1.0)
        assert len(audio.trim_silence(speech, SR)) == pytest.approx(len(speech), abs=SR * 0.05)

    def test_pure_silence_is_returned_unchanged_rather_than_emptied(self):
        quiet = np.zeros(SR, dtype=np.float32)
        assert len(audio.trim_silence(quiet, SR)) == SR

    def test_threshold_is_relative_so_a_quiet_segment_is_not_erased(self):
        quiet_speech = np.concatenate([audio.silence(SR, 0.5), sine(1.0, amplitude=0.005)])
        trimmed = audio.trim_silence(quiet_speech, SR)
        assert len(trimmed) < len(quiet_speech)
        assert len(trimmed) > SR * 0.9


class TestFadeEdges:
    def test_edges_start_and_end_at_zero(self):
        faded = audio.fade_edges(np.ones(SR, dtype=np.float32), SR, fade_ms=10.0)
        assert faded[0] == pytest.approx(0.0, abs=1e-6)
        assert faded[-1] == pytest.approx(0.0, abs=1e-6)

    def test_the_middle_is_untouched(self):
        source = np.ones(SR, dtype=np.float32)
        faded = audio.fade_edges(source, SR, fade_ms=10.0)
        assert faded[SR // 2] == pytest.approx(1.0)

    def test_the_input_is_not_modified_in_place(self):
        source = np.ones(SR, dtype=np.float32)
        audio.fade_edges(source, SR, fade_ms=10.0)
        assert source[0] == pytest.approx(1.0)

    def test_a_segment_shorter_than_the_fade_is_handled(self):
        short = np.ones(10, dtype=np.float32)
        assert audio.fade_edges(short, SR, fade_ms=100.0).shape == short.shape


class TestStitch:
    def test_pauses_land_between_segments(self):
        settings = audio.MasteringSettings(lead_sec=0.0, tail_sec=0.0, trim_silence=False)
        segments = [(sine(1.0), 0.5), (sine(1.0), 0.9)]
        result = audio.stitch(segments, SR, settings, normalize=False)
        # Two seconds of speech plus the 0.5 s pause; the pause after the LAST
        # segment is the tail's job, not the gap's.
        assert len(result) / SR == pytest.approx(2.5, abs=0.05)

    def test_lead_and_tail_are_added(self):
        settings = audio.MasteringSettings(lead_sec=0.3, tail_sec=0.6, trim_silence=False)
        result = audio.stitch([(sine(1.0), 0.0)], SR, settings, normalize=False)
        assert len(result) / SR == pytest.approx(1.9, abs=0.05)

    def test_joins_do_not_click(self):
        # A step discontinuity at a join shows up as a sample-to-sample jump far
        # larger than anything inside the waveform; the edge fades exist
        # precisely to prevent it. This segment stops mid-cycle, so without the
        # fade the join would drop straight from ~0.4 to silence.
        segment = sine(seconds=0.1013, amplitude=0.5, freq=317.0)
        settings = audio.MasteringSettings(trim_silence=False, lead_sec=0.0, tail_sec=0.0)
        result = audio.stitch([(segment, 0.2)] * 3, SR, settings, normalize=False)

        largest_jump_inside_a_segment = float(np.max(np.abs(np.diff(segment))))
        assert float(np.max(np.abs(np.diff(result)))) <= largest_jump_inside_a_segment * 1.5

    def test_the_finished_chapter_is_normalized_once(self):
        segments = [(sine(1.0, amplitude=0.01), 0.3), (sine(1.0, amplitude=0.01), 0.3)]
        result = audio.stitch(segments, SR, audio.MasteringSettings(target_rms_db=-20.0))
        assert audio.speech_rms_db(result, SR) == pytest.approx(-20.0, abs=0.5)

    def test_normalization_can_be_skipped(self):
        # Polish off as well: the studio chain deliberately changes the level
        # before the normalisation does, and what is under test here is only
        # that the normalisation itself can be skipped.
        loud = sine(1.0, amplitude=0.5)
        result = audio.stitch(
            [(loud, 0.0)],
            SR,
            audio.MasteringSettings(trim_silence=False, polish=False),
            normalize=False,
        )
        assert audio.peak_db(result) == pytest.approx(audio.peak_db(loud), abs=0.1)

    def test_no_segments(self):
        assert audio.stitch([], SR).size == 0


class TestAsFloatMono:
    def test_stereo_is_averaged(self):
        stereo = np.stack([np.ones(100), np.zeros(100)], axis=1).astype(np.float32)
        mono = audio.as_float_mono(stereo)
        assert mono.ndim == 1 and len(mono) == 100
        assert np.allclose(mono, 0.5)

    def test_conversion_does_not_alter_the_level(self):
        # Measurement must report what is in the file, so conversion stays pure.
        offset = np.full(1000, 0.5, dtype=np.float32)
        assert float(np.mean(audio.as_float_mono(offset))) == pytest.approx(0.5)

    def test_integer_input_is_converted(self):
        assert audio.as_float_mono(np.zeros(10, dtype=np.int16)).dtype == np.float32


class TestRemoveDc:
    def test_offset_is_centred(self):
        offset = np.full(1000, 0.5, dtype=np.float32)
        assert float(np.mean(audio.remove_dc(offset))) == pytest.approx(0.0, abs=1e-6)

    def test_a_centred_signal_is_left_alone(self):
        signal = sine(1.0)
        assert np.allclose(audio.remove_dc(signal), signal, atol=1e-6)

    def test_mastering_removes_the_offset(self):
        offset_speech = sine(1.0) + np.float32(0.2)
        mastered = audio.master_segment(offset_speech, SR)
        assert float(np.mean(mastered)) == pytest.approx(0.0, abs=1e-3)

    def test_empty(self):
        assert audio.remove_dc(np.zeros(0, dtype=np.float32)).size == 0


class TestSpeechSeconds:
    """How much of a file is voice — the measure a transcript is compared with."""

    def test_counts_only_the_speech(self):
        signal = np.concatenate([audio.silence(SR, 1.0), sine(3.0), audio.silence(SR, 2.0)])
        assert audio.speech_seconds(signal, SR) == pytest.approx(3.0, abs=0.15)

    def test_pauses_inside_the_speech_are_excluded_too(self):
        """What separates this from the span between the first and last word."""
        signal = np.concatenate(
            [sine(2.0), audio.silence(SR, 1.5), sine(2.0), audio.silence(SR, 1.5), sine(2.0)]
        )
        assert audio.speech_seconds(signal, SR) == pytest.approx(6.0, abs=0.3)

    def test_never_exceeds_the_file(self):
        signal = sine(2.0)
        assert audio.speech_seconds(signal, SR) <= 2.0

    def test_silence_measures_nothing(self):
        assert audio.speech_seconds(audio.silence(SR, 3.0), SR) == 0.0

    def test_empty_input_is_not_a_crash(self):
        assert audio.speech_seconds(np.zeros(0, dtype=np.float32), SR) == 0.0
        assert audio.speech_seconds(sine(1.0), 0) == 0.0

    def test_level_does_not_change_the_answer(self):
        """The threshold is relative, so a quiet take measures like a loud one."""
        loud = np.concatenate([sine(2.0, amplitude=0.5), audio.silence(SR, 2.0)])
        quiet = np.concatenate([sine(2.0, amplitude=0.005), audio.silence(SR, 2.0)])
        assert audio.speech_seconds(quiet, SR) == pytest.approx(
            audio.speech_seconds(loud, SR), abs=0.1
        )
