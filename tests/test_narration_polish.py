"""Tests for the studio chain and the LUFS measurement.

The loudness figure is the one number here that has a right answer defined
outside this repository, and it was cross-checked against ffmpeg's ebur128
filter on real narrated chapters: -20.98 against -20.9, -20.27 against -20.2,
-20.96 against -20.9. What the tests below pin is the behaviour that follows
from the specification — the gates, the 6 dB relation, the offset — so a change
that breaks it fails here rather than on someone's upload.
"""
import numpy as np
import pytest

from narration import audio, polish

SR = 24000


def sine(seconds: float, frequency: float = 1000.0, amplitude: float = 0.1) -> np.ndarray:
    t = np.arange(int(SR * seconds), dtype=np.float32) / SR
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def voice_like(seconds: float = 3.0) -> np.ndarray:
    """Something with a fundamental, harmonics and an envelope."""
    t = np.arange(int(SR * seconds), dtype=np.float32) / SR
    body = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate([180, 360, 720, 1440]))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    signal = (body * envelope).astype(np.float32)
    return signal * (0.1 / max(float(np.max(np.abs(signal))), 1e-9))


class TestHighpass:
    # Measured away from the edges: a zero-phase filter rings briefly at both
    # ends of a signal that starts mid-cycle. Real chapters open and close on
    # room tone, so this never happens to them — only to a bare test sine.
    STEADY = slice(int(SR * 0.5), int(SR * 1.5))

    def test_rumble_is_removed(self):
        rumble = sine(2.0, frequency=35.0, amplitude=0.3)
        cleaned = polish.highpass(rumble, SR, 80.0)
        assert audio.peak_db(cleaned[self.STEADY]) < audio.peak_db(rumble[self.STEADY]) - 20

    def test_the_voice_band_is_left_alone(self):
        voice = sine(2.0, frequency=400.0, amplitude=0.2)
        cleaned = polish.highpass(voice, SR, 80.0)
        assert audio.peak_db(cleaned[self.STEADY]) == pytest.approx(
            audio.peak_db(voice[self.STEADY]), abs=0.5
        )

    def test_it_does_not_shift_the_signal_in_time(self):
        """Zero-phase: a transient must not move, or consonants smear."""
        click = np.zeros(SR, dtype=np.float32)
        click[SR // 2] = 0.5
        filtered = polish.highpass(click, SR, 80.0)
        assert abs(int(np.argmax(np.abs(filtered))) - SR // 2) <= 2

    def test_a_very_short_signal_is_returned_rather_than_refused(self):
        tiny = sine(0.001)
        assert polish.highpass(tiny, SR, 80.0).size == tiny.size

    def test_a_nonsense_cutoff_is_ignored(self):
        voice = voice_like(0.5)
        assert np.array_equal(polish.highpass(voice, SR, 0.0), voice)
        assert np.array_equal(polish.highpass(voice, SR, SR), voice)


class TestDeess:
    @staticmethod
    def with_sibilance() -> np.ndarray:
        """A voice with one harsh 7 kHz burst in the middle of it."""
        voice = voice_like(3.0)
        burst = np.zeros_like(voice)
        start, end = int(SR * 1.4), int(SR * 1.6)
        burst[start:end] = sine(0.2, frequency=7000.0, amplitude=0.35)
        return voice + burst

    def test_the_sibilant_peak_is_pulled_down(self):
        harsh = self.with_sibilance()
        treated = polish.deess(harsh, SR)
        start, end = int(SR * 1.4), int(SR * 1.6)
        before = audio.peak_db(harsh[start:end])
        after = audio.peak_db(treated[start:end])
        assert after < before - 1.5

    def test_the_rest_of_the_voice_is_untouched(self):
        harsh = self.with_sibilance()
        treated = polish.deess(harsh, SR)
        quiet = slice(0, int(SR * 1.0))
        assert audio.peak_db(treated[quiet]) == pytest.approx(
            audio.peak_db(harsh[quiet]), abs=0.6
        )

    def test_it_can_be_switched_off(self):
        harsh = self.with_sibilance()
        off = polish.PolishSettings(deess=False)
        assert np.array_equal(polish.deess(harsh, SR, off), harsh)

    def test_a_voice_without_sibilance_is_barely_changed(self):
        """It must not sound like a blanket over a narrator who never hisses."""
        voice = voice_like(3.0)
        treated = polish.deess(voice, SR)
        assert audio.speech_rms_db(treated, SR) == pytest.approx(
            audio.speech_rms_db(voice, SR), abs=1.0
        )


class TestCompress:
    @staticmethod
    def uneven() -> np.ndarray:
        """A quiet line followed by a loud one — the case that loses sentences."""
        return np.concatenate([voice_like(2.0) * 0.15, voice_like(2.0)])

    def test_the_gap_between_quiet_and_loud_closes(self):
        source = self.uneven()
        treated = polish.compress(source, SR)
        half = source.size // 2
        before = audio.speech_rms_db(source[half:], SR) - audio.speech_rms_db(source[:half], SR)
        after = audio.speech_rms_db(treated[half:], SR) - audio.speech_rms_db(treated[:half], SR)
        assert after < before

    def test_it_never_raises_the_level(self):
        """Make-up gain belongs to the normalisation, not here."""
        source = self.uneven()
        treated = polish.compress(source, SR)
        assert audio.peak_db(treated) <= audio.peak_db(source) + 0.01

    def test_the_threshold_follows_the_signal_not_dbfs(self):
        """A quiet chapter and a loud one must be treated the same way."""
        loud = self.uneven()
        quiet = loud * 0.05
        loud_reduction = audio.peak_db(loud) - audio.peak_db(polish.compress(loud, SR))
        quiet_reduction = audio.peak_db(quiet) - audio.peak_db(polish.compress(quiet, SR))
        assert loud_reduction == pytest.approx(quiet_reduction, abs=0.5)

    def test_it_can_be_switched_off(self):
        source = self.uneven()
        off = polish.PolishSettings(compress=False)
        assert np.array_equal(polish.compress(source, SR, off), source)


class TestLimit:
    @staticmethod
    def with_peaks() -> np.ndarray:
        """Sustained speech with a few short spikes standing well above it."""
        voice = voice_like(4.0)
        for position in (0.7, 1.9, 3.1):
            start = int(SR * position)
            voice[start : start + int(SR * 0.004)] *= 6.0
        return np.clip(voice, -1.0, 1.0)

    def test_the_crest_factor_comes_down(self):
        source = self.with_peaks()
        limited = polish.limit(source, SR)
        before = audio.peak_db(source) - audio.speech_rms_db(source, SR)
        after = audio.peak_db(limited) - audio.speech_rms_db(limited, SR)
        assert after < before - 2.0

    def test_it_leaves_a_chapter_that_is_already_controlled_alone(self):
        voice = voice_like(4.0)
        limited = polish.limit(voice, SR)
        assert audio.speech_rms_db(limited, SR) == pytest.approx(
            audio.speech_rms_db(voice, SR), abs=0.5
        )

    def test_the_target_crest_is_respected(self):
        source = self.with_peaks()
        settings = polish.PolishSettings(limit_crest_db=10.0)
        limited = polish.limit(source, SR, settings)
        crest = audio.peak_db(limited) - audio.speech_rms_db(limited, SR)
        assert crest <= 10.0 + 2.0

    def test_it_can_be_switched_off(self):
        source = self.with_peaks()
        off = polish.PolishSettings(limit=False)
        assert np.array_equal(polish.limit(source, SR, off), source)

    def test_the_chain_ends_with_more_headroom_than_it_started(self):
        """This is what stops the ceiling from dragging the chapter quiet."""
        source = self.with_peaks()
        treated = polish.polish(source, SR)
        before = audio.peak_db(source) - audio.speech_rms_db(source, SR)
        after = audio.peak_db(treated) - audio.speech_rms_db(treated, SR)
        assert after < before


class TestLoudness:
    def test_doubling_the_amplitude_adds_six_units(self):
        voice = voice_like(4.0)
        assert polish.loudness_lufs(voice * 2, SR) == pytest.approx(
            polish.loudness_lufs(voice, SR) + 6.02, abs=0.05
        )

    def test_silence_has_no_loudness(self):
        assert polish.loudness_lufs(audio.silence(SR, 3.0), SR) == float("-inf")

    def test_a_signal_shorter_than_a_block_has_none_either(self):
        assert polish.loudness_lufs(sine(0.2), SR) == float("-inf")

    def test_silence_between_sentences_does_not_drag_it_down(self):
        """The gates exist for exactly this: a pause is not quiet programme."""
        speech = voice_like(4.0)
        with_pauses = np.concatenate([speech, audio.silence(SR, 4.0), speech])
        assert polish.loudness_lufs(with_pauses, SR) == pytest.approx(
            polish.loudness_lufs(speech, SR), abs=0.6
        )

    def test_it_is_reported_in_the_acx_report(self):
        report = audio.acx_report(voice_like(4.0), SR)
        assert "lufs" in report
        assert np.isfinite(report["lufs"])

    def test_it_is_reported_but_never_gated_on(self):
        """No standard states one LUFS target, so none is invented here."""
        report = audio.acx_report(voice_like(4.0), SR)
        assert not any(key.startswith("lufs") and key.endswith("_ok") for key in report)


class TestTheChain:
    def test_order_is_correct_then_control(self):
        """Rumble must be gone before the compressor can duck on it."""
        rumble = sine(3.0, frequency=30.0, amplitude=0.4)
        source = voice_like(3.0) + rumble
        treated = polish.polish(source, SR)
        # What is left below 80 Hz is a fraction of what went in.
        low_before = np.std(source - polish.highpass(source, SR, 80.0))
        low_after = np.std(treated - polish.highpass(treated, SR, 80.0))
        assert low_after < low_before / 4

    def test_nothing_happens_when_everything_is_off(self):
        source = voice_like(2.0)
        off = polish.PolishSettings(
            highpass_hz=0.0, expand=False, deess=False, compress=False, limit=False
        )
        assert not off.enabled
        assert np.array_equal(polish.polish(source, SR, off), source)

    def test_it_does_not_set_the_level(self):
        """Levelling happens once, later, in stitch."""
        source = voice_like(3.0) * 0.02
        treated = polish.polish(source, SR)
        assert audio.speech_rms_db(treated, SR) < -20

    def test_empty_audio_survives(self):
        assert polish.polish(np.zeros(0, dtype=np.float32), SR).size == 0


class TestWiring:
    def test_a_stitched_chapter_is_polished_by_default(self):
        segments = [(voice_like(2.0) + sine(2.0, 30.0, 0.4), 0.0)]
        polished = audio.stitch(segments, SR, audio.MasteringSettings(), normalize=False)
        raw = audio.stitch(
            segments, SR, audio.MasteringSettings(polish=False), normalize=False
        )
        assert not np.array_equal(polished, raw)

    def test_it_can_be_turned_off_from_the_mastering_settings(self):
        settings = audio.MasteringSettings(polish=False, trim_silence=False)
        source = voice_like(1.0)
        result = audio.stitch([(source, 0.0)], SR, settings, normalize=False)
        # The audio passed through untouched between the room tone and the
        # de-click fades, which stitch applies whatever the polish setting.
        lead = int(SR * settings.lead_sec)
        middle = slice(lead + 1000, lead + source.size - 1000)
        assert np.allclose(result[middle], source[1000:-1000], atol=1e-6)

    def test_the_flag_survives_a_saved_plan(self):
        """plan.json holds flat JSON; a nested dataclass would not round-trip."""
        import dataclasses

        payload = dataclasses.asdict(audio.MasteringSettings())
        assert payload["polish"] is True
        assert all(not isinstance(value, dict) for value in payload.values())


class TestExpandDown:
    """Le fond doit descendre sous la limite ACX sans emporter la voix.

    Une voix clonée hérite du bruit de sa référence : Alex Somerset rend un
    plancher à -58 dBFS quand Aurore est à -70, et l'ACX refuse au-dessus de
    -60. Mesuré sur un chapitre réel : -58,7 devient -72,3, pour 1,2 dB de
    parole en moins que la normalisation qui suit rattrape.
    """

    def _voix_bruitee(self, sr=44100, secondes=6.0):
        # Parole intermittente sur un fond constant, comme un chapitre.
        n = int(sr * secondes)
        t = np.arange(n) / sr
        parole = 0.2 * np.sin(2 * np.pi * 150 * t)
        enveloppe = ((t % 2.0) < 1.0).astype(np.float32)   # 1 s de voix, 1 s de silence
        fond = 0.0012 * np.random.default_rng(0).standard_normal(n)
        return (parole * enveloppe + fond).astype(np.float32), sr

    def test_the_floor_comes_down(self):
        x, sr = self._voix_bruitee()
        avant = audio.noise_floor_db(x, sr)
        apres = audio.noise_floor_db(polish.expand_down(x, sr), sr)
        assert apres < avant - 6

    def test_speech_is_left_almost_alone(self):
        x, sr = self._voix_bruitee()
        avant = audio.speech_rms_db(x, sr)
        apres = audio.speech_rms_db(polish.expand_down(x, sr), sr)
        assert abs(apres - avant) < 4

    def test_the_reduction_is_capped(self):
        # Un silence creusé sans limite s'entend comme un trou.
        x, sr = self._voix_bruitee()
        y = polish.expand_down(x, sr)
        creux = 20 * np.log10(np.abs(y).max() / (np.abs(x).max() + 1e-12) + 1e-12)
        assert creux > -6

    def test_disabling_it_changes_nothing(self):
        x, sr = self._voix_bruitee()
        s = polish.PolishSettings(expand=False)
        assert np.array_equal(polish.expand_down(x, sr, s), audio.as_float_mono(x))

    def test_silence_survives_it(self):
        assert polish.expand_down(np.zeros(0, dtype=np.float32), 44100).size == 0
