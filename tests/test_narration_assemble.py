"""Tests for joining chapters into a single chaptered audiobook.

ffmpeg is not assumed to be installed — the point of most of these tests is that
its absence costs only the encoding step, never the synthesized audio.
"""
import numpy as np
import pytest
import soundfile as sf

from narration import assemble
from narration.assemble import Chapter

SR = 24000


@pytest.fixture
def chapter_files(tmp_path):
    """Three chapters of 1.0 s, 2.0 s and 0.5 s."""
    paths = []
    for index, seconds in enumerate([1.0, 2.0, 0.5], start=1):
        t = np.arange(int(SR * seconds)) / SR
        path = tmp_path / f"chapitre_{index:03d}.wav"
        sf.write(str(path), (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR)
        paths.append(path)
    return paths


class TestConcat:
    def test_durations_and_offsets(self, chapter_files, tmp_path):
        chapters, sample_rate = assemble.concat_chapters(
            chapter_files, tmp_path / "book.wav", gap_sec=1.0
        )
        assert sample_rate == SR
        assert [round(c.duration_sec, 2) for c in chapters] == [1.0, 2.0, 0.5]
        # Each chapter starts after the previous one plus the 1 s gap.
        assert [round(c.start_sec, 2) for c in chapters] == [0.0, 2.0, 5.0]

    def test_total_length_includes_the_gaps(self, chapter_files, tmp_path):
        out = tmp_path / "book.wav"
        assemble.concat_chapters(chapter_files, out, gap_sec=1.0)
        info = sf.info(str(out))
        assert info.frames / info.samplerate == pytest.approx(5.5, abs=0.01)

    def test_output_is_16_bit_pcm(self, chapter_files, tmp_path):
        out = tmp_path / "book.wav"
        assemble.concat_chapters(chapter_files, out)
        assert sf.info(str(out)).subtype == "PCM_16"

    def test_titles_default_to_filenames(self, chapter_files, tmp_path):
        chapters, _ = assemble.concat_chapters(chapter_files, tmp_path / "book.wav")
        assert chapters[0].title == "Chapitre 001"

    def test_explicit_titles_win(self, chapter_files, tmp_path):
        chapters, _ = assemble.concat_chapters(
            chapter_files, tmp_path / "book.wav", titles=["Le début", "Le milieu", "La fin"]
        )
        assert [c.title for c in chapters] == ["Le début", "Le milieu", "La fin"]

    def test_no_chapters_is_an_error(self, tmp_path):
        with pytest.raises(ValueError):
            assemble.concat_chapters([], tmp_path / "book.wav")

    def test_a_missing_chapter_is_reported_by_name(self, chapter_files, tmp_path):
        with pytest.raises(FileNotFoundError, match="manquant"):
            assemble.concat_chapters(
                chapter_files + [tmp_path / "manquant.wav"], tmp_path / "book.wav"
            )

    def test_mismatched_sample_rates_are_refused(self, chapter_files, tmp_path):
        odd = tmp_path / "chapitre_004.wav"
        sf.write(str(odd), np.zeros(16000, dtype=np.float32), 16000)
        with pytest.raises(ValueError, match="sample rate"):
            assemble.concat_chapters(chapter_files + [odd], tmp_path / "book.wav")


class TestFfmetadata:
    def test_header_and_tags(self):
        text = assemble.build_ffmetadata(
            [Chapter(path=None, title="Un", start_sec=0.0, duration_sec=1.0)],
            title="Mon Livre",
            author="Edwin",
        )
        assert text.startswith(";FFMETADATA1")
        assert "title=Mon Livre" in text
        assert "artist=Edwin" in text

    def test_markers_are_contiguous_so_players_never_show_a_gap(self):
        chapters = [
            Chapter(path=None, title="Un", start_sec=0.0, duration_sec=1.0),
            Chapter(path=None, title="Deux", start_sec=2.5, duration_sec=1.0),
        ]
        text = assemble.build_ffmetadata(chapters)
        # The first marker runs to where the second begins (2500 ms), not to the
        # end of its own audio (1000 ms) — the silence belongs to chapter one.
        assert "START=0\nEND=2500" in text
        assert "START=2500\nEND=3500" in text

    def test_special_characters_are_escaped(self):
        text = assemble.build_ffmetadata(
            [Chapter(path=None, title="A=B; #1", start_sec=0.0, duration_sec=1.0)]
        )
        assert r"title=A\=B\; \#1" in text

    def test_a_newline_in_a_title_cannot_break_the_format(self):
        text = assemble.build_ffmetadata(
            [Chapter(path=None, title="Deux\nlignes", start_sec=0.0, duration_sec=1.0)]
        )
        assert "title=Deux lignes" in text
        assert text.count("[CHAPTER]") == 1


class TestFfmpegCommand:
    def test_m4b_uses_aac_and_faststart(self, tmp_path):
        command = assemble.ffmpeg_command(tmp_path / "b.wav", tmp_path / "b.txt", tmp_path / "b.m4b")
        assert "aac" in command and "+faststart" in command

    def test_mp3_uses_lame(self, tmp_path):
        command = assemble.ffmpeg_command(tmp_path / "b.wav", tmp_path / "b.txt", tmp_path / "b.mp3")
        assert "libmp3lame" in command
        assert "+faststart" not in command

    def test_metadata_is_mapped_from_the_chapter_file(self, tmp_path):
        command = assemble.ffmpeg_command(tmp_path / "b.wav", tmp_path / "b.txt", tmp_path / "b.m4b")
        assert command[command.index("-map_metadata") + 1] == "1"

    def test_a_cover_is_attached_as_a_picture(self, tmp_path):
        command = assemble.ffmpeg_command(
            tmp_path / "b.wav", tmp_path / "b.txt", tmp_path / "b.m4b", cover_path=tmp_path / "c.jpg"
        )
        assert "attached_pic" in command


class TestBitrate:
    def test_each_container_has_a_default(self, tmp_path):
        """64k AAC is what Audible streams; MP3 needs more to sound the same."""
        m4b = assemble.ffmpeg_command(tmp_path / "b.wav", tmp_path / "b.txt", tmp_path / "b.m4b")
        mp3 = assemble.ffmpeg_command(tmp_path / "b.wav", tmp_path / "b.txt", tmp_path / "b.mp3")
        assert m4b[m4b.index("-b:a") + 1] == "64k"
        assert mp3[mp3.index("-b:a") + 1] == "128k"

    def test_it_can_be_raised_for_an_archive_copy(self, tmp_path):
        command = assemble.ffmpeg_command(
            tmp_path / "b.wav", tmp_path / "b.txt", tmp_path / "b.m4b", bitrate="192k"
        )
        assert command[command.index("-b:a") + 1] == "192k"

    def test_a_user_may_write_it_however_they_like(self):
        assert assemble.normalize_bitrate(128) == "128k"
        assert assemble.normalize_bitrate("128") == "128k"
        assert assemble.normalize_bitrate("128k") == "128k"
        assert assemble.normalize_bitrate("128K") == "128k"

    def test_nothing_means_keep_the_default(self):
        assert assemble.normalize_bitrate(None) is None
        assert assemble.normalize_bitrate("") is None
        assert assemble.normalize_bitrate("   ") is None

    def test_nonsense_is_refused_rather_than_passed_to_ffmpeg(self):
        for value in ("beaucoup", "-64", "0", "12x8"):
            with pytest.raises(ValueError):
                assemble.normalize_bitrate(value)

    def test_it_reaches_the_encoder_through_assemble(self, chapter_files, tmp_path, monkeypatch):
        monkeypatch.setattr(assemble, "find_ffmpeg", lambda: None)
        result = assemble.assemble(chapter_files, tmp_path / "livre.m4b", bitrate=96)
        assert "96k" in result.pending_command


class TestAssemble:
    def test_audio_and_markers_survive_a_missing_ffmpeg(self, chapter_files, tmp_path, monkeypatch):
        monkeypatch.setattr(assemble, "find_ffmpeg", lambda: None)
        result = assemble.assemble(chapter_files, tmp_path / "livre.m4b", title="Mon Livre")

        assert result.wav_path.is_file()
        assert result.metadata_path.is_file()
        assert result.output_path is None
        assert result.pending_command and result.pending_command[0] == "ffmpeg"
        assert "ffmpeg" in result.message

    def test_the_pending_command_is_the_one_that_would_have_run(self, chapter_files, tmp_path, monkeypatch):
        monkeypatch.setattr(assemble, "find_ffmpeg", lambda: None)
        result = assemble.assemble(chapter_files, tmp_path / "livre.m4b")
        assert str(result.wav_path) in result.pending_command
        assert str(result.metadata_path) in result.pending_command

    def test_a_wav_target_needs_no_encoder_at_all(self, chapter_files, tmp_path, monkeypatch):
        monkeypatch.setattr(assemble, "find_ffmpeg", lambda: None)
        result = assemble.assemble(chapter_files, tmp_path / "livre.wav")
        assert result.output_path == tmp_path / "livre.wav"
        assert result.pending_command is None

    def test_a_failing_ffmpeg_still_leaves_the_audio(self, chapter_files, tmp_path, monkeypatch):
        import subprocess

        monkeypatch.setattr(assemble, "find_ffmpeg", lambda: "ffmpeg")
        monkeypatch.setattr(
            assemble.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "Encoder not found"),
        )
        result = assemble.assemble(chapter_files, tmp_path / "livre.m4b")
        assert result.wav_path.is_file()
        assert result.output_path is None
        assert "Encoder not found" in result.message

    def test_chapter_totals(self, chapter_files, tmp_path, monkeypatch):
        monkeypatch.setattr(assemble, "find_ffmpeg", lambda: None)
        result = assemble.assemble(chapter_files, tmp_path / "livre.m4b", gap_sec=1.0)
        assert len(result.chapters) == 3
        assert result.duration_sec == pytest.approx(5.5, abs=0.01)
        assert result.sample_rate == SR
