from pathlib import Path

from tts_workflow import (
    build_srt_entries,
    build_voxcpm_payload,
    parse_structured_shorts,
    seconds_to_srt,
    split_text_blocks,
)


def test_split_text_blocks_keeps_sentence_groups_under_limit():
    text = "Primera frase corta. Segunda frase corta. Tercera frase demasiado larga, con parte uno, parte dos."

    blocks = split_text_blocks(text, max_chars=55)

    assert blocks == [
        "Primera frase corta. Segunda frase corta.",
        "Tercera frase demasiado larga, con parte uno,",
        "parte dos.",
    ]


def test_seconds_to_srt_rounds_to_milliseconds():
    assert seconds_to_srt(61.2345) == "00:01:01,234"


def test_build_srt_entries_uses_accumulated_durations():
    lines = build_srt_entries([("hola", 1.0), ("mundo", 2.5)])

    assert lines == [
        "1\n00:00:00,000 --> 00:00:01,000\nhola\n",
        "2\n00:00:01,000 --> 00:00:03,500\nmundo\n",
    ]


def test_parse_structured_shorts_reads_script_lines(tmp_path):
    input_file = Path(tmp_path) / "shorts.txt"
    input_file.write_text(
        "SHORT 1\nTitulo: A\nScript: primer short\n---\nSHORT 2\nScript: segundo short\n",
        encoding="utf-8",
    )

    assert parse_structured_shorts(str(input_file), encoding="utf-8") == ["primer short", "segundo short"]


def test_build_voxcpm_payload_preserves_quality_parameters():
    payload = build_voxcpm_payload(
        text="bloque",
        model_id="openbmb/VoxCPM2",
        prompt_text="prompt",
        reference_wav="ref.wav",
        cfg_value=2.0,
        inference_timesteps=12,
        normalize=False,
    )

    assert payload == {
        "text": "bloque",
        "model_id": "openbmb/VoxCPM2",
        "prompt_text": "prompt",
        "prompt_wav_path": "ref.wav",
        "reference_wav_path": "ref.wav",
        "cfg_value": 2.0,
        "inference_timesteps": 12,
        "normalize": False,
        "denoise": False,
        "trim_silence_vad": False,
    }
