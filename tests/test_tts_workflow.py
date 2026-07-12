from pathlib import Path

from tts_workflow import (
    AudioInfo,
    audit_audio_cache,
    build_srt_entries,
    build_voxcpm_payload,
    cache_wav_path,
    load_cache_manifest,
    parse_structured_shorts,
    payload_cache_key,
    prepare_cached_wav,
    record_cached_wav,
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


def test_payload_cache_key_is_stable_and_changes_with_quality_parameters(tmp_path):
    ref = Path(tmp_path) / "ref.wav"
    ref.write_bytes(b"ref")
    base = build_voxcpm_payload(
        text="bloque",
        model_id="openbmb/VoxCPM2",
        prompt_text="prompt",
        reference_wav=str(ref),
        cfg_value=2.0,
        inference_timesteps=12,
        normalize=False,
    )
    same = dict(base)
    changed = dict(base, inference_timesteps=10)

    assert payload_cache_key(base) == payload_cache_key(same)
    assert payload_cache_key(base) != payload_cache_key(changed)


def test_record_cached_wav_writes_manifest_and_promotes_cache_file(tmp_path):
    output_dir = Path(tmp_path) / "blocks"
    output_dir.mkdir()
    block_path = output_dir / "bloque_001.wav"
    block_path.write_bytes(b"RIFF")
    payload = build_voxcpm_payload(
        text="bloque",
        model_id="openbmb/VoxCPM2",
        prompt_text="prompt",
        reference_wav="ref.wav",
        cfg_value=2.0,
        inference_timesteps=12,
        normalize=False,
    )
    cache_key = payload_cache_key(payload)

    record_cached_wav(str(output_dir), cache_key, payload, str(block_path), AudioInfo(10, 20, 2.0))

    manifest = load_cache_manifest(str(output_dir))
    assert cache_key in manifest["entries"]
    assert Path(cache_wav_path(str(output_dir), cache_key)).read_bytes() == b"RIFF"
    assert manifest["entries"][cache_key]["duration"] == 2.0


def test_prepare_cached_wav_reuses_hash_cache_for_new_block_path(tmp_path):
    output_dir = Path(tmp_path) / "blocks"
    payload = build_voxcpm_payload(
        text="bloque",
        model_id="openbmb/VoxCPM2",
        prompt_text="prompt",
        reference_wav="ref.wav",
        cfg_value=2.0,
        inference_timesteps=12,
        normalize=False,
    )
    cache_key = payload_cache_key(payload)
    cache_path = Path(cache_wav_path(str(output_dir), cache_key))
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"RIFF")
    block_path = output_dir / "renamed_block.wav"

    returned_key, source = prepare_cached_wav(str(output_dir), str(block_path), payload)

    assert returned_key == cache_key
    assert source == "hash-cache"
    assert block_path.read_bytes() == b"RIFF"


def test_audit_audio_cache_counts_reusable_and_missing_blocks(tmp_path):
    output_dir = Path(tmp_path) / "blocks"
    blocks = ["uno", "dos"]

    def build_payload(text):
        return build_voxcpm_payload(
            text=text,
            model_id="openbmb/VoxCPM2",
            prompt_text="prompt",
            reference_wav="ref.wav",
            cfg_value=2.0,
            inference_timesteps=12,
            normalize=False,
        )

    first_key = payload_cache_key(build_payload("uno"))
    first_cache = Path(cache_wav_path(str(output_dir), first_key))
    first_cache.parent.mkdir(parents=True)
    first_cache.write_bytes(b"RIFF")

    audit = audit_audio_cache(
        blocks,
        str(output_dir),
        build_payload,
        lambda index: str(output_dir / f"bloque_{index:03d}.wav"),
    )

    assert audit == {"total": 2, "reusable": 1, "missing": 1}
