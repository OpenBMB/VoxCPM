from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "src" / "voxcpm" / "cli.py"
V1_MODEL_PATH = ROOT / "models" / "openbmb__VoxCPM1.5"
V2_MODEL_PATH = ROOT / "models" / "VoxCPM2-1B-newaudiovae-6hz-nope-sft"


pkg = types.ModuleType("voxcpm")
pkg.__path__ = [str(ROOT / "src" / "voxcpm")]
sys.modules.setdefault("voxcpm", pkg)

core_stub = types.ModuleType("voxcpm.core")


class StubVoxCPM:
    pass


core_stub.VoxCPM = StubVoxCPM
sys.modules["voxcpm.core"] = core_stub

spec = importlib.util.spec_from_file_location("voxcpm.cli", CLI_PATH)
cli = importlib.util.module_from_spec(spec)
sys.modules["voxcpm.cli"] = cli
assert spec.loader is not None
spec.loader.exec_module(cli)


class DummyTTSModel:
    sample_rate = 16000


class DummyModel:
    def __init__(self):
        self.tts_model = DummyTTSModel()
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return np.zeros(160, dtype=np.float32)


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["voxcpm", *argv])
    cli.main()


def patch_soundfile_write(monkeypatch):
    soundfile_stub = types.SimpleNamespace(write=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "soundfile", soundfile_stub)


def test_parser_defaults_to_voxcpm2():
    parser = cli._build_parser()
    args = parser.parse_args(["design", "--text", "hello", "--output", "out.wav"])
    assert args.hf_model_id == "openbmb/VoxCPM2"
    assert args.device == "auto"
    assert args.no_optimize is False


def test_load_model_respects_no_optimize_for_local_model(monkeypatch):
    calls = {}

    class FakeVoxCPM:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs
            self.tts_model = DummyTTSModel()

    monkeypatch.setattr(core_stub, "VoxCPM", FakeVoxCPM)
    args = cli._build_parser().parse_args(
        [
            "design",
            "--text",
            "hello",
            "--output",
            "out.wav",
            "--model-path",
            str(V2_MODEL_PATH),
            "--no-optimize",
        ]
    )

    cli.load_model(args)

    assert calls["kwargs"]["device"] == "auto"
    assert calls["kwargs"]["optimize"] is False


def test_load_model_defaults_optimize_for_hf(monkeypatch):
    calls = {}

    class FakeVoxCPM:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls["kwargs"] = kwargs
            return DummyModel()

    monkeypatch.setattr(core_stub, "VoxCPM", FakeVoxCPM)
    args = cli._build_parser().parse_args(
        [
            "design",
            "--text",
            "hello",
            "--output",
            "out.wav",
        ]
    )

    cli.load_model(args)

    assert calls["kwargs"]["device"] == "auto"
    assert calls["kwargs"]["optimize"] is True


def test_load_model_respects_no_optimize_for_hf(monkeypatch):
    calls = {}

    class FakeVoxCPM:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls["kwargs"] = kwargs
            return DummyModel()

    monkeypatch.setattr(core_stub, "VoxCPM", FakeVoxCPM)
    args = cli._build_parser().parse_args(
        [
            "design",
            "--text",
            "hello",
            "--output",
            "out.wav",
            "--no-optimize",
        ]
    )

    cli.load_model(args)

    assert calls["kwargs"]["device"] == "auto"
    assert calls["kwargs"]["optimize"] is False


def test_load_model_passes_explicit_device_to_hf(monkeypatch):
    calls = {}

    class FakeVoxCPM:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls["kwargs"] = kwargs
            return DummyModel()

    monkeypatch.setattr(core_stub, "VoxCPM", FakeVoxCPM)
    args = cli._build_parser().parse_args(
        [
            "design",
            "--text",
            "hello",
            "--output",
            "out.wav",
            "--device",
            "mps",
        ]
    )

    cli.load_model(args)

    assert calls["kwargs"]["device"] == "mps"


def test_design_subcommand_applies_control(monkeypatch, tmp_path):
    dummy_model = DummyModel()
    monkeypatch.setattr(cli, "load_model", lambda args: dummy_model)
    patch_soundfile_write(monkeypatch)

    run_main(
        monkeypatch,
        [
            "design",
            "--text",
            "hello",
            "--control",
            "warm female voice",
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    assert dummy_model.calls[0]["text"] == "(warm female voice)hello"
    assert dummy_model.calls[0]["prompt_wav_path"] is None
    assert dummy_model.calls[0]["reference_wav_path"] is None


def test_clone_subcommand_reads_prompt_file(monkeypatch, tmp_path):
    dummy_model = DummyModel()
    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"RIFF")
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("prompt transcript\n", encoding="utf-8")

    monkeypatch.setattr(cli, "load_model", lambda args: dummy_model)
    patch_soundfile_write(monkeypatch)

    run_main(
        monkeypatch,
        [
            "clone",
            "--text",
            "hello",
            "--prompt-audio",
            str(prompt_audio),
            "--prompt-file",
            str(prompt_file),
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    assert dummy_model.calls[0]["prompt_wav_path"] == str(prompt_audio)
    assert dummy_model.calls[0]["prompt_text"] == "prompt transcript"


def test_clone_rejects_reference_audio_for_v1_local_model(monkeypatch, tmp_path):
    reference_audio = tmp_path / "ref.wav"
    reference_audio.write_bytes(b"RIFF")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voxcpm",
            "clone",
            "--text",
            "hello",
            "--reference-audio",
            str(reference_audio),
            "--model-path",
            str(V1_MODEL_PATH),
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()


def test_clone_rejects_reference_audio_for_v1_hf_model_id(monkeypatch, tmp_path):
    reference_audio = tmp_path / "ref.wav"
    reference_audio.write_bytes(b"RIFF")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voxcpm",
            "clone",
            "--text",
            "hello",
            "--reference-audio",
            str(reference_audio),
            "--hf-model-id",
            "openbmb/VoxCPM1.5",
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()


def test_legacy_root_args_still_work_and_warn(monkeypatch, tmp_path, capsys):
    dummy_model = DummyModel()
    monkeypatch.setattr(cli, "load_model", lambda args: dummy_model)
    patch_soundfile_write(monkeypatch)

    run_main(
        monkeypatch,
        [
            "--text",
            "hello",
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    captured = capsys.readouterr()
    assert "deprecated" in captured.err
    assert dummy_model.calls[0]["text"] == "hello"


def test_batch_subcommand_applies_control(monkeypatch, tmp_path):
    dummy_model = DummyModel()
    input_file = tmp_path / "texts.txt"
    input_file.write_text("hello\nworld\n", encoding="utf-8")

    monkeypatch.setattr(cli, "load_model", lambda args: dummy_model)
    patch_soundfile_write(monkeypatch)

    run_main(
        monkeypatch,
        [
            "batch",
            "--input",
            str(input_file),
            "--output-dir",
            str(tmp_path / "outs"),
            "--control",
            "calm narrator",
        ],
    )

    assert [call["text"] for call in dummy_model.calls] == [
        "(calm narrator)hello",
        "(calm narrator)world",
    ]


def test_design_writes_timestamp_json_when_requested(monkeypatch, tmp_path):
    dummy_model = DummyModel()
    timestamp_calls = []

    def fake_align_audio_file(**kwargs):
        timestamp_calls.append(kwargs)
        return {
            "audio_path": kwargs["audio_path"],
            "sample_rate": kwargs["sample_rate"],
            "backend": kwargs["backend"],
            "level": kwargs["level"],
            "text": kwargs["text"],
            "items": [{"text": "hello", "start": 0.0, "end": 0.5, "level": "word"}],
            "warning": None,
        }

    monkeypatch.setattr(cli, "load_model", lambda args: dummy_model)
    monkeypatch.setattr(cli, "align_audio_file", fake_align_audio_file)
    patch_soundfile_write(monkeypatch)

    output = tmp_path / "out.wav"
    run_main(
        monkeypatch,
        [
            "design",
            "--text",
            "hello",
            "--output",
            str(output),
            "--timestamps",
            "--timestamp-language",
            "en",
        ],
    )

    timestamp_path = tmp_path / "out.timestamps.json"
    payload = json.loads(timestamp_path.read_text(encoding="utf-8"))
    assert payload["audio_path"] == str(output)
    assert payload["text"] == "hello"
    assert payload["items"][0]["text"] == "hello"
    assert timestamp_calls[0]["language"] == "en"
    assert timestamp_calls[0]["level"] == "word"


def test_timestamp_alignment_failure_warns_by_default(monkeypatch, tmp_path, capsys):
    dummy_model = DummyModel()

    def fake_align_audio_file(**kwargs):
        raise RuntimeError("alignment failed")

    monkeypatch.setattr(cli, "load_model", lambda args: dummy_model)
    monkeypatch.setattr(cli, "align_audio_file", fake_align_audio_file)
    patch_soundfile_write(monkeypatch)

    run_main(
        monkeypatch,
        [
            "design",
            "--text",
            "hello",
            "--output",
            str(tmp_path / "out.wav"),
            "--timestamps",
        ],
    )

    assert "Timestamp alignment failed" in capsys.readouterr().err


def test_timestamp_alignment_failure_exits_in_strict_mode(monkeypatch, tmp_path):
    dummy_model = DummyModel()

    def fake_align_audio_file(**kwargs):
        raise RuntimeError("alignment failed")

    monkeypatch.setattr(cli, "load_model", lambda args: dummy_model)
    monkeypatch.setattr(cli, "align_audio_file", fake_align_audio_file)
    patch_soundfile_write(monkeypatch)

    with pytest.raises(SystemExit):
        run_main(
            monkeypatch,
            [
                "design",
                "--text",
                "hello",
                "--output",
                str(tmp_path / "out.wav"),
                "--timestamps",
                "--timestamp-strict",
            ],
        )


def test_batch_writes_one_timestamp_json_per_output(monkeypatch, tmp_path):
    dummy_model = DummyModel()

    def fake_align_audio_file(**kwargs):
        return {
            "audio_path": kwargs["audio_path"],
            "sample_rate": kwargs["sample_rate"],
            "backend": kwargs["backend"],
            "level": kwargs["level"],
            "text": kwargs["text"],
            "items": [{"text": kwargs["text"], "start": 0.0, "end": 0.5, "level": "word"}],
            "warning": None,
        }

    input_file = tmp_path / "texts.txt"
    input_file.write_text("hello\nworld\n", encoding="utf-8")
    output_dir = tmp_path / "outs"

    monkeypatch.setattr(cli, "load_model", lambda args: dummy_model)
    monkeypatch.setattr(cli, "align_audio_file", fake_align_audio_file)
    patch_soundfile_write(monkeypatch)

    run_main(
        monkeypatch,
        [
            "batch",
            "--input",
            str(input_file),
            "--output-dir",
            str(output_dir),
            "--timestamps",
        ],
    )

    first = json.loads((output_dir / "output_001.timestamps.json").read_text(encoding="utf-8"))
    second = json.loads((output_dir / "output_002.timestamps.json").read_text(encoding="utf-8"))
    assert first["text"] == "hello"
    assert second["text"] == "world"


def test_legacy_clone_with_prompt_file_still_works(monkeypatch, tmp_path, capsys):
    dummy_model = DummyModel()
    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"RIFF")
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("legacy transcript", encoding="utf-8")

    monkeypatch.setattr(cli, "load_model", lambda args: dummy_model)
    patch_soundfile_write(monkeypatch)

    run_main(
        monkeypatch,
        [
            "--text",
            "hello",
            "--prompt-audio",
            str(prompt_audio),
            "--prompt-file",
            str(prompt_file),
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    captured = capsys.readouterr()
    assert "deprecated" in captured.err
    assert dummy_model.calls[0]["prompt_text"] == "legacy transcript"


def test_invalid_prompt_text_and_prompt_file_combination(monkeypatch, tmp_path, capsys):
    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"RIFF")
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("transcript", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voxcpm",
            "clone",
            "--text",
            "hello",
            "--prompt-audio",
            str(prompt_audio),
            "--prompt-text",
            "inline transcript",
            "--prompt-file",
            str(prompt_file),
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

    assert "Use either --prompt-text or --prompt-file" in capsys.readouterr().err


def test_missing_prompt_file_reports_parser_error(monkeypatch, tmp_path, capsys):
    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"RIFF")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voxcpm",
            "clone",
            "--text",
            "hello",
            "--prompt-audio",
            str(prompt_audio),
            "--prompt-file",
            str(tmp_path / "missing.txt"),
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

    assert "prompt text file" in capsys.readouterr().err


def test_design_rejects_prompt_audio_args(monkeypatch, tmp_path, capsys):
    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"RIFF")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voxcpm",
            "design",
            "--text",
            "hello",
            "--prompt-audio",
            str(prompt_audio),
            "--prompt-text",
            "transcript",
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

    assert "does not accept prompt/reference audio" in capsys.readouterr().err


def test_clone_rejects_prompt_audio_without_transcript(monkeypatch, tmp_path, capsys):
    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"RIFF")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voxcpm",
            "clone",
            "--text",
            "hello",
            "--prompt-audio",
            str(prompt_audio),
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

    assert "--prompt-audio requires --prompt-text or --prompt-file" in capsys.readouterr().err


def test_clone_rejects_transcript_without_prompt_audio(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voxcpm",
            "clone",
            "--text",
            "hello",
            "--prompt-text",
            "transcript",
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

    assert "--prompt-text/--prompt-file requires --prompt-audio" in capsys.readouterr().err


def test_batch_rejects_control_with_prompt_transcript(monkeypatch, tmp_path, capsys):
    input_file = tmp_path / "texts.txt"
    input_file.write_text("hello\n", encoding="utf-8")
    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"RIFF")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voxcpm",
            "batch",
            "--input",
            str(input_file),
            "--output-dir",
            str(tmp_path / "outs"),
            "--control",
            "calm narrator",
            "--prompt-audio",
            str(prompt_audio),
            "--prompt-text",
            "transcript",
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

    assert "--control cannot be used together" in capsys.readouterr().err


def test_batch_invalid_args_do_not_create_output_directory(monkeypatch, tmp_path):
    input_file = tmp_path / "texts.txt"
    input_file.write_text("hello\n", encoding="utf-8")
    prompt_audio = tmp_path / "prompt.wav"
    prompt_audio.write_bytes(b"RIFF")
    output_dir = tmp_path / "outs"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voxcpm",
            "batch",
            "--input",
            str(input_file),
            "--output-dir",
            str(output_dir),
            "--control",
            "calm narrator",
            "--prompt-audio",
            str(prompt_audio),
            "--prompt-text",
            "transcript",
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()

    assert not output_dir.exists()


def test_detect_model_architecture_uses_local_configs():
    parser = cli._build_parser()
    v1_args = parser.parse_args(
        [
            "clone",
            "--text",
            "hello",
            "--reference-audio",
            "ref.wav",
            "--model-path",
            str(V1_MODEL_PATH),
            "--output",
            "out.wav",
        ]
    )
    v2_args = parser.parse_args(
        [
            "clone",
            "--text",
            "hello",
            "--reference-audio",
            "ref.wav",
            "--model-path",
            str(V2_MODEL_PATH),
            "--output",
            "out.wav",
        ]
    )

    assert cli.detect_model_architecture(v1_args) == "voxcpm"
    assert cli.detect_model_architecture(v2_args) == "voxcpm2"


def test_parser_accepts_seed():
    parser = cli._build_parser()
    # Default seed should be None
    args = parser.parse_args(["design", "--text", "hello", "--output", "out.wav"])
    assert args.seed is None

    # Custom seed should be parsed as int
    args = parser.parse_args(["design", "--text", "hello", "--output", "out.wav", "--seed", "42"])
    assert args.seed == 42


def test_design_subcommand_passes_seed(monkeypatch, tmp_path):
    dummy_model = DummyModel()
    monkeypatch.setattr(cli, "load_model", lambda args: dummy_model)
    patch_soundfile_write(monkeypatch)

    run_main(
        monkeypatch,
        [
            "design",
            "--text",
            "hello",
            "--seed",
            "123",
            "--output",
            str(tmp_path / "out.wav"),
        ],
    )

    assert dummy_model.calls[0]["seed"] == 123


def test_batch_subcommand_passes_seed(monkeypatch, tmp_path):
    dummy_model = DummyModel()
    input_file = tmp_path / "texts.txt"
    input_file.write_text("hello\nworld\n", encoding="utf-8")

    monkeypatch.setattr(cli, "load_model", lambda args: dummy_model)
    patch_soundfile_write(monkeypatch)

    run_main(
        monkeypatch,
        [
            "batch",
            "--input",
            str(input_file),
            "--output-dir",
            str(tmp_path / "outs"),
            "--seed",
            "999",
        ],
    )

    assert len(dummy_model.calls) == 2
    assert dummy_model.calls[0]["seed"] == 999
    assert dummy_model.calls[1]["seed"] == 999
