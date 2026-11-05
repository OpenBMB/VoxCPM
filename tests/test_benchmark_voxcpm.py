from __future__ import annotations

import argparse
import importlib.util
import json
import runpy
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "benchmarks" / "benchmark_voxcpm.py"
SAMPLE_RATE = 16_000
MEBIBYTE = 1024 * 1024

spec = importlib.util.spec_from_file_location("benchmark_voxcpm", BENCHMARK_PATH)
benchmark_voxcpm = importlib.util.module_from_spec(spec)
sys.modules["benchmark_voxcpm"] = benchmark_voxcpm
assert spec.loader is not None
spec.loader.exec_module(benchmark_voxcpm)


class FakeModel:
    class TTSModel:
        sample_rate = SAMPLE_RATE
        device = "cpu"

    tts_model = TTSModel()

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [0.0] * SAMPLE_RATE


class FakeCudaMemory:
    def __init__(self, peak_bytes=256 * MEBIBYTE):
        self.peak_bytes = peak_bytes
        self.bound_model = None
        self.reset_calls = 0

    def bind(self, model):
        self.bound_model = model

    def reset(self):
        self.reset_calls += 1

    def synchronize(self):
        pass

    def peak_megabytes(self):
        return self.peak_bytes / MEBIBYTE


def test_run_benchmark_reports_load_rtf_throughput_and_peak_memory():
    model = FakeModel()
    clock = iter([0.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]).__next__
    memory = FakeCudaMemory()

    result = benchmark_voxcpm.run_benchmark(
        model_loader=lambda: model,
        texts=("first", "second"),
        iterations=2,
        warmup_runs=1,
        generation_options={"cfg_value": 2.0, "inference_timesteps": 10, "seed": 42},
        clock=clock,
        memory=memory,
    )

    assert result["model_load_seconds"] == pytest.approx(2.0)
    assert result["summary"] == {
        "runs": 4,
        "generated_audio_seconds": pytest.approx(4.0),
        "generation_seconds": pytest.approx(2.0),
        "mean_rtf": pytest.approx(0.5),
        "audio_seconds_per_second": pytest.approx(2.0),
        "utterances_per_second": pytest.approx(2.0),
        "peak_cuda_memory_mb": pytest.approx(256.0),
    }
    assert [run["text"] for run in result["runs"]] == ["first", "second", "first", "second"]
    assert all(run["rtf"] == pytest.approx(0.5) for run in result["runs"])
    assert memory.reset_calls == 1
    assert memory.bound_model is model
    assert [call["text"] for call in model.calls[:2]] == ["first", "second"]
    assert len(model.calls) == 6


@pytest.mark.parametrize(
    ("sample_rate", "samples", "clock_values", "message"),
    [
        (0, SAMPLE_RATE, [0.0, 1.0, 2.0, 3.0], "sample rate"),
        (SAMPLE_RATE, 0, [0.0, 1.0, 2.0, 3.0], "empty audio"),
        (SAMPLE_RATE, SAMPLE_RATE, [0.0, 1.0, 2.0, 2.0], "elapsed time"),
    ],
)
def test_run_benchmark_rejects_invalid_measurements(sample_rate, samples, clock_values, message):
    model = FakeModel()
    model.tts_model = type("TTSModel", (), {"sample_rate": sample_rate})()
    model.generate = lambda **kwargs: [0.0] * samples

    with pytest.raises(benchmark_voxcpm.BenchmarkError, match=message):
        benchmark_voxcpm.run_benchmark(
            model_loader=lambda: model,
            texts=("test",),
            iterations=1,
            warmup_runs=0,
            generation_options={},
            clock=iter(clock_values).__next__,
            memory=FakeCudaMemory(),
        )


def test_main_reads_prompt_file_and_writes_machine_readable_result(monkeypatch, tmp_path):
    input_path = tmp_path / "prompts.txt"
    input_path.write_text("hello\n\nworld\n", encoding="utf-8")
    output_path = tmp_path / "result.json"
    recorded = {}

    def fake_run_benchmark(**kwargs):
        recorded.update(kwargs)
        kwargs["model_loader"]()
        return {
            "model_load_seconds": 1.25,
            "runtime": {
                "device": "cpu",
                "optimization": {
                    "state": "none",
                    "components": {
                        "base_lm": False,
                        "residual_lm": False,
                        "feature_encoder": False,
                        "feature_decoder": False,
                    },
                },
            },
            "summary": {"runs": 2},
            "runs": [],
        }

    def fake_load_model(**kwargs):
        recorded["model_options"] = kwargs
        return FakeModel()

    monkeypatch.setattr(benchmark_voxcpm, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(benchmark_voxcpm, "load_model", fake_load_model)
    monkeypatch.setattr(
        benchmark_voxcpm,
        "resolve_model_source",
        lambda model_id, revision: ("/cache/snapshots/abc123", "abc123"),
    )

    exit_code = benchmark_voxcpm.main(
        [
            "--input-file",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            "local/model",
            "--device",
            "cpu",
            "--iterations",
            "2",
            "--warmup-runs",
            "0",
            "--no-optimize",
            "--denoiser",
            "--cfg-value",
            "3.0",
            "--inference-timesteps",
            "12",
            "--seed",
            "7",
        ]
    )

    assert exit_code == 0
    assert recorded["texts"] == ("hello", "world")
    assert recorded["iterations"] == 2
    assert recorded["warmup_runs"] == 0
    assert recorded["generation_options"] == {
        "cfg_value": 3.0,
        "inference_timesteps": 12,
        "seed": 7,
    }
    assert recorded["model_options"] == {
        "model_id": "/cache/snapshots/abc123",
        "device": "cpu",
        "load_denoiser": True,
        "optimize": False,
    }
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["runs"] == 2
    assert payload["config"] == {
        "model": "local/model",
        "model_revision": None,
        "model_commit": "abc123",
        "device": "cpu",
        "device_actual": "cpu",
        "device_requested": "cpu",
        "iterations": 2,
        "warmup_runs": 0,
        "denoiser": True,
        "optimize_requested": False,
        "optimize_attempted": False,
        "optimization_wrapper_state": "none",
        "cfg_value": 3.0,
        "inference_timesteps": 12,
        "seed": 7,
    }


def test_main_reports_input_errors_without_loading_model(capsys):
    exit_code = benchmark_voxcpm.main(["--text", "   "])

    assert exit_code == 2
    assert "non-empty prompt" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("iterations", "warmup_runs", "message"),
    [(0, 0, "Iterations"), (1, -1, "Warm-up")],
)
def test_run_benchmark_rejects_invalid_run_counts(iterations, warmup_runs, message):
    with pytest.raises(benchmark_voxcpm.BenchmarkError, match=message):
        benchmark_voxcpm.run_benchmark(
            model_loader=FakeModel,
            texts=("test",),
            iterations=iterations,
            warmup_runs=warmup_runs,
            generation_options={},
            memory=FakeCudaMemory(),
        )


def test_run_benchmark_wraps_model_load_and_generation_errors():
    def fail_load():
        raise OSError("checkpoint unavailable")

    with pytest.raises(benchmark_voxcpm.BenchmarkError, match="Unable to load model"):
        benchmark_voxcpm.run_benchmark(
            model_loader=fail_load,
            texts=("test",),
            iterations=1,
            warmup_runs=0,
            generation_options={},
            clock=iter([0.0]).__next__,
            memory=FakeCudaMemory(),
        )

    model = FakeModel()
    model.generate = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("generation failed"))
    with pytest.raises(benchmark_voxcpm.BenchmarkError, match="prompt 1"):
        benchmark_voxcpm.run_benchmark(
            model_loader=lambda: model,
            texts=("test",),
            iterations=1,
            warmup_runs=0,
            generation_options={},
            clock=iter([0.0, 1.0, 2.0]).__next__,
            memory=FakeCudaMemory(),
        )

    with pytest.raises(benchmark_voxcpm.BenchmarkError, match="Warm-up generation"):
        benchmark_voxcpm.run_benchmark(
            model_loader=lambda: model,
            texts=("test",),
            iterations=1,
            warmup_runs=1,
            generation_options={},
            clock=iter([0.0, 1.0]).__next__,
            memory=FakeCudaMemory(),
        )


def test_cuda_memory_tracks_available_cuda_and_noops_without_it():
    class FakeCuda:
        def __init__(self, available):
            self.available = available
            self.synchronize_calls = []
            self.reset_calls = []
            self.max_calls = []

        def is_available(self):
            return self.available

        def synchronize(self, device=None):
            self.synchronize_calls.append(device)

        def reset_peak_memory_stats(self, device=None):
            self.reset_calls.append(device)

        def max_memory_allocated(self, device=None):
            self.max_calls.append(device)
            return 512 * MEBIBYTE

    available_cuda = FakeCuda(True)
    available_memory = benchmark_voxcpm.CudaMemory(types.SimpleNamespace(cuda=available_cuda))
    available_memory.bind(types.SimpleNamespace(tts_model=types.SimpleNamespace(device="cuda:1")))
    available_memory.synchronize()
    available_memory.reset()

    assert available_memory.available is True
    assert available_memory.peak_megabytes() == 512.0
    assert available_cuda.synchronize_calls == ["cuda:1", "cuda:1", "cuda:1"]
    assert available_cuda.reset_calls == ["cuda:1"]
    assert available_cuda.max_calls == ["cuda:1"]

    cpu_on_cuda = FakeCuda(True)
    unavailable_memory = benchmark_voxcpm.CudaMemory(types.SimpleNamespace(cuda=cpu_on_cuda))
    unavailable_memory.bind(types.SimpleNamespace(tts_model=types.SimpleNamespace(device="cpu")))
    unavailable_memory.synchronize()
    unavailable_memory.reset()

    assert unavailable_memory.available is False
    assert unavailable_memory.peak_megabytes() is None
    assert cpu_on_cuda.synchronize_calls == []
    assert cpu_on_cuda.reset_calls == []
    assert cpu_on_cuda.max_calls == []


@pytest.mark.parametrize(
    ("device", "requested", "expected"),
    [
        ("cuda", True, True),
        ("cuda:1", True, False),
        ("cpu", True, False),
        ("cuda", False, False),
    ],
)
def test_optimization_attempt_only_enables_supported_cuda_device(device, requested, expected):
    assert benchmark_voxcpm._optimization_attempt(device=device, requested=requested) is expected


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_resolve_device_expands_auto_in_runtime_order(cuda_available, mps_available, expected):
    torch_module = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps_available),
        ),
    )

    assert benchmark_voxcpm._resolve_device(torch_module, "auto") == expected
    assert benchmark_voxcpm._resolve_device(torch_module, "cuda:1") == "cuda:1"
    assert benchmark_voxcpm._resolve_device(torch_module, " CUDA:1 ") == "cuda:1"
    with pytest.raises(benchmark_voxcpm.BenchmarkError, match="Device cannot be empty"):
        benchmark_voxcpm._resolve_device(torch_module, "  ")


def test_load_model_forwards_options_and_wraps_provider_error(monkeypatch):
    calls = {}

    class StubVoxCPM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs
            return "model"

    module = types.SimpleNamespace(VoxCPM=StubVoxCPM)
    monkeypatch.setitem(sys.modules, "voxcpm", module)
    assert (
        benchmark_voxcpm.load_model(
            model_id="model/id",
            device="cuda:1",
            load_denoiser=True,
            optimize=False,
        )
        == "model"
    )
    assert calls == {
        "args": ("model/id",),
        "kwargs": {
            "device": "cuda:1",
            "load_denoiser": True,
            "optimize": False,
        },
    }

    class BrokenVoxCPM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise OSError("download failed")

    module = types.SimpleNamespace(VoxCPM=BrokenVoxCPM)
    monkeypatch.setitem(sys.modules, "voxcpm", module)
    with pytest.raises(benchmark_voxcpm.BenchmarkError, match="model/id"):
        benchmark_voxcpm.load_model(
            model_id="model/id",
            device="cpu",
            load_denoiser=False,
            optimize=True,
        )


@pytest.mark.parametrize(
    ("available", "device", "expected_name", "expected_calls"),
    [(True, "cuda:1", "Second GPU", ["cuda:1"]), (True, "cpu", None, []), (False, "cuda:1", None, [])],
)
def test_describe_environment_reports_selected_cuda_device(
    monkeypatch, available, device, expected_name, expected_calls
):
    requested_devices = []
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        get_device_name=lambda selected_device: requested_devices.append(selected_device) or "Second GPU",
    )
    torch_module = types.SimpleNamespace(
        cuda=cuda,
        version=types.SimpleNamespace(cuda="12.4"),
        __version__="2.5.0",
    )
    monkeypatch.setattr(benchmark_voxcpm.platform, "processor", lambda: "Test Processor")

    environment = benchmark_voxcpm.describe_environment(torch_module, device)

    assert environment["torch"] == "2.5.0"
    assert environment["cuda"] == "12.4"
    assert environment["cuda_device"] == expected_name
    assert environment["processor"] == "Test Processor"
    assert requested_devices == expected_calls
    assert environment["platform"]
    assert environment["python"]


def test_describe_environment_falls_back_to_machine_identifier(monkeypatch):
    monkeypatch.setattr(benchmark_voxcpm.platform, "processor", lambda: "")
    monkeypatch.setattr(benchmark_voxcpm.platform, "machine", lambda: "arm64")
    torch_module = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        version=types.SimpleNamespace(cuda=None),
        __version__="2.5.0",
    )

    assert benchmark_voxcpm.describe_environment(torch_module, "mps")["processor"] == "arm64"


def test_optimization_status_reports_full_partial_and_eager_components():
    compiled_function = types.SimpleNamespace(_torchdynamo_orig_callable=object())
    compiled_module = types.SimpleNamespace(_orig_mod=object())
    tts_model = types.SimpleNamespace(
        base_lm=types.SimpleNamespace(forward_step=compiled_function),
        residual_lm=types.SimpleNamespace(forward_step=compiled_function),
        feat_encoder=compiled_module,
        feat_decoder=types.SimpleNamespace(estimator=compiled_module),
    )
    model = types.SimpleNamespace(tts_model=tts_model)

    expected_components = {
        "base_lm": True,
        "residual_lm": True,
        "feature_encoder": True,
        "feature_decoder": True,
    }
    assert benchmark_voxcpm._optimization_status(model) == {
        "state": "full",
        "components": expected_components,
    }

    tts_model.feat_decoder.estimator = object()
    expected_components["feature_decoder"] = False
    assert benchmark_voxcpm._optimization_status(model) == {
        "state": "partial",
        "components": expected_components,
    }

    assert benchmark_voxcpm._optimization_status(types.SimpleNamespace(tts_model=object())) == {
        "state": "none",
        "components": {
            "base_lm": False,
            "residual_lm": False,
            "feature_encoder": False,
            "feature_decoder": False,
        },
    }


def test_resolve_model_source_records_hub_commit_and_preserves_local_path(monkeypatch, tmp_path):
    commit = "a" * 40
    snapshot_path = tmp_path / "models--openbmb--VoxCPM2" / "snapshots" / commit
    snapshot_path.mkdir(parents=True)
    calls = []
    hub = types.SimpleNamespace(
        snapshot_download=lambda **kwargs: calls.append(kwargs) or str(snapshot_path),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    assert benchmark_voxcpm.resolve_model_source("openbmb/VoxCPM2", "release") == (str(snapshot_path), commit)
    assert calls == [{"repo_id": "openbmb/VoxCPM2", "revision": "release"}]

    local_model = tmp_path / "local-model"
    local_model.mkdir()
    assert benchmark_voxcpm.resolve_model_source(str(local_model), None) == (str(local_model), None)
    assert len(calls) == 1

    with pytest.raises(benchmark_voxcpm.BenchmarkError, match="local model.*revision"):
        benchmark_voxcpm.resolve_model_source(str(local_model), "ignored")

    hub.snapshot_download = lambda **kwargs: str(tmp_path / "unresolved-model")
    with pytest.raises(benchmark_voxcpm.BenchmarkError, match="immutable commit"):
        benchmark_voxcpm.resolve_model_source("openbmb/VoxCPM2", None)

    def fail_download(**kwargs):
        raise OSError("Hub unavailable")

    hub.snapshot_download = fail_download
    with pytest.raises(benchmark_voxcpm.BenchmarkError, match="Unable to resolve model"):
        benchmark_voxcpm.resolve_model_source("openbmb/VoxCPM2", None)


def test_argument_validators_reject_out_of_range_values():
    with pytest.raises(argparse.ArgumentTypeError, match="at least one"):
        benchmark_voxcpm._positive_integer("0")
    with pytest.raises(argparse.ArgumentTypeError, match="negative"):
        benchmark_voxcpm._non_negative_integer("-1")


def test_prompt_file_and_output_boundaries_are_reported(monkeypatch, tmp_path, capsys):
    args = benchmark_voxcpm._build_parser().parse_args(["--input-file", str(tmp_path / "missing.txt")])
    with pytest.raises(benchmark_voxcpm.BenchmarkError, match="Unable to read prompt file"):
        benchmark_voxcpm._read_texts(args)

    invalid_utf8_path = tmp_path / "invalid-utf8.txt"
    invalid_utf8_path.write_bytes(b"\xff")
    args = benchmark_voxcpm._build_parser().parse_args(["--input-file", str(invalid_utf8_path)])
    with pytest.raises(benchmark_voxcpm.BenchmarkError, match="Unable to read prompt file"):
        benchmark_voxcpm._read_texts(args)

    benchmark_voxcpm._write_result({"status": "ok"}, None)
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}

    def fail_write(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(
        benchmark_voxcpm.BenchmarkError,
        match="Unable to write benchmark result",
    ):
        benchmark_voxcpm._write_result({"status": "ok"}, tmp_path / "result.json")


def test_script_entry_point_returns_main_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(BENCHMARK_PATH), "--text", " "])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(BENCHMARK_PATH), run_name="__main__")

    assert exc_info.value.code == 2
    assert "non-empty prompt" in capsys.readouterr().err
