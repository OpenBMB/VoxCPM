#!/usr/bin/env python3
"""Measure VoxCPM model load and text-to-speech generation performance."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

DEFAULT_MODEL_ID = "openbmb/VoxCPM2"
DEFAULT_DEVICE = "auto"
CUDA_DEVICE = "cuda"
MPS_DEVICE = "mps"
CPU_DEVICE = "cpu"
DEFAULT_ITERATIONS = 1
DEFAULT_WARMUP_RUNS = 1
DEFAULT_CFG_VALUE = 2.0
DEFAULT_INFERENCE_TIMESTEPS = 10
DEFAULT_SEED = 42
DEFAULT_TEXTS = (
    "VoxCPM turns written language into natural speech.",
    "This second sentence measures throughput across multiple prompts.",
)
BYTES_PER_MEBIBYTE = 1024 * 1024
EXIT_SUCCESS = 0
EXIT_USAGE_ERROR = 2
JSON_INDENT = 2
SNAPSHOTS_DIRECTORY = "snapshots"
COMPILED_FUNCTION_ATTRIBUTE = "_torchdynamo_orig_callable"
COMPILED_MODULE_ATTRIBUTE = "_orig_mod"
OPTIMIZATION_NONE = "none"
OPTIMIZATION_PARTIAL = "partial"
OPTIMIZATION_FULL = "full"


class BenchmarkError(RuntimeError):
    """Raised when a benchmark cannot produce a trustworthy measurement."""


class CudaMemory:
    """Synchronize CUDA timing and expose peak allocated model memory."""

    def __init__(self, torch_module: Any):
        self._torch = torch_module
        self._device: str | None = None

    def bind(self, model: Any) -> None:
        """Select the CUDA device actually used by the loaded model."""
        resolved_device = str(getattr(getattr(model, "tts_model", None), "device", ""))
        self._device = resolved_device if resolved_device.startswith(CUDA_DEVICE) else None

    @property
    def available(self) -> bool:
        return self._device is not None and bool(self._torch.cuda.is_available())

    def synchronize(self) -> None:
        if self.available:
            self._torch.cuda.synchronize(device=self._device)

    def reset(self) -> None:
        if self.available:
            self.synchronize()
            self._torch.cuda.reset_peak_memory_stats(device=self._device)

    def peak_megabytes(self) -> float | None:
        if not self.available:
            return None
        self.synchronize()
        return self._torch.cuda.max_memory_allocated(device=self._device) / BYTES_PER_MEBIBYTE


def _validated_texts(texts: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(text.strip() for text in texts if text.strip())
    if not normalized:
        raise BenchmarkError("Provide at least one non-empty prompt.")
    return normalized


def run_benchmark(
    *,
    model_loader: Callable[[], Any],
    texts: Iterable[str],
    iterations: int,
    warmup_runs: int,
    generation_options: dict[str, Any],
    clock: Callable[[], float] = time.perf_counter,
    memory: Any,
) -> dict[str, Any]:
    """Run a deterministic sequence of warm-up and measured generations."""
    benchmark_texts = _validated_texts(texts)
    if iterations < 1:
        raise BenchmarkError("Iterations must be at least one.")
    if warmup_runs < 0:
        raise BenchmarkError("Warm-up runs cannot be negative.")

    load_started = clock()
    try:
        model = model_loader()
    except Exception as exc:
        raise BenchmarkError(f"Unable to load model: {exc}") from exc
    memory.bind(model)
    memory.synchronize()
    model_load_seconds = clock() - load_started

    sample_rate = getattr(getattr(model, "tts_model", None), "sample_rate", 0)
    if sample_rate <= 0:
        raise BenchmarkError("The loaded model reported an invalid sample rate.")

    for _ in range(warmup_runs):
        for warmup_text in benchmark_texts:
            try:
                model.generate(text=warmup_text, **generation_options)
            except Exception as exc:
                raise BenchmarkError(f"Warm-up generation failed: {exc}") from exc

    memory.reset()
    runs = []
    for iteration in range(iterations):
        for text_index, text in enumerate(benchmark_texts):
            memory.synchronize()
            generation_started = clock()
            try:
                audio = model.generate(text=text, **generation_options)
            except Exception as exc:
                raise BenchmarkError(f"Generation failed for prompt {text_index + 1}: {exc}") from exc
            memory.synchronize()
            generation_seconds = clock() - generation_started
            if generation_seconds <= 0:
                raise BenchmarkError("Generation elapsed time must be positive.")

            sample_count = len(audio)
            if sample_count <= 0:
                raise BenchmarkError("The model generated empty audio.")
            audio_seconds = sample_count / sample_rate
            runs.append(
                {
                    "iteration": iteration + 1,
                    "prompt": text_index + 1,
                    "text": text,
                    "generation_seconds": generation_seconds,
                    "audio_seconds": audio_seconds,
                    "rtf": generation_seconds / audio_seconds,
                }
            )

    total_generation_seconds = sum(run["generation_seconds"] for run in runs)
    total_audio_seconds = sum(run["audio_seconds"] for run in runs)
    run_count = len(runs)
    return {
        "model_load_seconds": model_load_seconds,
        "runtime": {
            "device": str(getattr(model.tts_model, "device", "unknown")),
            "optimization": _optimization_status(model),
        },
        "summary": {
            "runs": run_count,
            "generated_audio_seconds": total_audio_seconds,
            "generation_seconds": total_generation_seconds,
            "mean_rtf": sum(run["rtf"] for run in runs) / run_count,
            "audio_seconds_per_second": total_audio_seconds / total_generation_seconds,
            "utterances_per_second": run_count / total_generation_seconds,
            "peak_cuda_memory_mb": memory.peak_megabytes(),
        },
        "runs": runs,
    }


def load_model(*, model_id: str, device: str, load_denoiser: bool, optimize: bool) -> Any:
    """Load VoxCPM without adding imports to benchmark discovery."""
    try:
        from voxcpm import VoxCPM

        return VoxCPM.from_pretrained(
            model_id,
            device=device,
            load_denoiser=load_denoiser,
            optimize=optimize,
        )
    except Exception as exc:
        raise BenchmarkError(f"Unable to initialize {model_id!r}: {exc}") from exc


def resolve_model_source(model_id: str, revision: str | None) -> tuple[str, str | None]:
    """Download a Hub snapshot once and return its immutable commit identifier."""
    model_path = Path(model_id)
    if model_path.is_dir():
        if revision is not None:
            raise BenchmarkError("A local model directory cannot be combined with --revision.")
        return str(model_path), None
    try:
        from huggingface_hub import snapshot_download

        snapshot_path = Path(snapshot_download(repo_id=model_id, revision=revision))
    except Exception as exc:
        raise BenchmarkError(f"Unable to resolve model {model_id!r}: {exc}") from exc

    if snapshot_path.parent.name != SNAPSHOTS_DIRECTORY:
        raise BenchmarkError(f"Unable to determine the immutable commit for model {model_id!r}.")
    return str(snapshot_path), snapshot_path.name


def describe_environment(torch_module: Any, device: str) -> dict[str, Any]:
    """Collect enough environment metadata to compare benchmark reports."""
    cuda_available = bool(torch_module.cuda.is_available())
    selected_cuda = cuda_available and device.startswith(CUDA_DEVICE)
    device_name = torch_module.cuda.get_device_name(device) if selected_cuda else None
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch_module.__version__,
        "cuda": torch_module.version.cuda,
        "cuda_device": device_name,
    }


def _resolve_device(torch_module: Any, requested_device: str) -> str:
    """Resolve ``auto`` using VoxCPM's CUDA, MPS, then CPU preference."""
    normalized_device = requested_device.strip().lower()
    if not normalized_device:
        raise BenchmarkError("Device cannot be empty.")
    if normalized_device != DEFAULT_DEVICE:
        return normalized_device
    if torch_module.cuda.is_available():
        return CUDA_DEVICE
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return MPS_DEVICE
    return CPU_DEVICE


def _optimization_attempt(*, device: str, requested: bool) -> bool:
    """Attempt compilation only where VoxCPM supports it."""
    return requested and device == CUDA_DEVICE


def _optimization_status(model: Any) -> dict[str, Any]:
    """Report compile wrappers per component, including partially optimized models."""
    tts_model = getattr(model, "tts_model", None)
    base_lm = getattr(tts_model, "base_lm", None)
    residual_lm = getattr(tts_model, "residual_lm", None)
    feature_decoder = getattr(tts_model, "feat_decoder", None)
    components = {
        "base_lm": hasattr(getattr(base_lm, "forward_step", None), COMPILED_FUNCTION_ATTRIBUTE),
        "residual_lm": hasattr(getattr(residual_lm, "forward_step", None), COMPILED_FUNCTION_ATTRIBUTE),
        "feature_encoder": hasattr(getattr(tts_model, "feat_encoder", None), COMPILED_MODULE_ATTRIBUTE),
        "feature_decoder": hasattr(getattr(feature_decoder, "estimator", None), COMPILED_MODULE_ATTRIBUTE),
    }
    detected_count = sum(components.values())
    if detected_count == len(components):
        state = OPTIMIZATION_FULL
    elif detected_count:
        state = OPTIMIZATION_PARTIAL
    else:
        state = OPTIMIZATION_NONE
    return {"state": state, "components": components}


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--text", action="append", help="Prompt to benchmark; repeat for a batch")
    prompt_group.add_argument("--input-file", type=Path, help="UTF-8 file containing one prompt per line")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hugging Face model ID or local model directory")
    parser.add_argument("--revision", help="Hugging Face revision; the resolved snapshot commit is recorded")
    parser.add_argument(
        "--device", default=DEFAULT_DEVICE, help="Runtime device such as auto, cuda, cuda:0, mps, or cpu"
    )
    parser.add_argument("--iterations", type=_positive_integer, default=DEFAULT_ITERATIONS)
    parser.add_argument("--warmup-runs", type=_non_negative_integer, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--cfg-value", type=float, default=DEFAULT_CFG_VALUE)
    parser.add_argument("--inference-timesteps", type=_positive_integer, default=DEFAULT_INFERENCE_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--denoiser", action="store_true", help="Include the optional denoiser in model loading")
    parser.add_argument("--no-optimize", action="store_true", help="Disable the default compiled inference path")
    parser.add_argument("--output", type=Path, help="Write JSON here instead of standard output")
    return parser


def _read_texts(args: argparse.Namespace) -> tuple[str, ...]:
    if args.input_file is not None:
        try:
            return _validated_texts(args.input_file.read_text(encoding="utf-8").splitlines())
        except (OSError, UnicodeError) as exc:
            raise BenchmarkError(f"Unable to read prompt file {args.input_file}: {exc}") from exc
    return _validated_texts(args.text or DEFAULT_TEXTS)


def _write_result(result: dict[str, Any], output: Path | None) -> None:
    payload = json.dumps(result, indent=JSON_INDENT, sort_keys=True) + "\n"
    if output is None:
        print(payload, end="")
        return
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    except OSError as exc:
        raise BenchmarkError(f"Unable to write benchmark result {output}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        import torch

        texts = _read_texts(args)
        resolved_device = _resolve_device(torch, args.device)
        optimize_requested = not args.no_optimize
        optimize_attempted = _optimization_attempt(device=resolved_device, requested=optimize_requested)
        resolved_model, model_commit = resolve_model_source(args.model, args.revision)
        memory = CudaMemory(torch)
        result = run_benchmark(
            model_loader=lambda: load_model(
                model_id=resolved_model,
                device=resolved_device,
                load_denoiser=args.denoiser,
                optimize=optimize_attempted,
            ),
            texts=texts,
            iterations=args.iterations,
            warmup_runs=args.warmup_runs,
            generation_options={
                "cfg_value": args.cfg_value,
                "inference_timesteps": args.inference_timesteps,
                "seed": args.seed,
            },
            memory=memory,
        )
        actual_device = result["runtime"]["device"]
        optimization = result["runtime"]["optimization"]
        result["environment"] = describe_environment(torch, actual_device)
        result["config"] = {
            "model": args.model,
            "model_revision": args.revision,
            "model_commit": model_commit,
            "device": resolved_device,
            "device_actual": actual_device,
            "device_requested": args.device,
            "iterations": args.iterations,
            "warmup_runs": args.warmup_runs,
            "denoiser": args.denoiser,
            "optimize_requested": optimize_requested,
            "optimize_attempted": optimize_attempted,
            "optimization_wrapper_state": optimization["state"],
            "cfg_value": args.cfg_value,
            "inference_timesteps": args.inference_timesteps,
            "seed": args.seed,
        }
        _write_result(result, args.output)
    except BenchmarkError as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
