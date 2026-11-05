#!/usr/bin/env python3
"""
VoxCPM Command Line Interface

VoxCPM2-first CLI for voice design, cloning, and batch processing.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from voxcpm.timestamps import align_audio_file

DEFAULT_HF_MODEL_ID = "openbmb/VoxCPM2"

# -----------------------------
# Validators
# -----------------------------


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def validate_file_exists(file_path: str, file_type: str = "file") -> Path:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_type} '{file_path}' does not exist")
    return path


def require_file_exists(file_path: str, parser, file_type: str = "file") -> Path:
    try:
        return validate_file_exists(file_path, file_type)
    except FileNotFoundError as exc:
        parser.error(str(exc))


def validate_output_path(output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def validate_ranges(args, parser):
    """Validate numeric argument ranges."""
    if not (0.1 <= args.cfg_value <= 10.0):
        parser.error("--cfg-value must be between 0.1 and 10.0 (recommended: 1.0–3.0)")

    if not (1 <= args.inference_timesteps <= 100):
        parser.error("--inference-timesteps must be between 1 and 100 (recommended: 4–30)")

    if args.lora_r <= 0:
        parser.error("--lora-r must be a positive integer")

    if args.lora_alpha <= 0:
        parser.error("--lora-alpha must be a positive integer")

    if not (0.0 <= args.lora_dropout <= 1.0):
        parser.error("--lora-dropout must be between 0.0 and 1.0")


def warn_legacy_mode():
    print(
        "Warning: legacy root CLI arguments are deprecated. Prefer `voxcpm design|clone|batch ...`.",
        file=sys.stderr,
    )


def build_final_text(text: str, control: str | None) -> str:
    control = (control or "").strip()
    return f"({control}){text}" if control else text


def resolve_prompt_text(args, parser) -> str | None:
    prompt_text = getattr(args, "prompt_text", None)
    prompt_file = getattr(args, "prompt_file", None)

    if prompt_text and prompt_file:
        parser.error("Use either --prompt-text or --prompt-file, not both.")

    if prompt_file:
        prompt_path = require_file_exists(prompt_file, parser, "prompt text file")
        return prompt_path.read_text(encoding="utf-8").strip()

    if prompt_text:
        return prompt_text.strip()

    return None


def detect_model_architecture(args) -> str | None:
    model_location = getattr(args, "model_path", None) or getattr(args, "hf_model_id", None)
    if not model_location:
        return None

    if os.path.isdir(model_location):
        config_path = Path(model_location) / "config.json"
        if not config_path.exists():
            return None

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f).get("architecture", "voxcpm").lower()

    model_hint = str(model_location).lower()
    if "voxcpm2" in model_hint:
        return "voxcpm2"
    if "voxcpm1.5" in model_hint or "voxcpm-1.5" in model_hint or "voxcpm_1.5" in model_hint:
        return "voxcpm"

    return None


def validate_prompt_related_args(args, parser, prompt_text: str | None):
    if prompt_text and not args.prompt_audio:
        parser.error("--prompt-text/--prompt-file requires --prompt-audio.")

    if args.prompt_audio and not prompt_text:
        parser.error("--prompt-audio requires --prompt-text or --prompt-file.")

    if args.control and prompt_text:
        parser.error("--control cannot be used together with --prompt-text or --prompt-file.")


def validate_reference_support(args, parser):
    if not getattr(args, "reference_audio", None):
        return

    arch = detect_model_architecture(args)
    if arch == "voxcpm":
        parser.error("--reference-audio is only supported with VoxCPM2 models.")


def validate_design_args(args, parser):
    prompt_text = resolve_prompt_text(args, parser)
    if args.prompt_audio or args.reference_audio or prompt_text:
        parser.error("`design` does not accept prompt/reference audio. Use `clone` instead.")


def validate_clone_args(args, parser):
    prompt_text = resolve_prompt_text(args, parser)
    validate_prompt_related_args(args, parser, prompt_text)
    validate_reference_support(args, parser)

    if not args.prompt_audio and not args.reference_audio:
        parser.error("`clone` requires --reference-audio, or --prompt-audio with --prompt-text/--prompt-file.")

    return prompt_text


def validate_batch_args(args, parser):
    prompt_text = resolve_prompt_text(args, parser)
    validate_prompt_related_args(args, parser, prompt_text)
    validate_reference_support(args, parser)
    return prompt_text


# -----------------------------
# Model loading
# -----------------------------


def load_model(args):
    from voxcpm.core import VoxCPM

    print("Loading VoxCPM model...", file=sys.stderr)

    zipenhancer_path = getattr(args, "zipenhancer_path", None) or os.environ.get("ZIPENHANCER_MODEL_PATH", None)

    # Build LoRA config if provided
    lora_config = None
    lora_weights_path = getattr(args, "lora_path", None)
    if lora_weights_path:
        from voxcpm.model.voxcpm import LoRAConfig

        lora_config = LoRAConfig(
            enable_lm=not args.lora_disable_lm,
            enable_dit=not args.lora_disable_dit,
            enable_proj=args.lora_enable_proj,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

        print(
            f"LoRA config: r={lora_config.r}, alpha={lora_config.alpha}, "
            f"lm={lora_config.enable_lm}, dit={lora_config.enable_dit}, proj={lora_config.enable_proj}",
            file=sys.stderr,
        )

    # Load local model if specified
    if args.model_path:
        try:
            model = VoxCPM(
                voxcpm_model_path=args.model_path,
                zipenhancer_model_path=zipenhancer_path,
                enable_denoiser=not args.no_denoiser,
                optimize=not args.no_optimize,
                device=args.device,
                lora_config=lora_config,
                lora_weights_path=lora_weights_path,
                attention_backend=getattr(args, "attention_backend", "auto"),
                max_cache_length=getattr(args, "max_cache_length", None),
            )
            print("Model loaded (local).", file=sys.stderr)
            return model
        except Exception as e:
            print(f"Failed to load model (local): {e}", file=sys.stderr)
            sys.exit(1)

    # Load from Hugging Face Hub
    try:
        model = VoxCPM.from_pretrained(
            hf_model_id=args.hf_model_id,
            load_denoiser=not args.no_denoiser,
            zipenhancer_model_id=zipenhancer_path,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            optimize=not args.no_optimize,
            device=args.device,
            lora_config=lora_config,
            lora_weights_path=lora_weights_path,
            attention_backend=getattr(args, "attention_backend", "auto"),
            max_cache_length=getattr(args, "max_cache_length", None),
        )
        print("Model loaded (from_pretrained).", file=sys.stderr)
        return model
    except Exception as e:
        print(f"Failed to load model (from_pretrained): {e}", file=sys.stderr)
        sys.exit(1)


# -----------------------------
# Commands
# -----------------------------


def _run_single(args, parser, *, text: str, output: str, prompt_text: str | None):
    output_path = validate_output_path(output)

    if args.prompt_audio:
        require_file_exists(args.prompt_audio, parser, "prompt audio file")
    if args.reference_audio:
        require_file_exists(args.reference_audio, parser, "reference audio file")

    model = load_model(args)

    audio_array = model.generate(
        text=text,
        prompt_wav_path=args.prompt_audio,
        prompt_text=prompt_text,
        reference_wav_path=args.reference_audio,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        normalize=args.normalize,
        denoise=args.denoise and (args.prompt_audio is not None or args.reference_audio is not None),
        seed=args.seed,
    )

    import soundfile as sf

    sf.write(str(output_path), audio_array, model.tts_model.sample_rate)

    duration = len(audio_array) / model.tts_model.sample_rate
    print(f"Saved audio to: {output_path} ({duration:.2f}s)", file=sys.stderr)
    maybe_write_timestamps(
        args,
        text=text,
        audio_path=output_path,
        sample_rate=model.tts_model.sample_rate,
    )


def cmd_design(args, parser):
    validate_design_args(args, parser)
    final_text = build_final_text(args.text, args.control)
    return _run_single(args, parser, text=final_text, output=args.output, prompt_text=None)


def cmd_clone(args, parser):
    prompt_text = validate_clone_args(args, parser)
    final_text = build_final_text(args.text, args.control)
    return _run_single(args, parser, text=final_text, output=args.output, prompt_text=prompt_text)


def cmd_validate(args, parser):
    from voxcpm.training.validate import (
        print_validation_report,
        validate_manifest,
    )

    manifest = str(require_file_exists(args.manifest, parser, "manifest file"))
    result = validate_manifest(
        manifest_path=manifest,
        sample_rate=args.sample_rate,
        max_samples=args.max_samples,
        verbose=args.verbose,
    )
    print_validation_report(result, manifest)
    if not result.is_valid:
        sys.exit(1)


def cmd_batch(args, parser):
    input_file = require_file_exists(args.input, parser, "input file")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    if not texts:
        sys.exit("Error: Input file is empty")

    prompt_text = validate_batch_args(args, parser)
    model = load_model(args)

    import soundfile as sf

    prompt_audio_path = None
    if args.prompt_audio:
        prompt_audio_path = str(require_file_exists(args.prompt_audio, parser, "prompt audio file"))

    reference_audio_path = None
    if args.reference_audio:
        reference_audio_path = str(require_file_exists(args.reference_audio, parser, "reference audio file"))

    success_count = 0

    for i, text in enumerate(texts, 1):
        try:
            final_text = build_final_text(text, args.control)
            audio_array = model.generate(
                text=final_text,
                prompt_wav_path=prompt_audio_path,
                prompt_text=prompt_text,
                reference_wav_path=reference_audio_path,
                cfg_value=args.cfg_value,
                inference_timesteps=args.inference_timesteps,
                normalize=args.normalize,
                denoise=args.denoise and (prompt_audio_path is not None or reference_audio_path is not None),
                seed=args.seed,
            )

            output_file = output_dir / f"output_{i:03d}.wav"
            sf.write(str(output_file), audio_array, model.tts_model.sample_rate)

            duration = len(audio_array) / model.tts_model.sample_rate
            print(f"Saved: {output_file} ({duration:.2f}s)", file=sys.stderr)
            maybe_write_timestamps(
                args,
                text=final_text,
                audio_path=output_file,
                sample_rate=model.tts_model.sample_rate,
            )
            success_count += 1

        except Exception as e:
            print(f"Failed on line {i}: {e}", file=sys.stderr)

    print(f"\nBatch finished: {success_count}/{len(texts)} succeeded", file=sys.stderr)


def default_timestamp_path(audio_path: Path) -> Path:
    return audio_path.with_suffix(".timestamps.json")


def maybe_write_timestamps(args, *, text: str, audio_path: Path, sample_rate: int) -> None:
    if not getattr(args, "timestamps", False):
        return

    timestamp_output = getattr(args, "timestamp_output", None)
    output_path = Path(timestamp_output) if timestamp_output else default_timestamp_path(audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = align_audio_file(
            audio_path=str(audio_path),
            text=text,
            sample_rate=sample_rate,
            backend=args.timestamp_backend,
            level=args.timestamp_level,
            model_name=args.timestamp_model,
            device=args.timestamp_device,
            language=args.timestamp_language,
        )
    except Exception as exc:
        if getattr(args, "timestamp_strict", False):
            raise SystemExit(f"Timestamp alignment failed: {exc}") from exc
        print(f"Warning: Timestamp alignment failed: {exc}", file=sys.stderr)
        return

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Saved timestamps to: {output_path}", file=sys.stderr)


# -----------------------------
# Parser
# -----------------------------


def _add_common_generation_args(parser):
    parser.add_argument("--text", "-t", help="Text to synthesize")
    parser.add_argument(
        "--control",
        type=str,
        help="Control instruction for VoxCPM2 voice design/cloning",
    )
    parser.add_argument(
        "--cfg-value",
        type=float,
        default=2.0,
        help="CFG guidance scale (float, recommended 1.0–3.0, default: 2.0)",
    )
    parser.add_argument(
        "--inference-timesteps",
        type=int,
        default=10,
        help="Inference steps (int, recommended 4–30, default: 10)",
    )
    parser.add_argument("--normalize", action="store_true", help="Enable text normalization")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation (default: None)",
    )


def _add_prompt_reference_args(parser):
    parser.add_argument(
        "--prompt-audio",
        "-pa",
        help="Prompt audio file path (continuation mode, requires --prompt-text or --prompt-file)",
    )
    parser.add_argument("--prompt-text", "-pt", help="Text corresponding to the prompt audio")
    parser.add_argument("--prompt-file", type=str, help="Text file corresponding to the prompt audio")
    parser.add_argument(
        "--reference-audio",
        "-ra",
        help="Reference audio for voice cloning (VoxCPM2 only)",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Enable prompt/reference speech enhancement",
    )


def _add_model_args(parser):
    parser.add_argument("--model-path", type=str, help="Local VoxCPM model path")
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default=DEFAULT_HF_MODEL_ID,
        help=f"Hugging Face repo id (default: {DEFAULT_HF_MODEL_ID})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Runtime device: auto, cpu, mps, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument("--cache-dir", type=str, help="Cache directory for Hub downloads")
    parser.add_argument("--local-files-only", action="store_true", help="Disable network access")
    parser.add_argument("--no-denoiser", action="store_true", help="Disable denoiser model loading")
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable model optimization during loading",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "cudnn", "math"],
        default="auto",
        help="LM decode attention backend (default: auto; cudnn requires CUDA)",
    )
    parser.add_argument(
        "--max-cache-length",
        type=_positive_int,
        default=None,
        help="KV-cache token capacity including text, reference, and generation (default: model config)",
    )
    parser.add_argument(
        "--zipenhancer-path",
        type=str,
        help="ZipEnhancer model id or local path (or env ZIPENHANCER_MODEL_PATH)",
    )


def _add_lora_args(parser):
    parser.add_argument("--lora-path", type=str, help="Path to LoRA weights")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank (positive int, default: 32)")
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (positive int, default: 16)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout rate (0.0–1.0, default: 0.0)",
    )
    parser.add_argument("--lora-disable-lm", action="store_true", help="Disable LoRA on LM layers")
    parser.add_argument("--lora-disable-dit", action="store_true", help="Disable LoRA on DiT layers")
    parser.add_argument(
        "--lora-enable-proj",
        action="store_true",
        help="Enable LoRA on projection layers",
    )


def _add_timestamp_args(parser, *, include_output: bool = True):
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Run post-generation timestamp alignment and write a JSON sidecar file",
    )
    if include_output:
        parser.add_argument(
            "--timestamp-output",
            type=str,
            help="Output timestamp JSON path (default: output audio path with .timestamps.json suffix)",
        )
    parser.add_argument(
        "--timestamp-level",
        choices=["segment", "word", "char"],
        default="word",
        help="Timestamp granularity (default: word; char is best-effort)",
    )
    parser.add_argument(
        "--timestamp-backend",
        choices=["stable-ts"],
        default="stable-ts",
        help="Timestamp alignment backend (default: stable-ts)",
    )
    parser.add_argument(
        "--timestamp-model",
        default="base",
        help="stable-ts Whisper model name (default: base)",
    )
    parser.add_argument(
        "--timestamp-language",
        default=None,
        help="Language hint for timestamp alignment, e.g. zh or en",
    )
    parser.add_argument(
        "--timestamp-device",
        default=None,
        help="Device for timestamp alignment, e.g. cuda or cpu",
    )
    parser.add_argument(
        "--timestamp-strict",
        action="store_true",
        help="Fail the command if timestamp alignment fails",
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        description="VoxCPM CLI - VoxCPM2-first voice design, cloning, and batch processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  voxcpm design --text "Hello world" --output out.wav
  voxcpm design --text "Hello world" --control "warm female voice" --output out.wav
  voxcpm clone --text "Hello" --reference-audio ref.wav --output out.wav
  voxcpm batch --input texts.txt --output-dir ./outs --reference-audio ref.wav
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    design_parser = subparsers.add_parser("design", help="Generate speech with VoxCPM2-first voice design")
    _add_common_generation_args(design_parser)
    _add_prompt_reference_args(design_parser)
    _add_model_args(design_parser)
    _add_lora_args(design_parser)
    _add_timestamp_args(design_parser)
    design_parser.add_argument("--output", "-o", required=True, help="Output audio file path")

    clone_parser = subparsers.add_parser("clone", help="Clone a voice with reference/prompt audio")
    _add_common_generation_args(clone_parser)
    _add_prompt_reference_args(clone_parser)
    _add_model_args(clone_parser)
    _add_lora_args(clone_parser)
    _add_timestamp_args(clone_parser)
    clone_parser.add_argument("--output", "-o", required=True, help="Output audio file path")

    batch_parser = subparsers.add_parser("batch", help="Batch-generate one line per output file")
    batch_parser.add_argument("--input", "-i", required=True, help="Input text file (one text per line)")
    batch_parser.add_argument("--output-dir", "-od", required=True, help="Output directory")
    batch_parser.add_argument(
        "--control",
        type=str,
        help="Control instruction for VoxCPM2 voice design/cloning",
    )
    _add_prompt_reference_args(batch_parser)
    batch_parser.add_argument(
        "--cfg-value",
        type=float,
        default=2.0,
        help="CFG guidance scale (float, recommended 1.0–3.0, default: 2.0)",
    )
    batch_parser.add_argument(
        "--inference-timesteps",
        type=int,
        default=10,
        help="Inference steps (int, recommended 4–30, default: 10)",
    )
    batch_parser.add_argument("--normalize", action="store_true", help="Enable text normalization")
    batch_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation (default: None)",
    )
    _add_model_args(batch_parser)
    _add_lora_args(batch_parser)
    _add_timestamp_args(batch_parser, include_output=False)

    # Validate subcommand
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a training data manifest (JSONL) before fine-tuning",
    )
    validate_parser.add_argument("--manifest", "-m", required=True, help="Path to JSONL training manifest")
    validate_parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="Expected audio sample rate in Hz (default: 16000)",
    )
    validate_parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of samples to validate (0 = all, default: 0)",
    )
    validate_parser.add_argument("--verbose", "-v", action="store_true", help="Print per-sample progress")

    # Legacy root arguments
    parser.add_argument("--input", "-i", help="Input text file (batch mode only)")
    parser.add_argument("--output-dir", "-od", help="Output directory (batch mode only)")
    _add_common_generation_args(parser)
    parser.add_argument("--output", "-o", help="Output audio file path (single or clone mode)")
    _add_prompt_reference_args(parser)
    _add_model_args(parser)
    _add_lora_args(parser)
    _add_timestamp_args(parser)

    return parser


def _dispatch_legacy(args, parser):
    warn_legacy_mode()

    if args.input and args.text:
        parser.error("Use either batch mode (--input) or single mode (--text), not both.")

    if args.input:
        if not args.output_dir:
            parser.error("Batch mode requires --output-dir")
        return cmd_batch(args, parser)

    if not args.text or not args.output:
        parser.error("Single-sample legacy mode requires --text and --output")

    if args.prompt_audio or args.prompt_text or args.prompt_file or args.reference_audio:
        return cmd_clone(args, parser)

    return cmd_design(args, parser)


# -----------------------------
# Entrypoint
# -----------------------------


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "validate":
        return cmd_validate(args, parser)

    validate_ranges(args, parser)

    if args.command == "design":
        if not args.text:
            parser.error("`design` requires --text")
        return cmd_design(args, parser)

    if args.command == "clone":
        if not args.text or not args.output:
            parser.error("`clone` requires --text and --output")
        return cmd_clone(args, parser)

    if args.command == "batch":
        return cmd_batch(args, parser)

    return _dispatch_legacy(args, parser)


if __name__ == "__main__":
    main()
