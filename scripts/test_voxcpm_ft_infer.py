#!/usr/bin/env python3
"""
Full finetune inference script (no LoRA).

Checkpoint directory contains complete model files (pytorch_model.bin, config.json, audiovae.pth, etc.),
can be loaded directly via VoxCPM.

Usage:

    python scripts/test_voxcpm_ft_infer.py \
        --ckpt_dir /path/to/checkpoints/step_0001000 \
        --text "Hello, I am the finetuned VoxCPM." \
        --output ft_test.wav

With voice cloning:

    python scripts/test_voxcpm_ft_infer.py \
        --ckpt_dir /path/to/checkpoints/step_0001000 \
        --text "Hello, this is voice cloning result." \
        --prompt_audio path/to/ref.wav \
        --prompt_text "Reference audio transcript" \
        --seed 42 \
        --output ft_clone.wav
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import soundfile as sf

from voxcpm.core import VoxCPM


def parse_args():
    parser = argparse.ArgumentParser("VoxCPM full-finetune inference test (no LoRA)")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to the full-finetune checkpoint directory",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--prompt_audio",
        type=str,
        default=None,
        help="Path to reference audio for voice cloning",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default=None,
        help="Transcript of the reference audio",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output wav (auto-generated if omitted)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading VoxCPM full-finetune model from: {args.ckpt_dir}")
    # Initialize the core model directly from checkpoint directory
    model = VoxCPM.from_pretrained(args.ckpt_dir)

    # Determine final output path if not explicitly provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "clone" if args.prompt_audio else "tts"
        args.output = f"{prefix}_{timestamp}.wav"
        print(f"No output path specified. Defaulting to: {args.output}")

    if args.prompt_audio:
        if not args.prompt_text:
            print("Error: --prompt_text is required when --prompt_audio is provided.")
            sys.exit(1)
        
        print(f"Running zero-shot voice cloning using reference: {args.prompt_audio}")
        wav, sr = model.voice_clone(
            text=args.text,
            prompt_audio=args.prompt_audio,
            prompt_text=args.prompt_text
        )
    else:
        print("Running standard text-to-speech inference...")
        wav, sr = model.tts(text=args.text)

    # Ensure output directory exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the generated audio
    sf.write(out_path, wav, sr)
    print(f"Successfully saved generated audio to {out_path}")


if __name__ == "__main__":
    main()
        "--ckpt_dir",
        type=str,
        required=True,
        help="Checkpoint directory (contains pytorch_model.bin, config.json, audiovae.pth, etc.)",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Target text to synthesize",
    )
    parser.add_argument(
        "--prompt_audio",
        type=str,
        default="",
        help="Optional: reference audio path for voice cloning",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default="",
        help="Optional: transcript of reference audio",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ft_test.wav",
        help="Output wav file path",
    )
    parser.add_argument(
        "--cfg_value",
        type=float,
        default=2.0,
        help="CFG scale (default: 2.0)",
    )
    parser.add_argument(
        "--inference_timesteps",
        type=int,
        default=10,
        help="Diffusion inference steps (default: 10)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=600,
        help="Max generation steps",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable text normalization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation (default: None)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model from checkpoint directory (no denoiser)
    print(f"[FT Inference] Loading model: {args.ckpt_dir}", file=sys.stderr)
    model = VoxCPM.from_pretrained(
        hf_model_id=args.ckpt_dir,
        load_denoiser=False,
        optimize=True,
    )

    # Run inference
    prompt_wav_path = args.prompt_audio if args.prompt_audio else None
    prompt_text = args.prompt_text if args.prompt_text else None

    print(f"[FT Inference] Synthesizing: text='{args.text}'", file=sys.stderr)
    if prompt_wav_path:
        print(f"[FT Inference] Using reference audio: {prompt_wav_path}", file=sys.stderr)
        print(f"[FT Inference] Reference text: {prompt_text}", file=sys.stderr)

    if args.seed is not None:
        print(f"[FT Inference] Using seed: {args.seed}", file=sys.stderr)

    audio_np = model.generate(
        text=args.text,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
        normalize=args.normalize,
        denoise=False,
        seed=args.seed,
    )

    # Save audio
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio_np, model.tts_model.sample_rate)

    print(
        f"[FT Inference] Saved to: {out_path}, duration: {len(audio_np) / model.tts_model.sample_rate:.2f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
