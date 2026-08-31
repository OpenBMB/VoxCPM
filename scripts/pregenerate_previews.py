"""Pre-generate the voice-preview .wav files for every preset voice.

Run this ONCE to warm the preview cache so the "Écouter un aperçu" button in the
web UI returns instantly instead of synthesizing on demand. Safe to re-run — it
skips voices whose preview already exists.

Usage (from the repo root, using the project venv):
    ./.venv/Scripts/python.exe scripts/pregenerate_previews.py
    ./.venv/Scripts/python.exe scripts/pregenerate_previews.py --device cpu --force

The denoiser is never loaded here (previews use no reference audio), so startup is
fast and does not touch ModelScope.
"""
import argparse
import sys
from pathlib import Path

import soundfile as sf

# Make the repo root importable so we can reuse app.py's presets and constants.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402


def remaster_existing() -> int:
    """Bring already-generated previews to the same level, without the model.

    Raw generations span some 15 dB depending on the voice, which makes the
    quietest presets hard to hear and makes auditioning unfair — level is heard
    as quality. Levelling is a post-process, so this costs seconds rather than
    the minutes per voice that regenerating would.
    """
    from narration import audio as audio_tools  # noqa: PLC0415 - only needed here

    changed = 0
    for voice in app.PRESET_VOICES:
        path = app._PREVIEW_DIR / f"preview_{voice['seed']}.wav"
        if not path.is_file():
            print(f"seed={voice['seed']}: no preview yet, skipping", flush=True)
            continue
        wav, sample_rate = sf.read(str(path), dtype="float32")
        before = audio_tools.speech_rms_db(wav, sample_rate)
        mastered = app.master_preview(sample_rate, wav)
        after = audio_tools.speech_rms_db(mastered, sample_rate)
        sf.write(str(path), mastered, sample_rate)
        changed += 1
        print(f"seed={voice['seed']}: {before:6.1f} -> {after:6.1f} dBFS  {voice['name']}", flush=True)
    print(f"Re-levelled {changed} preview(s).", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="auto, cpu, mps, cuda, or cuda:N (default: cpu)")
    parser.add_argument("--model-id", default="openbmb/VoxCPM2", help="Model path or HF repo id")
    parser.add_argument("--force", action="store_true", help="Regenerate even if the preview already exists")
    parser.add_argument("--remaster", action="store_true",
                        help="Re-level existing previews in place and exit — no model, no generation")
    args = parser.parse_args()

    app._PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    if args.remaster:
        return remaster_existing()

    demo = app.VoxCPMDemo(model_id=args.model_id, device=args.device, load_denoiser=False)

    total = len(app.PRESET_VOICES)
    for i, voice in enumerate(app.PRESET_VOICES, start=1):
        seed = voice["seed"]
        out = app._PREVIEW_DIR / f"preview_{seed}.wav"
        tag = f"[{i}/{total}] seed={seed}"
        if out.is_file() and not args.force:
            print(f"{tag}: already exists, skipping -> {out.name}", flush=True)
            continue
        print(f"{tag}: generating (this is slow on CPU) ...", flush=True)
        sr, wav, _ = demo.generate_tts_audio(
            text_input=app._PREVIEW_TEXT,
            control_instruction=voice["description"],
            cfg_value_input=voice.get("cfg", 2.0),
            do_normalize=voice.get("normalize", True),
            inference_timesteps=int(voice.get("diffusion_steps", 10)),
            seed=seed,
        )
        wav = app.master_preview(sr, wav)
        sf.write(str(out), wav, sr)
        print(f"{tag}: saved -> {out.name} ({len(wav) / sr:.2f}s)", flush=True)

    print("Done. Preview cache is warm.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
