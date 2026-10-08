"""VoxCPM2 TTS quick test - runs on Apple Silicon MPS."""
import time
import sys

import soundfile as sf
from voxcpm import VoxCPM

MODEL_PATH = "./pretrained_models/VoxCPM2"
OUTPUT_EN = "test_output_en.wav"
OUTPUT_ZH = "test_output_zh.wav"

print("=" * 60)
print("VoxCPM2 TTS Quick Test")
print("=" * 60)

# Load model
t0 = time.time()
print("\n[1/3] Loading model from local path...")
model = VoxCPM.from_pretrained(MODEL_PATH, load_denoiser=False)
print(f"    Loaded in {time.time() - t0:.1f}s")
print(f"    Sample rate: {model.tts_model.sample_rate} Hz")

# Generate English
text_en = "VoxCPM2 is a tokenizer-free text to speech system for multilingual speech generation."
print(f"\n[2/3] Generating English audio...")
print(f"    Text: {text_en}")
t0 = time.time()
wav = model.generate(
    text=text_en,
    cfg_value=2.0,
    inference_timesteps=10,
    seed=42,
)
gen_time = time.time() - t0
sf.write(OUTPUT_EN, wav, model.tts_model.sample_rate)
print(f"    Generated {len(wav)} samples in {gen_time:.1f}s")
print(f"    Audio duration: {len(wav) / model.tts_model.sample_rate:.1f}s")
print(f"    RTF (real-time factor): {gen_time / (len(wav) / model.tts_model.sample_rate):.2f}")
print(f"    Saved: {OUTPUT_EN}")

# Generate Chinese
text_zh = "VoxCPM2 是一个无分词器的端到端语音合成系统，支持三十种语言。"
print(f"\n[3/3] Generating Chinese audio...")
print(f"    Text: {text_zh}")
t0 = time.time()
wav = model.generate(
    text=text_zh,
    cfg_value=2.0,
    inference_timesteps=10,
    seed=42,
)
gen_time = time.time() - t0
sf.write(OUTPUT_ZH, wav, model.tts_model.sample_rate)
print(f"    Generated {len(wav)} samples in {gen_time:.1f}s")
print(f"    Audio duration: {len(wav) / model.tts_model.sample_rate:.1f}s")
print(f"    RTF (real-time factor): {gen_time / (len(wav) / model.tts_model.sample_rate):.2f}")
print(f"    Saved: {OUTPUT_ZH}")

print("\n" + "=" * 60)
print("Done! All outputs saved.")
print("=" * 60)