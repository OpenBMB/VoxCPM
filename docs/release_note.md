# VoxCPM1.5 Release Notes

**Release Date:** December 5, 2025

## 🎉 Overview


We’re thrilled to introduce a major upgrade that improves audio quality and efficiency of VoxCPM, while maintaining the core capabilities of context-aware speech generation and zero-shot voice cloning.

| Feature | VoxCPM | VoxCPM1.5 |
|---------|------------|------------|
| **Audio VAE Sampling Rate** | 16kHz | 44.1kHz |
| **LM Token Rate** | 12.5Hz | 6.25Hz |
| **Patch Size** | 2 | 4 |
| **SFT Support** | ✅ | ✅ |
| **LoRA Support** | ✅ | ✅ |

## 🎵 Model Updates

### 🔊 AudioVAE Sampling Rate: 16kHz → 44.1kHz

The AudioVAE now supports 44.1kHz sampling rate, which allows the model to:
- 🎯 Clone better, preserving more high-frequency details and generate higher quality voice outputs


*Note: This upgrade enables higher quality generation when using high-quality reference audio, but does not guarantee that all generated audio will be high-fidelity. The output quality depends on the **prompt speech** quality.*

### ⚡ Token Rate: 12.5Hz → 6.25Hz

We reduced the token rate in LM backbone from 12.5Hz to 6.25Hz (LocEnc&LocDiT patch size increased from 2 to 4) while maintaining similar performance on evaluation benchmarks. This change:
- 💨 Reduces computational requirements for generating the same length of audio
- 📈 Provides a foundation for longer audio generation
- 🏗️ Paves the way for training larger models in the future

**Model Architecture Clarification**: The core architecture of VoxCPM1.5 remains unchanged from the technical report. The key modification is adjusting the patch size of the local modules (LocEnc & LocDiT) from 2 to 4, which reduces the LM processing rate from 12.5Hz to 6.25Hz. Since the local modules now need to handle longer contexts, we expanded their network depth, resulting in a slightly larger overall model parameter count.

**Generation Speed Clarification**: Although the model parameters have increased, VoxCPM1.5 only requires 6.25 tokens to generate 1 second of audio (compared to 12.5 tokens in the previous version). While the displayed generation speed (xx it/s) may appear slower, the actual Real-Time Factor (RTF = audio duration / processing time) shows no difference or may even be faster.

## 🔧 Fine-tuning Support

We support full fine-tuning and LoRA fine-tuning now, please see the [Fine-tuning Guide](finetune.md) for detailed instructions.
 

## 📚 Documentation

- Updated README with version comparison
- Added comprehensive fine-tuning guide
- Improved code comments and documentation


## 🙏 Our Thanks to You
This release wouldn’t be possible without the incredible feedback, testing, and contributions from our open-source community. Thank you for helping shape VoxCPM1.5!


## 📞 Let's Build Together
Questions, ideas, or want to contribute?

- 🐛 Report an issue: [GitHub Issues on OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM/issues)

- 📖 Dig into the docs: Check the [docs/](../docs/) folder for guides and API details

Enjoy the richer sound and powerful new features of VoxCPM1.5 🎉

We can't wait to hear what you create next! 🥂

## 🚀 What We're Working On

We're continuously improving VoxCPM and working on exciting new features:

- 🌍 **Multilingual TTS Support**: We are actively developing support for languages beyond Chinese and English.
- 🎯 **Controllable Expressive Speech Generation**: We are researching controllable speech generation that allows fine-grained control over speech attributes (emotion, timbre, prosody, etc.) through natural language instructions.
- 🎵 **Universal Audio Generation Foundation**: We also hope to explore VoxCPM as a unified audio generation foundation model capable of joint generation of speech, music, and sound effects. However, this is a longer-term vision.

**📅 Next Release**: We plan to release the next version in Q1 2026, which will include significant improvements and new features. Stay tuned for updates! We're committed to making VoxCPM even more powerful and versatile.

## ❓ Frequently Asked Questions (FAQ)

### Q: Does VoxCPM support fine-tuning for personalized voice customization?

**A:** Yes! VoxCPM now supports both full fine-tuning (SFT) and efficient LoRA fine-tuning. You can train personalized voice models on your own data. Please refer to the [Fine-tuning Guide](finetune.md) for detailed instructions and examples.

### Q: Is 16kHz audio quality sufficient for my use case?

**A:** We have upgraded the AudioVAE to support 44.1kHz sampling rate in VoxCPM1.5, which provides higher quality audio output with better preservation of high-frequency details. This upgrade enables better voice cloning quality and more natural speech synthesis when using high-quality reference audio.

### Q: Has the stability issue been resolved?

**A:** We have made stability optimizations in VoxCPM1.5, including improvements to the inference code logic, training data, and model architecture. Based on community feedback, we collected some stability issues such as:
- Increased noise and reverberation
- Audio artifacts (e.g., howling/squealing)
- Unstable speaking rate (speeding up)
- Volume fluctuations (increases or decreases)
- Noise artifacts at the beginning and end of audio
- Synthesis issues with very short texts (e.g., "hello")

**What we've improved:**
- By adjusting inference code logic and optimizing training data, we have largely fixed the beginning/ending artifacts.
- By reducing the LM processing rate (12.5Hz → 6.25Hz), we have improved stability on longer speech generation cases.

**What remains:** We acknowledge that long speech stability issues have not been completely resolved. Particularly for highly expressive or complex reference speech, error accumulation during autoregressive generation can still occur. We will continue to analyze and optimize this in future versions.

### Q: Does VoxCPM plan to support multilingual TTS?

**A:** Currently, VoxCPM is primarily trained on Chinese and English data. We are actively researching and developing multilingual TTS support for more languages beyond Chinese and English. Please let us know what languages you'd like to see supported!

### Q: Does VoxCPM plan to support controllable generation (emotion, style, fine-grained control)?

**A:** Currently, VoxCPM only supports zero-shot voice cloning and context-aware speech generation. Direct control over specific speech attributes (emotion, style, fine-grained prosody) is limited. However, we are actively researching instruction-controllable expressive speech generation with fine-grained control capabilities, working towards a human instruction-to-speech generation model!

### Q: Does VoxCPM support different hardware chips (e.g., Ascend 910B, XPU, NPU)?

**A:** Currently, we have not yet adapted VoxCPM for different hardware chips. Our main focus remains on developing new model capabilities and improving stability. We encourage you to check if community developers have done similar work, and we warmly welcome everyone to contribute and promote such adaptations together!

These features are under active development, and we look forward to sharing updates in future releases!


