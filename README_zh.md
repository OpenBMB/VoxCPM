<h2 align="center">VoxCPM2：基于连续表征的多语言语音合成、创意音色设计与高保真声音克隆</h2>

<p align="center">
  <a href="./README.md">English</a> | <b>中文</b>
</p>

<p align="center">
  <a href="https://github.com/OpenBMB/VoxCPM/"><img src="https://img.shields.io/badge/Project%20Page-GitHub-blue" alt="Project Page"></a>
  <a href="https://huggingface.co/spaces/OpenBMB/VoxCPM-Demo"><img src="https://img.shields.io/badge/Live%20Playground-Demo-orange" alt="Live Playground"></a>
  <a href="https://voxcpm.readthedocs.io/zh-cn/latest/"><img src="https://img.shields.io/badge/Docs-ReadTheDocs-8CA1AF" alt="Documentation"></a>
  <a href="https://huggingface.co/openbmb/VoxCPM2"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VoxCPM2-yellow" alt="Hugging Face"></a>
  <a href="https://modelscope.cn/models/OpenBMB/VoxCPM2"><img src="https://img.shields.io/badge/ModelScope-VoxCPM2-purple" alt="ModelScope"></a>
  <a href="https://openbmb.github.io/voxcpm2-demopage/"><img src="https://img.shields.io/badge/DemoPage-Audio Samples-red"></a>
  
</p>

<div align="center">
  <img src="assets/voxcpm_logo.png" alt="VoxCPM Logo" width="35%">
  <br><br>
  <a href="https://trendshift.io/repositories/17704" target="_blank"><img src="https://trendshift.io/api/badge/repositories/17704" alt="OpenBMB%2FVoxCPM | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<br>

<p align="center">
  👋 欢迎加入社区，参与讨论与交流！
  <br>
  <a href="./assets/feishu-group.png" style="display:inline-block;vertical-align:middle; margin-left: 10px;">
    <img src="./assets/feishu-logo.png" width="16" height="16" style="vertical-align:middle;"> 飞书群
  </a>
  &nbsp;|&nbsp;
  <a href="https://discord.gg/KZUx7tVNwz" style="display:inline-block;vertical-align:middle;">
    <img src="./assets/discord-logo.png" width="16" height="16" style="vertical-align:middle;"> Discord
  </a>
</p>

VoxCPM 是一个**无离散音频分词器**（Tokenizer-Free）的语音合成系统，通过端到端的**扩散自回归架构**直接生成连续语音表征，绕过对音频的离散编码步骤，实现高度自然且富有表现力的语音合成。

**VoxCPM2** 是最新的版本 — 基于 [MiniCPM-4](https://github.com/OpenBMB/MiniCPM) 基座构建，总计 **20亿** 参数，在超过 **200万小时** 的多语种音频数据上训练，支持 **30种全球语言+9种中文方言**、**音色设计**、**可控声音克隆**，原生输出 **48kHz** 高质量音频。

### ✨ 核心特性

- 🌍 **30种语言语音合成** — 直接输入原始文本即可合成（支持语言详见下文），无需额外语言标签
- 🎨 **音色设计** — 用自然语言描述（性别、年龄、音色、情绪、语速……）凭空创建全新音色，无需参考音频
- 🎛️ **可控声音克隆** — 从参考音频片段克隆任意声音，可叠加风格指令控制情绪、语速和表现力，同时保持原始音色
- 🎙️ **极致克隆** — 提供参考音频及其文本内容，模型接着参考音频进行无缝续写，从而精准还原声音细节特征（与 VoxCPM1.5 一致）
- 🔊 **48kHz 高质量音频** — 输入 16kHz 参考音频，通过 AudioVAE V2 的非对称编解码设计直接输出 48kHz 高质量音频，内置超分能力
- 🧠 **语境感知合成** — 根据文本内容自动推断合适的韵律和表现力
- ⚡ **实时流式合成** — 在 NVIDIA RTX 4090 上 RTF 低至 ~0.3，通过 [Nano-vLLM](https://github.com/a710128/nanovllm-voxcpm) 或 [vLLM-Omni](https://github.com/vllm-project/vllm-omni)（官方 vLLM 全模态服务，原生支持 VoxCPM2，提供 PagedAttention 与 OpenAI 兼容 API）加速后可达 ~0.13
- 📜 **完全开源，商用就绪** — 权重和代码基于 [Apache-2.0](LICENSE) 协议发布，免费商用

<summary><b>🌍 支持的语言（30种）</b></summary>
<br>
阿拉伯语、缅甸语、中文、丹麦语、荷兰语、英语、芬兰语、法语、德语、希腊语、希伯来语、印地语、印尼语、意大利语、日语、高棉语、韩语、老挝语、马来语、挪威语、波兰语、葡萄牙语、俄语、西班牙语、斯瓦希里语、瑞典语、菲律宾语、泰语、土耳其语、越南语

中国方言：四川话、粤语、吴语、东北话、河南话、陕西话、山东话、天津话、闽南话


### 最新动态

* **[2026.04]** 🔥 发布 **VoxCPM2** — 20亿参数，30种语言，音色设计与可控声音克隆，48kHz 音频输出！[模型权重](https://huggingface.co/openbmb/VoxCPM2) | [使用文档](https://voxcpm.readthedocs.io/zh-cn/latest/) | [在线体验](https://huggingface.co/spaces/OpenBMB/VoxCPM-Demo) | [官网体验](https://voxcpm.modelbest.cn/) (适用国内访问) | [技术报告](https://arxiv.org/abs/2606.06928)
* **[2025.12]** 🎉 开源 **VoxCPM1.5** [模型权重](https://huggingface.co/openbmb/VoxCPM1.5)，支持 SFT 和 LoRA 微调。(**🏆 GitHub Trending #1**)
* **[2025.09]** 🔥 发布 VoxCPM [技术报告](https://arxiv.org/abs/2509.24650)。
* **[2025.09]** 🎉 开源 **VoxCPM-0.5B** [模型权重](https://huggingface.co/openbmb/VoxCPM-0.5B) (**🏆 HuggingFace Trending #1**)

---

## 目录

- [快速开始](#-快速开始)
  - [安装](#安装)
  - [Python API](#python-api)
  - [命令行使用](#命令行使用)
  - [Web Demo](#web-demo)
  - [生产部署](#-生产部署nano-vllm)
- [模型与版本](#-模型与版本)
- [性能评测](#-性能评测)
- [微调](#%EF%B8%8F-微调)
- [文档](#-文档)
- [生态与社区](#-生态与社区)
- [风险与局限性](#%EF%B8%8F-风险与局限性)
- [引用](#-引用)

---

## 🚀 快速开始

### 安装

```sh
pip install voxcpm
```

> **环境要求：** Python ≥ 3.10 (<3.13)，PyTorch ≥ 2.5.0，CUDA ≥ 12.0。详见 [快速开始文档](https://voxcpm.readthedocs.io/zh-cn/latest/quickstart.html)。

### Python API

#### 🗣️ 文本转语音

```python
from voxcpm import VoxCPM
import soundfile as sf

model = VoxCPM.from_pretrained(
  "openbmb/VoxCPM2",
  load_denoiser=False,
)

wav = model.generate(
    text="VoxCPM2 是目前推荐使用的多语言语音合成版本。",
    cfg_value=2.0,
    inference_timesteps=10,
)
sf.write("demo.wav", wav, model.tts_model.sample_rate)
print("已保存: demo.wav")
```

如果你希望先从 ModelScope 下载模型到本地（适用于国内网络访问），可以使用：

```bash
pip install modelscope
```

```python
from modelscope import snapshot_download
snapshot_download("OpenBMB/VoxCPM2", local_dir='./pretrained_models/VoxCPM2') # 指定模型保存的本地路径

from voxcpm import VoxCPM
import soundfile as sf
model = VoxCPM.from_pretrained('./pretrained_models/VoxCPM2', load_denoiser=False)

wav = model.generate(
    text="VoxCPM2 是目前推荐使用的多语言语音合成版本。",
    cfg_value=2.0,
    inference_timesteps=10,
)
sf.write("demo.wav", wav, model.tts_model.sample_rate)
```

#### 🎨 音色设计

用自然语言描述创建全新音色，无需参考音频。**格式：** 在 `text` 开头用括号写入音色描述（如 `"(音色描述)要合成的文本。"`）：

```python
wav = model.generate(
    text="(年轻女性，声音温柔甜美)你好，欢迎使用VoxCPM2！",
    cfg_value=2.0,
    inference_timesteps=10,
)
sf.write("voice_design.wav", wav, model.tts_model.sample_rate)
```

#### 🎛️ 可控声音克隆

上传一段参考音频，模型克隆其音色，同时可以使用控制指令调节语速、情绪或风格。

```python
wav = model.generate(
    text="这是VoxCPM2生成的克隆语音。",
    reference_wav_path="path/to/voice.wav",
)
sf.write("clone.wav", wav, model.tts_model.sample_rate)

wav = model.generate(
    text="(稍快一点，欢快的语气)这是带风格控制的克隆语音。",
    reference_wav_path="path/to/voice.wav",
    cfg_value=2.0,
    inference_timesteps=10,
)
sf.write("controllable_clone.wav", wav, model.tts_model.sample_rate)
```

#### 🎙️ 极致克隆

提供参考音频及其精确文本转录，实现基于音频续写的高保真克隆。为获得最高克隆相似度，可将同一音频同时传给 `reference_wav_path` 和 `prompt_wav_path`：

```python
wav = model.generate(
    text="这是使用VoxCPM2的极致克隆演示。",
    prompt_wav_path="path/to/voice.wav",
    prompt_text="参考音频的文本转录。",
    reference_wav_path="path/to/voice.wav",  # 可选，提升相似度
)
sf.write("hifi_clone.wav", wav, model.tts_model.sample_rate)
```

<details>
<summary><b>🔄 流式 API</b></summary>

```python
import numpy as np

chunks = []
for chunk in model.generate_streaming(
    text="使用VoxCPM进行流式语音合成非常简单！",
):
    chunks.append(chunk)
wav = np.concatenate(chunks)
sf.write("streaming.wav", wav, model.tts_model.sample_rate)
```
</details>

#### 📖 长文本生成

对于较长文稿，`generate_long_form()` 会将文本切分为较短片段，并使用首段生成结果作为后续片段的稳定 prompt/reference 锚点，从而降低长句一次性自回归生成带来的音色漂移风险。

```python
wav = model.generate_long_form(
    text="第一段内容。第二段内容。第三段内容。",
    control="温暖稳定的旁白男声",
    max_chars=120,
    silence_ms=300,
)
sf.write("long_form.wav", wav, model.tts_model.sample_rate)
```

### 命令行使用

```bash
# 音色设计（无需参考音频）
voxcpm design \
  --text "VoxCPM2带来全新语音合成体验。" \
  --output out.wav

# 可控声音克隆（带风格控制）
voxcpm design \
  --text "VoxCPM2带来全新语音合成体验。" \
  --control "年轻女声，温暖温柔，略带微笑" \
  --output out.wav

# 长文本音色设计
voxcpm design \
  --text "第一段内容。第二段内容。第三段内容。" \
  --control "温暖稳定的旁白男声" \
  --long-form \
  --long-form-max-chars 120 \
  --output long_form.wav

# 声音克隆（参考音频）
voxcpm clone \
  --text "这是一个声音克隆的演示。" \
  --reference-audio path/to/voice.wav \
  --output out.wav

# 极致克隆（提示音频 + 转录文本）
voxcpm clone \
  --text "这是一个声音克隆的演示。" \
  --prompt-audio path/to/voice.wav \
  --prompt-text "参考音频转录文本" \
  --reference-audio path/to/voice.wav \
  --output out.wav

# 批量处理
voxcpm batch --input examples/input.txt --output-dir outs

# 帮助
voxcpm --help
```

### Web Demo

```bash
python app.py --port 8808  # 然后在浏览器打开 http://localhost:8808
```

使用 `--device` 选择运行设备：

```bash
python app.py --device auto
```

支持的取值包括 `auto`、`cpu`、`mps`、`cuda` 和 `cuda:N`。在 Apple Silicon Mac 上，`auto` 会在可用时使用 MPS。

### 🚢 生产部署（Nano-vLLM）

如需高吞吐量部署，使用 [**Nano-vLLM-VoxCPM**](https://github.com/a710128/nanovllm-voxcpm) — 基于 Nano-vLLM 构建的专用推理引擎，支持并发请求和异步 API。

```bash
pip install nano-vllm-voxcpm
```

```python
from nanovllm_voxcpm import VoxCPM
import numpy as np, soundfile as sf

server = VoxCPM.from_pretrained(model="/path/to/VoxCPM", devices=[0])
chunks = list(server.generate(target_text="你好，我来自VoxCPM！"))
sf.write("out.wav", np.concatenate(chunks), 48000)
server.stop()
```

> **在 NVIDIA RTX 4090 上 RTF 低至 ~0.13**（标准 PyTorch 实现约 ~0.3），支持批量并发请求和 FastAPI HTTP 服务。详见 [Nano-vLLM-VoxCPM 仓库](https://github.com/a710128/nanovllm-voxcpm)。

### 🏭 生产环境部署（vLLM-Omni）

如需生产级多租户部署，使用 [**vLLM-Omni**](https://github.com/vllm-project/vllm-omni) — 官方 vLLM 项目的全模态扩展，原生支持 **VoxCPM2**。具备 PagedAttention KV 缓存、连续批处理，以及与 OpenAI 完全兼容的 `/v1/audio/speech` 接口。

```bash
# 从源码安装（最新 main 分支 —— vllm-omni 正在快速迭代）
uv pip install vllm==0.19.0 --torch-backend=auto
git clone https://github.com/vllm-project/vllm-omni.git && cd vllm-omni
uv pip install -e .
```

其他平台（ROCm、XPU、MUSA、NPU）与 Docker 镜像请参考 [vLLM-Omni 安装文档](https://vllm-omni.readthedocs.io/en/latest/getting_started/installation/)。

```bash
# 启动 OpenAI 兼容的 TTS 服务（--omni 启用全模态服务）
vllm serve openbmb/VoxCPM2 --omni --port 8000

# 任意 OpenAI 客户端均可调用
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"openbmb/VoxCPM2","input":"你好，欢迎使用 VoxCPM2 on vLLM-Omni！","voice":"default"}' \
  --output out.wav
```

> 基于上游 vLLM 调度器构建，开箱即用支持批量并发、流式分块输出和多 GPU 部署。完整示例见 [VoxCPM2 部署样例](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/voxcpm2)。

> **完整参数说明、多场景示例与声音克隆技巧 →** [快速开始指南](https://voxcpm.readthedocs.io/zh-cn/latest/quickstart.html) | [使用指南](https://voxcpm.readthedocs.io/zh-cn/latest/usage_guide.html) | [Cookbook](https://voxcpm.readthedocs.io/zh-cn/latest/cookbook.html)

---

## 📦 模型与版本

| | **VoxCPM2** | **VoxCPM1.5** | **VoxCPM-0.5B** |
|---|:---:|:---:|:---:|
| **状态** | 🟢 最新版本 | 稳定版 | 旧版 |
| **主模型参数量** | 2B | 0.6B | 0.5B |
| **音频采样率** | 48kHz | 44.1kHz | 16kHz |
| **LM处理码率** | 6.25Hz | 6.25Hz | 12.5Hz |
| **语言支持数量** | 30 | 2（中文、英文） | 2（中文、英文） |
| **克隆模式** | 隔离参考音频（无需文本） & 音频续写 | 仅音频续写 | 仅音频续写 |
| **音色设计** | ✅ | — | — |
| **可控声音克隆** | ✅ | — | — |
| **SFT / LoRA** | ✅ | ✅ | ✅ |
| **RTF (RTX 4090)** | ~0.30 | ~0.15 | ~0.17 |
| **RTF Nano-VLLM (RTX 4090)** | ~0.13 | ~0.08 | ~0.10 |
| **显存占用** | ~8 GB | ~6 GB | ~5 GB |
| **模型权重** | [🤗 HF](https://huggingface.co/openbmb/VoxCPM2) / [MS](https://modelscope.cn/models/OpenBMB/VoxCPM2) | [🤗 HF](https://huggingface.co/openbmb/VoxCPM1.5) / [MS](https://modelscope.cn/models/OpenBMB/VoxCPM1.5) | [🤗 HF](https://huggingface.co/openbmb/VoxCPM-0.5B) / [MS](https://modelscope.cn/models/OpenBMB/VoxCPM-0.5B) |
| **技术报告** | [arXiv](https://arxiv.org/abs/2606.06928) | — | [arXiv](https://arxiv.org/abs/2509.24650) [ICLR 2026](https://openreview.net/forum?id=h5KLpGoqzC) |
| **Demo 页面** | [音频示例](https://openbmb.github.io/voxcpm2-demopage) | — | [音频示例](https://openbmb.github.io/VoxCPM-demopage) |

VoxCPM2 采用**连续音频表征、扩散自回归**范式，模型在 **AudioVAE** 的连续隐空间中通过四阶段处理：**LocEnc → TSLM → RALM → LocDiT**，实现丰富的表现力语音合成和 48kHz 原生音频输出。

<div align="center">
  <img src="assets/voxcpm2_model.png" alt="VoxCPM2 模型架构" width="90%">
</div>

> 完整架构细节、VoxCPM2 升级内容和模型对比表见 [架构设计文档](https://voxcpm.readthedocs.io/zh-cn/latest/models/architecture.html)。

---

## 📊 性能评测

VoxCPM2 在公开的零样本和可控 TTS 基准测试中取得了 SOTA 或可比的结果。

### Seed-TTS-eval

<details>
<summary><b>Seed-TTS-eval WER(⬇)&SIM(⬆) 结果（点击展开）</b></summary>

| Model | Parameters | Open-Source | test-EN | | test-ZH | | test-Hard | |
|------|------|------|:------------:|:--:|:------------:|:--:|:-------------:|:--:|
| | | | WER/%⬇ | SIM/%⬆| CER/%⬇| SIM/%⬆ | CER/%⬇ | SIM/%⬆ |
| MegaTTS3 | 0.5B | ❌ | 2.79 | 77.1 | 1.52 | 79.0 | - | - |
| DiTAR | 0.6B | ❌ | 1.69 | 73.5 | 1.02 | 75.3 | - | - |
| CosyVoice3 | 0.5B | ❌ | 2.02 | 71.8 | 1.16 | 78.0 | 6.08 | 75.8 |
| CosyVoice3 | 1.5B | ❌ | 2.22 | 72.0 | 1.12 | 78.1 | 5.83 | 75.8 |
| Seed-TTS | - | ❌ | 2.25 | 76.2 | 1.12 | 79.6 | 7.59 | 77.6 |
| MiniMax-Speech | - | ❌ | 1.65 | 69.2 | 0.83 | 78.3 | - | - |
| F5-TTS | 0.3B | ✅ | 2.00 | 67.0 | 1.53 | 76.0 | 8.67 | 71.3 |
| MaskGCT | 1B | ✅ | 2.62 | 71.7 | 2.27 | 77.4 | - | - |
| CosyVoice | 0.3B | ✅ | 4.29 | 60.9 | 3.63 | 72.3 | 11.75 | 70.9 |
| CosyVoice2 | 0.5B | ✅ | 3.09 | 65.9 | 1.38 | 75.7 | 6.83 | 72.4 |
| SparkTTS | 0.5B | ✅ | 3.14 | 57.3 | 1.54 | 66.0 | - | - |
| FireRedTTS | 0.5B | ✅ | 3.82 | 46.0 | 1.51 | 63.5 | 17.45 | 62.1 |
| FireRedTTS-2 | 1.5B | ✅ | 1.95 | 66.5 | 1.14 | 73.6 | - | - |
| Qwen2.5-Omni | 7B | ✅ | 2.72 | 63.2 | 1.70 | 75.2 | 7.97 | 74.7 |
| Qwen3-Omni | 30B-A3B | ✅ | 1.39 | - | 1.07 | - | - | - |
| OpenAudio-s1-mini | 0.5B | ✅ | 1.94 | 55.0 | 1.18 | 68.5 | 23.37 | 64.3 |
| IndexTTS2 | 1.5B | ✅ | 2.23 | 70.6 | 1.03 | 76.5 | 7.12 | 75.5 |
| VibeVoice | 1.5B | ✅ | 3.04 | 68.9 | 1.16 | 74.4 | - | - |
| HiggsAudio-v2 | 3B | ✅ | 2.44 | 67.7 | 1.50 | 74.0 | 55.07 | 65.6 |
| VoxCPM-0.5B | 0.6B | ✅ | 1.85 | 72.9 | 0.93 | 77.2 | 8.87 | 73.0 |
| VoxCPM1.5 | 0.8B | ✅ | 2.12 | 71.4 | 1.18 | 77.0 | 7.74 | 73.1 |
| MOSS-TTS |  | ✅ | 1.85 | 73.4 | 1.20 | 78.8 | - | - |
| Qwen3-TTS | 1.7B | ✅ | 1.23 | 71.7 | 1.22 | 77.0 | 6.76 | 74.8 |
| FishAudio S2 | 4B | ✅ | 0.99 | - | 0.54 | - | 5.99 | - |
| LongCat-Audio-DiT | 3.5B | ✅ | 1.50 | 78.6 | 1.09 | 81.8 | 6.04 | 79.7 |
| **VoxCPM2** | 2B | ✅ | 1.84 | 75.3 | 0.97| 79.5| 8.13 | 75.3 |  
</details>


### CV3-eval 
<details>
<summary><b>CV3-eval 多语言 WER/CER(⬇) 结果（点击展开）</b></summary>

| Model | zh | en | hard-zh | hard-en | ja | ko | de | es | fr | it | ru |
|-------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| CosyVoice2 | 4.08 | 6.32 | 12.58| 11.96| 9.13 | 19.7 |- | - | - | - | - |
| CosyVoice3-1.5B | 3.91 | 4.99 | 9.77 | 10.55 | 7.57 | 5.69 | 6.43 | 4.47 | 11.8 | 10.5 | 6.64 |
| Fish Audio S2 | 2.65 | 2.43 | 9.10 | 4.40 | 3.96 | 2.76 | 2.22 | 2.00 | 6.26 | 2.04 | 2.78 |
| **VoxCPM2** | 3.65 | 5.00 | 8.55 | 8.48 | 5.96 | 5.69 | 4.77 | 3.80 | 9.85 | 4.25 | 5.21 |
</details>

### MiniMax-Multilingual-Test

<details>
<summary><b>Minimax-MLS-test WER(⬇) 结果（点击展开）</b></summary>

| Language | Minimax | ElevenLabs | Qwen3-TTS | FishAudio S2 | **VoxCPM2** |
|----------|:-------:|:----------:|:--------------------:|:------------:|:-----------:|
| Arabic | **1.665** | 1.666 | – | 3.500 | 13.046 |
| Cantonese | 34.111 | 51.513 | – | **30.670** | 38.584 |
| Chinese | 2.252 | 16.026 | 0.928 | **0.730** | 1.136 |
| Czech | 3.875 | **2.108** | – | 2.840 | 24.132 |
| Dutch | 1.143 | **0.803** | – | 0.990 | 0.913 |
| English | 2.164 | 2.339 | **0.934** | 1.620 | 2.289 |
| Finnish | 4.666 | 2.964 | – | 3.330 | **2.632** |
| French | 4.099 | 5.216 | **2.858** | 3.050 | 4.534 |
| German | 1.906 | 0.572 | 1.235 | **0.550** | 0.679 |
| Greek | 2.016 | **0.991** | – | 5.740 | 2.844 |
| Hindi | 6.962 | **5.827** | – | 14.640 | 19.699 |
| Indonesian | 1.237 | **1.059** | – | 1.460 | 1.084 |
| Italian | 1.543 | 1.743 | **0.948** | 1.270 | 1.563 |
| Japanese | 3.519 | 10.646 | 3.823 | **2.760** | 4.628 |
| Korean | 1.747 | 1.865 | 1.755 | **1.180** | 1.962 |
| Polish | 1.415 | **0.766** | – | 1.260 | 1.141 |
| Portuguese | 1.877 | 1.331 | 1.526 | **1.140** | 1.938 |
| Romanian | 2.878 | **1.347** | – | 10.740 | 21.577 |
| Russian | 4.281 | 3.878 | 3.212 | **2.400** | 3.634 |
| Spanish | 1.029 | 1.084 | 1.126 | **0.910** | 1.438 |
| Thai | 2.701 | 73.936 | – | 4.230 | 2.961 |
| Turkish | 1.52 | 0.699 | – | 0.870 | 0.817 |
| Ukrainian | 1.082 | **0.997** | – | 2.300 | 6.316 |
| Vietnamese | **0.88** | 73.415 | – | 7.410 | 3.307 |

</details>

<details>
<summary><b>Minimax-MLS-test SIM(⬆) 结果（点击展开）</b></summary>

| Language | Minimax | ElevenLabs | Qwen3-TTS | FishAudio S2 | **VoxCPM2** |
|----------|:-------:|:----------:|:--------------------:|:------------:|:-----------:|
| Arabic | 73.6 | 70.6 | – | 75.0 | **79.1** |
| Cantonese | 77.8 | 67.0 | – | 80.5 | **83.5** |
| Chinese | 78.0 | 67.7 | 79.9 | 81.6 | **82.5** |
| Czech | 79.6 | 68.5 | – | **79.8** | 78.3 |
| Dutch | 73.8 | 68.0 | – | 73.0 | **80.8** |
| English | 75.6 | 61.3 | 77.5 | 79.7 | **85.4** |
| Finnish | 83.5 | 75.9 | – | 81.9 | **89.0** |
| French | 62.8 | 53.5 | 62.8 | 69.8 | **73.5** |
| German | 73.3 | 61.4 | 77.5 | 76.7 | **80.3** |
| Greek | 82.6 | 73.3 | – | 79.5 | **86.0** |
| Hindi | 81.8 | 73.0 | – | 82.1 | **85.6** |
| Indonesian | 72.9 | 66.0 | – | 76.3 | **80.0** |
| Italian | 69.9 | 57.9 | 81.7 | 74.7 | **78.0** |
| Japanese | 77.6 | 73.8 | 78.8 | 79.6 | **82.8** |
| Korean | 77.6 | 70.0 | 79.9 | 81.7 | **83.3** |
| Polish | 80.2 | 72.9 | – | 81.9 | **88.4** |
| Portuguese | 80.5 | 71.1 | 81.7 | 78.1 | **83.7** |
| Romanian | **80.9** | 69.9 | – | 73.3 | 79.7 |
| Russian | 76.1 | 67.6 | 79.2 | 79.0 | **81.1** |
| Spanish | 76.2 | 61.5 | 81.4 | 77.6 | **83.1** |
| Thai | 80.0 | 58.8 | – | 78.6 | **84.0** |
| Turkish | 77.9 | 59.6 | – | 83.5 | **87.1** |
| Ukrainian | 73.0 | 64.7 | – | 74.7 | **79.8** |
| Vietnamese | 74.3 | 36.9 | – | 74.0 | **80.6** |

</details>

### Internal 30-Language ASR Benchmark

我们额外进行了内部多语言可懂度评测：**30 语种 × 500 样本**，ASR 转写评估使用 **Gemini 3.1 Flash Lite API**。

<details>
<summary><b>内部30语种评测集ASR结果（点击展开）</b></summary>

| 语言 | 指标 | VoxCPM2 | Fish S2-Pro |
|---|---:|---:|---:|
| ar (阿拉伯语) | CER | 1.23% | 0.30% |
| da (丹麦语) | WER | 2.70% | 3.52% |
| de (德语) | WER | 0.96% | 0.64% |
| el (希腊语) | WER | 3.17% | 4.61% |
| en (英语) | WER | 0.42% | 1.03% |
| es (西班牙语) | WER | 1.33% | 0.64% |
| fi (芬兰语) | WER | 2.24% | 2.80% |
| fr (法语) | WER | 2.16% | 2.34% |
| he (希伯来语) | CER | 2.98% | 15.27% |
| hi (印地语) | CER | 0.79% | 0.91% |
| id (印尼语) | WER | 1.36% | 1.68% |
| it (意大利语) | WER | 1.65% | 1.08% |
| ja (日语) | CER | 2.40% | 1.82% |
| km (高棉语) | CER | 2.05% | 75.15% |
| ko (韩语) | CER | 0.95% | 0.29% |
| lo (老挝语) | CER | 1.90% | 87.40% |
| ms (马来语) | WER | 1.75% | 1.41% |
| my (缅甸语) | CER | 1.42% | 85.27% |
| nl (荷兰语) | WER | 1.25% | 1.68% |
| no (挪威语) | WER | 2.49% | 3.76% |
| pl (波兰语) | WER | 1.90% | 1.65% |
| pt (葡萄牙语) | WER | 1.48% | 1.49% |
| ru (俄语) | WER | 0.90% | 0.86% |
| sv (瑞典语) | WER | 2.22% | 2.63% |
| sw (斯瓦希里语) | CER | 1.07% | 2.02% |
| th (泰语) | CER | 0.94% | 1.92% |
| tl (菲律宾语) | WER | 2.63% | 4.00% |
| tr (土耳其语) | WER | 1.65% | 1.65% |
| vi (越南语) | WER | 1.56% | 5.56% |
| zh (中文) | CER | 0.92% | 1.02% |
| 平均（30 语种） |  | **1.68%** | - |

</details>


### InstructTTSEval

<details>
<summary><b>指令驱动音色设计结果 (点击展开)</b></summary>

| Model | InstructTTSEval-ZH | | | InstructTTSEval-EN | | |
|-------|:---:|:----:|:----:|:----:|:----:|:----:|
| | APS⬆| DSD⬆ | RP⬆| APS⬆ | DSD⬆ | RP⬆ |
| Hume | – | – | – | 83.0 | 75.3 | 54.3 |
| VoxInstruct | 47.5 | 52.3 | 42.6 | 54.9 | 57.0 | 39.3 |
| Parler-tts-mini | – | – | – | 63.4 | 48.7 | 28.6 |
| Parler-tts-large | – | – | – | 60.0 | 45.9 | 31.2 |
| PromptTTS | – | – | – | 64.3 | 47.2 | 31.4 |
| PromptStyle | – | – | – | 57.4 | 46.4 | 30.9 |
| VoiceSculptor | 75.7 | 64.7 | 61.5 | – | – | – |
| Mimo-Audio-7B-Instruct | 75.7 | 74.3 | 61.5 | 80.6 | 77.6 | 59.5 |
| Qwen3TTS-12Hz-1.7B-VD | **85.2** | **81.1** | **65.1** | 82.9 | 82.4 | 68.4 |
| **VoxCPM2** | **85.2** | 71.5 | 60.8 | **84.2** | **83.2** | **71.4** |

</details>

---

## ⚙️ 微调

VoxCPM 支持**全参数微调（SFT）** 和 **LoRA 微调**。仅需 **5-10分钟** 的音频数据，即可适配特定说话人、语言或领域。

```bash
# LoRA 微调（参数高效，推荐）
python scripts/train_voxcpm_finetune.py \
    --config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml

# 全参数微调
python scripts/train_voxcpm_finetune.py \
    --config_path conf/voxcpm_v2/voxcpm_finetune_all.yaml

# WebUI 训练与推理
python lora_ft_webui.py   # 然后打开 http://localhost:7860
```

> **完整指南 →** [微调文档](https://voxcpm.readthedocs.io/zh-cn/latest/finetuning/finetune.html)（数据准备、配置、训练、LoRA 热切换、常见问题）

---

## 📚 文档

完整文档：**[voxcpm.readthedocs.io](https://voxcpm.readthedocs.io/zh-cn/latest/)**

| 主题 | 链接 |
|---|---|
| 快速开始与安装 | [快速开始](https://voxcpm.readthedocs.io/zh-cn/latest/quickstart.html) |
| 使用指南与 Cookbook | [使用指南](https://voxcpm.readthedocs.io/zh-cn/latest/usage_guide.html) |
| VoxCPM 系列模型 | [模型列表](https://voxcpm.readthedocs.io/zh-cn/latest/models/version_history.html) |
| 微调（SFT & LoRA） | [微调指南](https://voxcpm.readthedocs.io/zh-cn/latest/finetuning/finetune.html) |
| 常见问题 | [FAQ](https://voxcpm.readthedocs.io/zh-cn/latest/faq.html) |

---

## 🌟 生态与社区

| 项目 | 说明 |
|---|---|
| [**Nano-vLLM**](https://github.com/a710128/nanovllm-voxcpm) | 高吞吐快速 GPU 推理引擎 |
| [**vLLM-Omni**](https://github.com/vllm-project/vllm-omni) | 官方 vLLM 全模态服务（原生支持 VoxCPM2）— PagedAttention、OpenAI 兼容 API |
| [**VoxCPM.cpp**](https://github.com/bluryar/VoxCPM.cpp) | GGML/GGUF：CPU、CUDA、Vulkan 推理 |
| [**VoxCPM-ONNX**](https://github.com/bluryar/VoxCPM-ONNX) | ONNX 导出，支持 CPU 推理 |
| [**VoxCPMANE**](https://github.com/0seba/VoxCPMANE) | Apple Neural Engine 后端 |
| [**voxcpm_rs**](https://github.com/madushan1000/voxcpm_rs) | Rust 重新实现 |
| [**ComfyUI-VoxCPM**](https://github.com/wildminder/ComfyUI-VoxCPM) | ComfyUI 节点工作流 |
| [**ComfyUI_RH_VoxCPM**](https://github.com/HM-RunningHub/ComfyUI_RH_VoxCPM) | 面向 VoxCPM 2 的功能更完整的 ComfyUI 工作流，支持多说话人、LoRA 和自动 ASR |
| [**ComfyUI-VoxCPMTTS**](https://github.com/1038lab/ComfyUI-VoxCPMTTS) | ComfyUI TTS 扩展 |
| [**TTS WebUI**](https://github.com/rsxdalv/tts_webui_extension.vox_cpm) | 浏览器端 TTS 扩展 |

> 完整生态见[文档](https://voxcpm.readthedocs.io/zh-cn/latest/)。社区项目非 OpenBMB 官方维护。做了什么有趣的东西？[提 Issue 或 PR](https://github.com/OpenBMB/VoxCPM/issues) 把它加进来！

---

## ⚠️ 风险与局限性

- **滥用风险：** VoxCPM 的声音克隆能力可生成高度逼真的合成语音。**严禁**将 VoxCPM 用于冒充他人、欺诈或虚假信息传播。我们强烈建议对所有 AI 生成的内容进行明确标注。
- **可控生成稳定性：** 音色设计和可控声音克隆的结果可能因生成次数而异 — 建议尝试生成 1~3 次以获得理想的音色或风格。我们正在积极提升可控性的一致性。
- **语言覆盖：** VoxCPM2 官方支持 30 种语言。对于未列入的语言，欢迎直接测试或使用自有数据进行微调。我们计划在未来版本中扩展语言覆盖。
- **使用说明：** 本模型基于 Apache-2.0 协议发布。用于生产部署时，我们建议针对具体场景进行充分的测试和安全评估。

---

## 📖 引用

如果 VoxCPM 对您有帮助，请考虑引用我们的工作并为仓库加星 ⭐！

```bib
@article{zhou2026voxcpm2,
  title   = {VoxCPM2 Technical Report},
  author  = {Zhou, Yixuan  and Zeng, Guoyang and Liu, Xin and Li, Xiang and Yu, Renjie and Gui, Jiancheng and Wu, Jiaheng and Wang, Ziyang and Shen, Xudong and Ye, Runchuan  and Zhang, Zhisheng and Zhou, Jiuyang and Bai, Bingsong and Sun, Weiyue and Deng, Mengyuan and Shi, Qundong and Wu, Zhiyong and Liu, Zhiyuan},
  journal = {arXiv preprint arXiv:2606.06928},
  year    = {2026},
}

@article{zhou2025voxcpm,
  title = {Voxcpm: Tokenizer-free TTS for context-aware speech generation and true-to-life voice cloning},
  author = {Zhou, Yixuan and Zeng, Guoyang and Liu, Xin and Li, Xiang and Yu, Renjie and Wang, Ziyang and Ye, Runchuan and Sun, Weiyue and Gui, Jiancheng and Li, Kehan and Wu, Zhiyong and Liu, Zhiyuan},
  journal = {arXiv preprint arXiv:2509.24650},
  year = {2025}
}
```

## 📄 许可证

VoxCPM 模型权重和代码基于 [Apache-2.0](LICENSE) 协议开源。

## 🙏 致谢

- [DiTAR](https://arxiv.org/abs/2502.03930) 扩散自回归骨干架构
- [MiniCPM-4](https://github.com/OpenBMB/MiniCPM) 语言模型基座
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 基于 Flow Matching 的 LocDiT 实现
- [DAC](https://github.com/descriptinc/descript-audio-codec) Audio VAE 骨干
- 感谢所有社区用户试用 VoxCPM、反馈问题、分享想法和贡献——你们的支持让项目持续进步

## 机构

<p>
  <a href="https://modelbest.cn/"><img src="assets/modelbest_logo.png" width="28px"> 面壁智能</a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://github.com/thuhcsi"><img src="assets/thuhcsi_logo.png" width="28px"> 清华大学人机交互实验室</a>
</p>

## ⭐ Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=OpenBMB/VoxCPM&type=Date)](https://star-history.com/#OpenBMB/VoxCPM&Date)
