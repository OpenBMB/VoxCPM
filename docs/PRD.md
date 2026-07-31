# VoxCPM 产品需求文档 (PRD)

> **版本**: 1.0 | **日期**: 2026-07-20 | **状态**: 已实现

---

## 1. 产品概述

### 1.1 产品定位

VoxCPM 是一个**无离散音频分词器（Tokenizer-Free）**的文本转语音（TTS）系统，通过端到端的**扩散自回归架构**直接生成连续语音表征。最新版本 **VoxCPM2** 拥有 20 亿参数，在超过 200 万小时的多语言语音数据上训练，支持 30 种语言 + 9 种中文方言，原生输出 48kHz 高质量音频。

### 1.2 核心价值

| 特性 | 描述 |
|------|------|
| **多语言** | 30 种全球语言 + 9 种中国方言，无需语言标签 |
| **音色设计** | 用自然语言描述凭空创造全新音色 |
| **声音克隆** | 从短音频克隆任意声音，可叠加风格控制 |
| **极致克隆** | 音频续写还原声音细节特征 |
| **48kHz 高品质** | 非对称编解码设计，内置超分能力 |
| **完全开源** | Apache-2.0 协议，免费商用 |

### 1.3 目标用户

- 需要高质量语音合成的开发者
- 需要声音克隆和定制音色的内容创作者
- 需要多语言/方言语音的出海业务
- 需要端侧部署的设备厂商

---

## 2. 模型版本对比

| | **VoxCPM2** | **VoxCPM1.5** | **VoxCPM-0.5B** |
|---|:---:|:---:|:---:|
| 状态 | 最新版本 | 稳定版 | 旧版 |
| 参数量 | 2B | 0.6B | 0.5B |
| 输出采样率 | 48kHz | 44.1kHz | 16kHz |
| 语言数 | 30 + 9 方言 | 2 (中/英) | 2 (中/英) |
| 音色设计 | 支持 | 不支持 | 不支持 |
| 可控克隆 | 支持 | 不支持 | 不支持 |
| 参考音频克隆(无需文本) | 支持 | 不支持 | 不支持 |
| 音频续写克隆 | 支持 | 支持 | 支持 |
| SFT / LoRA 微调 | 支持 | 支持 | 支持 |
| RTF (RTX 4090) | ~0.30 | ~0.15 | ~0.17 |

---

## 3. 功能模块

### 3.1 音色设计 (Voice Design)

**目标**: 无需参考音频，用自然语言描述创造全新音色。

**使用方式**: 在文本开头用括号写入音色描述：

```
(控制指令描述)目标合成文本
```

**支持的描述维度**:

| 维度 | 示例 |
|------|------|
| 性别 | 男性、女性 |
| 年龄 | 年轻、中老年、儿童 |
| 音色 | 温柔、低沉、沙哑、甜美、阴冷 |
| 情绪 | 暴躁、忧郁、兴奋、欢快、平静 |
| 语速 | 快速、缓慢、适中 |
| 方言 | 粤语、河南话、四川话 |

**示例**:

```
(年轻女性，声音温柔甜美)你好，欢迎使用VoxCPM2！
(中老年女性，声音低沉阴冷，语速缓慢而有力)哀家在这深宫待了四十年。
(Relaxed young male voice, slightly nasal, lazy drawl)Dude, did you see that set?
```

### 3.2 声音克隆 (Voice Cloning)

#### 3.2.1 参考音频克隆 (Controllable Cloning)

**目标**: 上传参考音频克隆音色，可叠加风格控制指令。

- **无需**提供参考音频的文本转录
- 通过 `ref_audio` tokens 结构化隔离参考音色
- 可叠加控制指令调节情绪/语速/风格

```python
wav = model.generate(
    text="这是VoxCPM2生成的克隆语音。",
    reference_wav_path="path/to/voice.wav",
)
```

#### 3.2.2 可控克隆 (Controllable Cloning with Style)

**目标**: 克隆音色同时用控制指令调节风格。

```python
wav = model.generate(
    text="(稍快一点，欢快的语气)这是带风格控制的克隆语音。",
    reference_wav_path="path/to/voice.wav",
)
```

#### 3.2.3 极致克隆 (Ultimate Cloning)

**目标**: 提供参考音频及其精确文本转录，通过音频续写实现最高保真度。

- 模型将参考音频视为已说出的前文，进行**音频续写**
- 控制指令与此模式**互斥**
- 为获得最高相似度，可将同一音频同时传给 `reference_wav_path` 和 `prompt_wav_path`

```python
wav = model.generate(
    text="这是极致克隆演示。",
    prompt_wav_path="path/to/voice.wav",
    prompt_text="参考音频的文本转录。",
    reference_wav_path="path/to/voice.wav",
)
```

#### 三种克隆模式对比

| 模式 | 需要参考音频 | 需要转录文本 | 支持控制指令 | 克隆保真度 | 版本支持 |
|------|:---:|:---:|:---:|:---:|:---:|
| 参考音频克隆 | 是 | 否 | 是 | 中 | 仅 VoxCPM2 |
| 可控克隆 | 是 | 否 | 是 | 中 | 仅 VoxCPM2 |
| 极致克隆 | 是 | 是 | 否 | 高 | 全部版本 |

### 3.3 流式生成 (Streaming)

**目标**: 边生成边输出，降低首包延迟。

```python
import numpy as np

chunks = []
for chunk in model.generate_streaming(text="流式语音合成"):
    chunks.append(chunk)
wav = np.concatenate(chunks)
```

**性能指标**:

| 平台 | RTF (标准) | RTF (Nano-vLLM) |
|------|:---:|:---:|
| RTX 4090 | ~0.30 | ~0.13 |

### 3.4 微调 (Fine-tuning)

#### 3.4.1 全参数微调 (SFT)

通过 `conf/voxcpm_v2/voxcpm_finetune_all.yaml` 配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| learning_rate | 1e-5 | 学习率 |
| batch_size | 2 | 批次大小 (grad_accum=8) |
| num_iters | - | 迭代次数 |
| max_steps | - | 最大步数 |
| warmup_steps | - | 预热步数 |

#### 3.4.2 LoRA 微调

通过 `conf/voxcpm_v2/voxcpm_finetune_lora.yaml` 配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| r | 32 | LoRA rank |
| alpha | 32 | LoRA alpha |
| dropout | 0.0 | LoRA dropout |
| enable_lm | true | 对 LM 层应用 LoRA |
| enable_dit | true | 对 DiT 层应用 LoRA |
| enable_proj | false | 对投影层应用 LoRA |

#### 3.4.3 训练数据格式

JSONL 格式，每行一个 JSON 对象：

```json
{"audio": "path/to/audio.wav", "text": "对应的文本转录"}
{"audio": "path/to/audio2.wav", "text": "带时长标注", "duration": 3.5}
```

#### 3.4.4 LoRA 微调 WebUI

端口 7860，包含两个标签页：

- **训练标签页**: 配置训练参数、启动/停止训练、实时日志
- **推理标签页**: 加载 LoRA 检查点、热切换不同 LoRA 模型、合成语音

### 3.5 时间戳对齐 (Timestamps)

**目标**: 为生成的音频提供文本-时间对齐信息。

| 粒度 | 说明 |
|------|------|
| `segment` | 段级时间戳 |
| `word` | 词级时间戳（默认） |
| `char` | 字符级时间戳（best-effort） |

**安装**: `pip install "voxcpm[timestamps]"`

**CLI 使用**:
```bash
voxcpm design --text "欢迎使用VoxCPM2。" --output out.wav \
    --timestamps --timestamp-level word --timestamp-language zh
```

**输出格式**: JSON 文件，包含每个文本片段的开始/结束时间。

### 3.6 文本规范化

基于 `wetext` 库，规范化数字、日期、缩写等：

```bash
voxcpm design --text "今天是2024年3月15日" --output out.wav --normalize
```

### 3.7 音频降噪

基于 ZipEnhancer (`iic/speech_zipenhancer_ans_multiloss_16k_base`)，对参考/提示音频进行降噪增强：

```python
wav = model.generate(
    text="...",
    reference_wav_path="noisy_ref.wav",
    denoise=True,
)
```

---

## 4. 接口规格

### 4.1 Python API

#### 4.1.1 模型加载

```python
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained(
    hf_model_id="openbmb/VoxCPM2",   # 或本地路径
    load_denoiser=True,                # 是否加载降噪器
    optimize=True,                      # 是否 torch.compile 优化
    device="auto",                      # auto/cpu/mps/cuda
    lora_config=None,                   # LoRA 配置
    lora_weights_path=None,             # LoRA 权重路径
)
```

#### 4.1.2 生成参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `text` | str | (必填) | 目标文本 |
| `prompt_wav_path` | str | None | 提示音频路径（续写模式） |
| `prompt_text` | str | None | 提示音频转录文本 |
| `reference_wav_path` | str | None | 参考音频路径（克隆模式） |
| `cfg_value` | float | 2.0 | CFG 引导尺度 |
| `inference_timesteps` | int | 10 | 扩散推理步数 |
| `min_len` | int | 2 | 最小音频长度 |
| `max_len` | int | 4096 | 最大 token 长度 |
| `normalize` | bool | False | 文本规范化 |
| `denoise` | bool | False | 音频降噪 |
| `seed` | int | None | 随机种子 |

#### 4.1.3 LoRA 管理

| 方法 | 说明 |
|------|------|
| `load_lora(path)` | 加载 LoRA 权重 |
| `unload_lora()` | 卸载 LoRA 权重 |
| `set_lora_enabled(bool)` | 启用/禁用 LoRA |
| `get_lora_state_dict()` | 获取 LoRA 参数 |
| `lora_enabled` | 检查 LoRA 是否已配置 |

### 4.2 CLI 工具

#### 4.2.1 子命令

| 命令 | 用途 | 必填参数 |
|------|------|---------|
| `voxcpm design` | 音色设计合成 | `--text`, `--output` |
| `voxcpm clone` | 声音克隆 | `--text`, `--output`, `--reference-audio` |
| `voxcpm batch` | 批量合成 | `--input`, `--output-dir` |
| `voxcpm validate` | 验证训练数据 | `--manifest` |

#### 4.2.2 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--text` / `-t` | - | 目标文本 |
| `--control` | None | 控制指令 |
| `--cfg-value` | 2.0 | CFG 引导尺度 (0.1-10.0) |
| `--inference-timesteps` | 10 | 推理步数 (1-100) |
| `--normalize` | False | 文本规范化 |
| `--seed` | None | 随机种子 |
| `--device` | auto | 运行设备 |
| `--model-path` | None | 本地模型路径 |
| `--no-denoiser` | False | 禁用降噪器 |
| `--no-optimize` | False | 禁用优化 |

#### 4.2.3 克隆专用参数

| 参数 | 说明 |
|------|------|
| `--reference-audio` / `-ra` | 参考音频路径 |
| `--prompt-audio` / `-pa` | 提示音频路径 |
| `--prompt-text` / `-pt` | 提示音频转录文本 |
| `--denoise` | 降噪参考/提示音频 |

### 4.3 Web Demo (app.py)

#### 4.3.1 后端模式

| 后端 | 参数 | 特点 |
|------|------|------|
| **Python** | `--backend python` | 完整功能，含 ASR 和降噪；MPS 上慢 |
| **CLI** | `--backend cli` | Metal 加速，快 15-20x；不支持 ASR |

#### 4.3.2 启动命令

**Python 后端**:
```bash
python app.py --model-id ./pretrained_models/VoxCPM2 --device mps --no-denoiser --port 8808
```

**CLI 后端**:
```bash
python app.py \
    --backend cli \
    --cli-bin /path/to/voxcpm2-cli \
    --cli-base-lm /path/to/VoxCPM2-BaseLM-Q8_0.gguf \
    --cli-acoustic /path/to/VoxCPM2-Acoustic-F16.gguf \
    --cli-timesteps 7 \
    --port 8808
```

#### 4.3.3 UI 组件

| 组件 | 说明 |
|------|------|
| 参考音频上传 | 支持上传/麦克风录制 |
| 极致克隆开关 | 开启后显示转录文本输入框 |
| 转录文本 | 参考音频文本转录（Python 模式自动 ASR） |
| 控制指令 | 音色描述文本 |
| 目标文本 | 要合成的文本 |
| 高级设置 | CFG、步数、种子、降噪、文本规范化 |
| 生成按钮 | 触发语音合成 |
| 音频输出 | 播放和下载生成结果 |

#### 4.3.4 高级设置

| 设置 | 范围 | 默认值 |
|------|------|--------|
| CFG (引导尺度) | 1.0 - 3.0 | 2.0 |
| LocDiT 步数 | 1 - 50 | 10 |
| 种子 | 0 - 2^32 | 随机 |
| 参考音频降噪 | 开/关 | 关 |
| 文本规范化 | 开/关 | 关 |

### 4.4 llama.cpp-omni CLI

#### 4.4.1 命令格式

```bash
voxcpm2-cli [options] <BaseLM.gguf> <Acoustic.gguf>
```

#### 4.4.2 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-t` / `--text` | - | 目标文本 |
| `-o` / `--output` | output.wav | 输出文件 |
| `-r` / `--reference` | - | 参考音频（克隆模式） |
| `--prompt-wav` | - | 提示音频（极致克隆） |
| `--prompt-text` | - | 提示音频转录文本 |
| `--timesteps` | 10 | CFM 推理步数 |
| `--cfg` | 2.0 | CFG 引导尺度 |
| `--seed` | 42 | 随机种子 |
| `--steps` | 200 | 最大解码步数 |
| `--cpu` | False | 使用 CPU 后端 |

#### 4.4.3 GGUF 权重

| 文件 | 量化 | 大小 | 说明 |
|------|------|------|------|
| VoxCPM2-BaseLM-F16.gguf | F16 | 3.0 GB | 全精度 LM |
| VoxCPM2-BaseLM-Q8_0.gguf | Q8_0 | 1.6 GB | 量化 LM（推荐） |
| VoxCPM2-Acoustic-F16.gguf | F16 | 1.7 GB | 声学模型 |

---

## 5. 多语言支持

### 5.1 支持的 30 种语言

阿拉伯语、缅甸语、中文、丹麦语、荷兰语、英语、芬兰语、法语、德语、希腊语、希伯来语、印地语、印尼语、意大利语、日语、高棉语、韩语、老挝语、马来语、挪威语、波兰语、葡萄牙语、俄语、西班牙语、斯瓦希里语、瑞典语、菲律宾语、泰语、土耳其语、越南语

### 5.2 支持的 9 种中国方言

四川话、粤语、吴语、东北话、河南话、陕西话、山东话、天津话、闽南话

### 5.3 使用方式

无需语言标签，直接输入原始文本即可合成。方言通过控制指令描述 + 方言文本词汇实现。

---

## 6. 音频规格

### 6.1 输入/输出

| 版本 | 参考音频输入 | 输出采样率 | 输出格式 |
|------|:---:|:---:|:---:|
| VoxCPM2 | 16kHz | **48kHz** | WAV (float32) |
| VoxCPM1.5 | 44.1kHz | 44.1kHz | WAV (float32) |
| VoxCPM-0.5B | 16kHz | 16kHz | WAV (float32) |

### 6.2 AudioVAE V2 设计

VoxCPM2 采用**非对称编解码**设计：
- 编码器：16kHz 输入 → 64 维 latent
- 解码器：64 维 latent → 48kHz 输出
- 内置超分能力，低质量输入也能输出高质量音频

---

## 7. 性能指标

### 7.1 各平台 RTF

| 平台 | Python (标准) | Nano-vLLM | llama.cpp-omni |
|------|:---:|:---:|:---:|
| RTX 4090 | ~0.30 | ~0.13 | - |
| Apple M4 (Metal) | ~60 (float32) | - | ~3.6 (timesteps=7) |

### 7.2 显存/内存占用

| 模型 | 显存 (GPU) | 内存 (Mac) |
|------|:---:|:---:|
| VoxCPM2 (Python) | ~8 GB | ~8 GB (float32) |
| VoxCPM2 (GGUF Q8_0) | - | ~3.3 GB |

---

## 8. 生态系统

| 项目 | 说明 | 适用场景 |
|------|------|---------|
| Nano-vLLM-VoxCPM | 高吞吐 GPU 推理引擎 | 生产并发服务 |
| vLLM-Omni | 官方 vLLM 全模态服务，OpenAI 兼容 API | 生产多租户部署 |
| llama.cpp-omni | C++ 推理引擎 (CPU/Metal/CUDA/Vulkan) | 端侧/消费级硬件 |
| ComfyUI-VoxCPM | ComfyUI 节点 | 创意工作流 |
| VoxCPM-ONNX | ONNX 导出 | 跨平台 CPU |

---

## 9. 环境要求

### 9.1 Python 方式

- Python >= 3.10 (< 3.13)
- PyTorch >= 2.5.0
- CUDA >= 12.0（GPU）/ MPS（Apple Silicon）

### 9.2 llama.cpp-omni 方式

- C++ 编译器 (CMake)
- macOS: 自动启用 Metal
- Linux: 自动检测 CUDA

### 9.3 安装

```bash
pip install voxcpm                    # 基础安装
pip install "voxcpm[timestamps]"      # 含时间戳功能
pip install "voxcpm[dev]"             # 含开发工具
```

---

## 10. 许可证

Apache-2.0，免费商用。
