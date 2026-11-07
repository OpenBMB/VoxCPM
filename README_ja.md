<h2 align="center">VoxCPM2：多言語音声生成・クリエイティブなボイスデザイン・本物そっくりのクローンを実現するトークナイザーフリー TTS</h2>

<p align="center">
  <a href="./README.md">English</a> | <a href="./README_zh.md">中文</a> | <b>日本語</b>
</p>

<p align="center">
  <a href="https://github.com/OpenBMB/VoxCPM/"><img src="https://img.shields.io/badge/Project%20Page-GitHub-blue" alt="Project Page"></a>
  <a href="https://huggingface.co/spaces/OpenBMB/VoxCPM-Demo"><img src="https://img.shields.io/badge/Live%20Playground-Demo-orange" alt="Live Playground"></a>
  <a href="https://voxcpm.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Docs-ReadTheDocs-8CA1AF" alt="Documentation"></a>
  <a href="https://huggingface.co/openbmb/VoxCPM2"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VoxCPM2-yellow" alt="Hugging Face"></a>
  <a href="https://modelscope.cn/models/OpenBMB/VoxCPM2"><img src="https://img.shields.io/badge/ModelScope-VoxCPM2-purple" alt="ModelScope"></a>
  <a href="https://openbmb.github.io/voxcpm2-demopage/"><img src="https://img.shields.io/badge/DemoPage-Audio Samples-red" alt="DemoPage"></a>
  <a href="https://arxiv.org/abs/2606.06928"><img src="https://img.shields.io/badge/arXiv-VoxCPM2%20Technical%20Report-red" alt="VoxCPM2 Technical Report"></a>
</p>

<div align="center">
  <img src="assets/voxcpm_logo.png" alt="VoxCPM Logo" width="35%">
  <br><br>
  <a href="https://trendshift.io/repositories/17704" target="_blank"><img src="https://trendshift.io/api/badge/repositories/17704" alt="OpenBMB%2FVoxCPM | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<br>

<p align="center">
  👋 ディスカッションとサポートのためのコミュニティにぜひご参加ください！
  <br>
  <a href="./assets/feishu-group.png" style="display:inline-block;vertical-align:middle; margin-left: 10px;">
    <img src="./assets/feishu-logo.png" width="16" height="16" style="vertical-align:middle;"> Feishu
  </a>
  &nbsp;|&nbsp;
  <a href="https://discord.gg/KZUx7tVNwz" style="display:inline-block;vertical-align:middle;">
    <img src="./assets/discord-logo.png" width="16" height="16" style="vertical-align:middle;"> Discord
  </a>
  &nbsp;|&nbsp;
  <a href="https://modelbest.feishu.cn/wiki/UtWxwcERfiRIpIkBOjuc3h9tn1D?fromScene=spaceOverview" style="display:inline-block;vertical-align:middle;">
    📚 MiniCPM Wiki
  </a>
</p>

VoxCPM は**トークナイザーフリー**のテキスト読み上げ（Text-to-Speech）システムです。エンドツーエンドの**拡散自己回帰アーキテクチャ**によって連続的な音声表現を直接生成し、離散的なトークン化を経ないことで、非常に自然で表現力豊かな音声合成を実現します。

**VoxCPM2** は最新のメジャーリリースです。**200 万時間以上**の多言語音声データで学習した **2B** パラメータのモデルで、新たに **30 言語**、**ボイスデザイン**、**制御可能なボイスクローン**、**48kHz** のスタジオ品質の音声出力に対応しました。[MiniCPM-4](https://github.com/OpenBMB/MiniCPM) をバックボーンとして構築されています。

### ✨ ハイライト

- 🌍 **30 言語の多言語対応** — 対応する 30 言語のいずれかでテキストを入力すれば、言語タグなしでそのまま合成できます
- 🎨 **ボイスデザイン** — 自然言語による説明（性別、年齢、声色、感情、話す速さ…）だけでまったく新しい声を作成できます。参照音声は不要です
- 🎛️ **制御可能なクローン** — 短い参照クリップから任意の声をクローンできます。元の音色を保ちながら、オプションのスタイル指示で感情、速さ、表現を調整できます
- 🎙️ **究極のクローン** — 声のあらゆるニュアンスを再現します。参照音声とその書き起こしの両方を与えると、モデルは参照から途切れなく続きを生成し、音色、リズム、感情、スタイルといった声の細部をすべて忠実に保ちます（VoxCPM1.5 と同じ方式）
- 🔊 **48kHz の高品質音声** — 16kHz の参照音声を受け取り、AudioVAE V2 の非対称なエンコード／デコード設計によって 48kHz のスタジオ品質の音声を直接出力します。超解像を内蔵しているため、外部のアップサンプラーは不要です
- 🧠 **文脈を考慮した合成** — テキストの内容から適切な韻律と表現力を自動的に推定します
- ⚡ **リアルタイムストリーミング** — NVIDIA RTX 4090 で RTF は約 0.3 まで下がり、[Nano-vLLM](https://github.com/a710128/nanovllm-voxcpm) または [vLLM-Omni](https://github.com/vllm-project/vllm-omni)（PagedAttention と OpenAI 互換 API を備えた、VoxCPM2 向けの公式 vLLM オムニモーダルサービング）で高速化すると約 0.13 になります
- 📜 **完全オープンソースで商用利用可能** — 重みとコードは [Apache-2.0](LICENSE) ライセンスで公開されており、商用利用も無料です

**🌍 対応言語（30）**  
アラビア語、ビルマ語、中国語、デンマーク語、オランダ語、英語、フィンランド語、フランス語、ドイツ語、ギリシャ語、ヘブライ語、ヒンディー語、インドネシア語、イタリア語、日本語、クメール語、韓国語、ラオ語、マレー語、ノルウェー語、ポーランド語、ポルトガル語、ロシア語、スペイン語、スワヒリ語、スウェーデン語、タガログ語、タイ語、トルコ語、ベトナム語

中国語の方言：四川话, 粤语, 吴语, 东北话, 河南话, 陕西话, 山东话, 天津话, 闽南话

### ニュース

- **[2026.04]** 🔥 **VoxCPM2** をリリースしました — 2B、30 言語、ボイスデザインと制御可能なボイスクローン、48kHz の音声出力！ [重み](https://huggingface.co/openbmb/VoxCPM2) | [ドキュメント](https://voxcpm.readthedocs.io/en/latest/) | [Playground](https://huggingface.co/spaces/OpenBMB/VoxCPM-Demo) | [テクニカルレポート](https://arxiv.org/abs/2606.06928)
- **[2025.12]** 🎉 SFT と LoRA ファインチューニングに対応した **VoxCPM1.5** の[重み](https://huggingface.co/openbmb/VoxCPM1.5)をオープンソース化しました。（**🏆 GitHub Trending 1 位**）
- **[2025.09]** 🔥 VoxCPM の[テクニカルレポート](https://arxiv.org/abs/2509.24650)を公開しました。
- **[2025.09]** 🎉 **VoxCPM-0.5B** の[重み](https://huggingface.co/openbmb/VoxCPM-0.5B)をオープンソース化しました（**🏆 HuggingFace Trending 1 位**）

---

## 目次

- [クイックスタート](#-クイックスタート)
  - [インストール](#インストール)
  - [Python API](#python-api)
  - [CLI の使い方](#cli-の使い方)
  - [Web デモ](#web-デモ)
  - [本番デプロイ](#-本番デプロイnano-vllm)
  - [オンデバイス推論（llama.cpp-omni）](#-オンデバイス推論llamacpp-omni)
- [モデルとバージョン](#-モデルとバージョン)
- [パフォーマンス](#-パフォーマンス)
- [ファインチューニング](#%EF%B8%8F-ファインチューニング)
- [ドキュメント](#-ドキュメント)
- [エコシステムとコミュニティ](#-エコシステムとコミュニティ)
- [リスクと制限事項](#%EF%B8%8F-リスクと制限事項)
- [引用](#-引用)

---

## 🚀 クイックスタート

### インストール

```sh
pip install voxcpm
```

> **動作要件：** Python ≥ 3.10（<3.13）、PyTorch ≥ 2.5.0、CUDA ≥ 12.0。詳しくは[クイックスタートのドキュメント](https://voxcpm.readthedocs.io/en/latest/quickstart.html)を参照してください。

### Python API

#### 🗣️ テキスト読み上げ

```python
from voxcpm import VoxCPM
import soundfile as sf

model = VoxCPM.from_pretrained(
  "openbmb/VoxCPM2",
  load_denoiser=False,
)

wav = model.generate(
    text="VoxCPM2 is the current recommended release for realistic multilingual speech synthesis.",
    cfg_value=2.0,
    inference_timesteps=10,
    seed=42,
)
sf.write("demo.wav", wav, model.tts_model.sample_rate)
print("saved: demo.wav")
```

先に ModelScope からダウンロードしたい場合は、次のようにします：

```bash
pip install modelscope
```

```python
from modelscope import snapshot_download
snapshot_download("OpenBMB/VoxCPM2", local_dir='./pretrained_models/VoxCPM2') # specify the local directory to save the model

from voxcpm import VoxCPM
import soundfile as sf
model = VoxCPM.from_pretrained("./pretrained_models/VoxCPM2", load_denoiser=False)

wav = model.generate(
    text="VoxCPM2 is the current recommended release for realistic multilingual speech synthesis.",
    cfg_value=2.0,
    inference_timesteps=10,
    seed=42,
)
sf.write("demo.wav", wav, model.tts_model.sample_rate)
```

#### 🎨 ボイスデザイン

自然言語による説明から声を作成します。参照音声は不要です。**形式：** 説明を `text` の先頭に括弧で囲んで入れます（例：`"(your voice description)The text to synthesize."`）：

```python
wav = model.generate(
    text="(A young woman, gentle and sweet voice)Hello, welcome to VoxCPM2!",
    cfg_value=2.0,
    inference_timesteps=10,
    seed=42,
)
sf.write("voice_design.wav", wav, model.tts_model.sample_rate)
```

#### 🎛️ 制御可能なボイスクローン

参照音声をアップロードします。モデルが音色をクローンし、さらに制御指示を使って速さ、感情、スタイルを調整することもできます。

```python
wav = model.generate(
    text="This is a cloned voice generated by VoxCPM2.",
    reference_wav_path="path/to/voice.wav",
)
sf.write("clone.wav", wav, model.tts_model.sample_rate)

wav = model.generate(
    text="(slightly faster, cheerful tone)This is a cloned voice with style control.",
    reference_wav_path="path/to/voice.wav",
    cfg_value=2.0,
    inference_timesteps=10,
    seed=42,
)
sf.write("controllable_clone.wav", wav, model.tts_model.sample_rate)
```

#### 🎙️ 究極のクローン

参照音声とその正確な書き起こしの両方を与えると、音声の続きを生成する方式のクローンにより、声のあらゆるニュアンスが再現されます。クローンの類似度を最大にするには、以下のように同じ参照クリップを `reference_wav_path` と `prompt_wav_path` の両方に渡します：

```python
wav = model.generate(
    text="This is an ultimate cloning demonstration using VoxCPM2.",
    prompt_wav_path="path/to/voice.wav",
    prompt_text="The transcript of the reference audio.",
    reference_wav_path="path/to/voice.wav", # optional, for better simliarity 
)
sf.write("hifi_clone.wav", wav, model.tts_model.sample_rate)
```

**🔄 ストリーミング API**

```python
import numpy as np

chunks = []
for chunk in model.generate_streaming(
    text="Streaming text to speech is easy with VoxCPM!",
):
    chunks.append(chunk)
wav = np.concatenate(chunks)
sf.write("streaming.wav", wav, model.tts_model.sample_rate)
```



### CLI の使い方

```bash
# Voice design (no reference audio needed)
voxcpm design \
  --text "VoxCPM2 brings studio-quality multilingual speech synthesis." \
  --output out.wav

# Controllable voice cloning with style control
voxcpm design \
  --text "VoxCPM2 brings studio-quality multilingual speech synthesis." \
  --control "Young female voice, warm and gentle, slightly smiling" \
  --seed 42 \
  --output out.wav

# Voice cloning (reference audio)
voxcpm clone \
  --text "This is a voice cloning demo." \
  --reference-audio path/to/voice.wav \
  --output out.wav

# Ultimate cloning (prompt audio + transcript)
voxcpm clone \
  --text "This is a voice cloning demo." \
  --prompt-audio path/to/voice.wav \
  --prompt-text "reference transcript" \
  --reference-audio path/to/voice.wav \ # optional, for better simliarity
  --output out.wav

# Batch processing
voxcpm batch --input examples/input.txt --output-dir outs

# Optional post-generation timestamps with stable-ts
pip install "voxcpm[timestamps]"
voxcpm design \
  --text "VoxCPM2 brings studio-quality multilingual speech synthesis." \
  --output out.wav \
  --timestamps \
  --timestamp-level word \
  --timestamp-language en

# Character timestamps are best-effort and are derived from word alignment
voxcpm design \
  --text "欢迎使用 VoxCPM2。" \
  --output out.wav \
  --timestamps \
  --timestamp-level char \
  --timestamp-language zh

# Help
voxcpm --help
```

### Web デモ

```bash
python app.py --port 8808  # then open in browser: http://localhost:8808
```

`--device` で実行デバイスを選択します：

```bash
python app.py --device auto
```

指定できる値は `auto`、`cpu`、`mps`、`cuda`、`cuda:N` です。Apple Silicon の Mac では、`auto` は MPS が利用できる場合に MPS を使います。

### 🚢 本番デプロイ（Nano-vLLM）

高スループットなサービングには **[Nano-vLLM-VoxCPM](https://github.com/a710128/nanovllm-voxcpm)** を使ってください。Nano-vLLM をベースにした専用の推論エンジンで、同時リクエストと非同期 API に対応しています。

```bash
pip install nano-vllm-voxcpm
```

```python
from nanovllm_voxcpm import VoxCPM
import numpy as np, soundfile as sf

server = VoxCPM.from_pretrained(model="/path/to/VoxCPM", devices=[0])
chunks = list(server.generate(target_text="Hello from VoxCPM!"))
sf.write("out.wav", np.concatenate(chunks), 48000)
server.stop()
```

> **NVIDIA RTX 4090 で RTF は約 0.13 まで低下**します（標準の PyTorch 実装では約 0.3）。バッチ化された同時リクエストと FastAPI の HTTP サーバーにも対応しています。デプロイの詳細は [Nano-vLLM-VoxCPM リポジトリ](https://github.com/a710128/nanovllm-voxcpm)を参照してください。

### 🏭 本番サービング（vLLM-Omni）

本番環境でのマルチテナントデプロイには **[vLLM-Omni](https://github.com/vllm-project/vllm-omni)** を使ってください。公式 vLLM プロジェクトのオムニモーダル拡張で、**VoxCPM2** をネイティブにサポートしています。PagedAttention の KV キャッシュ、連続バッチ処理、そのまま置き換えて使える **OpenAI 互換**の `/v1/audio/speech` エンドポイントを備えています。

```bash
# Install from source (latest main — vllm-omni is rapidly evolving)
uv pip install vllm==0.19.0 --torch-backend=auto
git clone https://github.com/vllm-project/vllm-omni.git && cd vllm-omni
uv pip install -e .
```

その他のプラットフォーム（ROCm、XPU、MUSA、NPU）や Docker イメージについては、[vLLM-Omni のインストールガイド](https://vllm-omni.readthedocs.io/en/latest/getting_started/installation/)を参照してください。

```bash
# Launch an OpenAI-compatible TTS server (--omni enables omni-modal serving)
vllm serve openbmb/VoxCPM2 --omni --port 8000

# Call it from any OpenAI client
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"openbmb/VoxCPM2","input":"Hello from VoxCPM2 on vLLM-Omni!","voice":"default"}' \
  --output out.wav
```

> 上流の vLLM スケジューラーの上に構築されており、バッチ化された同時リクエスト、チャンク単位のストリーミング配信、マルチ GPU デプロイにすぐ対応できます。デプロイの完全な手順は [VoxCPM2 のサンプル](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/voxcpm2)を参照してください。

### 📱 オンデバイス推論（llama.cpp-omni）

Python を使わないオンデバイス／エッジでのデプロイには **[llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni)** を使ってください。llama.cpp をベースにした高性能な C++ 推論エンジンで、**CPU / Metal / CUDA / Vulkan** 上で VoxCPM2 の GGUF をネイティブにサポートしています。

**1. GGUF の重みをダウンロード**します：[HuggingFace](https://huggingface.co/DennisHuang648/VoxCPM2-GGUF) | [ModelScope](https://modelscope.cn/models/DennisHuang/VoxCPM2-GGUF)。**BaseLM**（F16 または Q8_0）を 1 つと **Acoustic** ファイルが必要です。Q8_0 を選ぶと、品質の低下はごくわずかなままダウンロード量が半分になります。

**2. ビルド**

```bash
git clone https://github.com/tc-mb/llama.cpp-omni.git && cd llama.cpp-omni
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target voxcpm2-cli -j
```

> CMake が Metal（macOS）または CUDA（NVIDIA GPU 搭載の Linux）を自動検出します。

**3. 実行**

```bash
# Basic TTS
./build/bin/voxcpm2-cli \
    -t "Hello, this is VoxCPM2 running through llama.cpp-omni." \
    -o output.wav VoxCPM2-BaseLM-Q8_0.gguf VoxCPM2-Acoustic-F16.gguf

# Voice cloning (reference audio)
./build/bin/voxcpm2-cli \
    -t "Cloned voice." -r speaker.wav -o clone.wav \
    VoxCPM2-BaseLM-Q8_0.gguf VoxCPM2-Acoustic-F16.gguf

# Ultimate cloning (reference audio + transcript)
./build/bin/voxcpm2-cli \
    -t "Target text." --prompt-wav speaker.wav --prompt-text "transcript of speaker.wav" \
    -o clone.wav VoxCPM2-BaseLM-Q8_0.gguf VoxCPM2-Acoustic-F16.gguf
```

> **Apple M4 Pro / Metal で RTF 約 1.76（Q8_0）。** 主なフラグ：`--cfg`（ガイダンススケール）、`--timesteps`（CFM ステップ数）、`--seed`、`--temperature`、`--stream`。詳しくは [llama.cpp-omni リポジトリ](https://github.com/tc-mb/llama.cpp-omni)と [GGUF の重みのページ](https://huggingface.co/DennisHuang648/VoxCPM2-GGUF)を参照してください。

> **パラメータの完全なリファレンス、さまざまなシナリオの例、ボイスクローンのコツ →** [クイックスタートガイド](https://voxcpm.readthedocs.io/en/latest/quickstart.html) | [使い方ガイド](https://voxcpm.readthedocs.io/en/latest/usage_guide.html) | [Cookbook](https://voxcpm.readthedocs.io/en/latest/cookbook.html)

---

## 📦 モデルとバージョン


|                                 | **VoxCPM2**                                                                                          | **VoxCPM1.5**                                                                                            | **VoxCPM-0.5B**                                                                                              |
| ------------------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **ステータス** | 🟢 最新 | 安定版 | レガシー |
| **バックボーンのパラメータ数** | 2B | 0.6B | 0.5B |
| **音声サンプルレート** | 48kHz | 44.1kHz | 16kHz |
| **LM トークンレート** | 6.25Hz | 6.25Hz | 12.5Hz |
| **言語数** | 30 | 2（zh、en） | 2（zh、en） |
| **クローンモード** | 独立した参照と続き生成 | 続き生成のみ | 続き生成のみ |
| **ボイスデザイン** | ✅ | — | — |
| **制御可能なボイスクローン** | ✅ | — | — |
| **SFT / LoRA**                  | ✅                                                                                                    | ✅                                                                                                        | ✅                                                                                                            |
| **RTF (RTX 4090)**              | ~0.30                                                                                                | ~0.15                                                                                                    | ~0.17                                                                                                        |
| **RTF in Nano-VLLM (RTX 4090)** | ~0.13                                                                                                | ~0.08                                                                                                    | ~0.10                                                                                                        |
| **VRAM** | ~8 GB | ~6 GB | ~5 GB |
| **重み** | [🤗 HF](https://huggingface.co/openbmb/VoxCPM2) / [MS](https://modelscope.cn/models/OpenBMB/VoxCPM2) | [🤗 HF](https://huggingface.co/openbmb/VoxCPM1.5) / [MS](https://modelscope.cn/models/OpenBMB/VoxCPM1.5) | [🤗 HF](https://huggingface.co/openbmb/VoxCPM-0.5B) / [MS](https://modelscope.cn/models/OpenBMB/VoxCPM-0.5B) |
| **テクニカルレポート** | [arXiv](https://arxiv.org/abs/2606.06928) | — | [arXiv](https://arxiv.org/abs/2509.24650) [ICLR 2026](https://openreview.net/forum?id=h5KLpGoqzC) |
| **デモページ** | [音声サンプル](https://openbmb.github.io/voxcpm2-demopage) | — | [音声サンプル](https://openbmb.github.io/VoxCPM-demopage) |


VoxCPM2 は**トークナイザーフリーの拡散自己回帰**パラダイムに基づいて構築されています。モデルは完全に **AudioVAE V2** の潜在空間で動作し、**LocEnc → TSLM → RALM → LocDiT** という 4 段階のパイプラインに従うことで、豊かな表現力と 48kHz のネイティブ音声出力を実現しています。

<div align="center">
  <img src="assets/voxcpm2_model.png" alt="VoxCPM2 モデルアーキテクチャ" width="90%">
</div>

> アーキテクチャの詳細、VoxCPM2 固有の改良点、モデルの比較表については、[アーキテクチャ設計](https://voxcpm.readthedocs.io/en/latest/models/architecture.html)を参照してください。

---

## 📊 パフォーマンス

VoxCPM2 は、公開されているゼロショット TTS および制御可能な TTS のベンチマークで、最先端または同等の結果を達成しています。 

### Seed-TTS-eval

**Seed-TTS-eval の WER(⬇)&SIM(⬆) の結果（クリックで展開）**


| モデル | パラメータ数 | オープンソース | test-EN |        | test-ZH |        | test-Hard |        |
| ----------------- | ---------- | ----------- | ------- | ------ | ------- | ------ | --------- | ------ |
|                   |            |             | WER/%⬇  | SIM/%⬆ | CER/%⬇  | SIM/%⬆ | CER/%⬇    | SIM/%⬆ |
| MegaTTS3          | 0.5B       | ❌           | 2.79    | 77.1   | 1.52    | 79.0   | -         | -      |
| DiTAR             | 0.6B       | ❌           | 1.69    | 73.5   | 1.02    | 75.3   | -         | -      |
| CosyVoice3        | 0.5B       | ❌           | 2.02    | 71.8   | 1.16    | 78.0   | 6.08      | 75.8   |
| CosyVoice3        | 1.5B       | ❌           | 2.22    | 72.0   | 1.12    | 78.1   | 5.83      | 75.8   |
| Seed-TTS          | -          | ❌           | 2.25    | 76.2   | 1.12    | 79.6   | 7.59      | 77.6   |
| MiniMax-Speech    | -          | ❌           | 1.65    | 69.2   | 0.83    | 78.3   | -         | -      |
| F5-TTS            | 0.3B       | ✅           | 2.00    | 67.0   | 1.53    | 76.0   | 8.67      | 71.3   |
| MaskGCT           | 1B         | ✅           | 2.62    | 71.7   | 2.27    | 77.4   | -         | -      |
| CosyVoice         | 0.3B       | ✅           | 4.29    | 60.9   | 3.63    | 72.3   | 11.75     | 70.9   |
| CosyVoice2        | 0.5B       | ✅           | 3.09    | 65.9   | 1.38    | 75.7   | 6.83      | 72.4   |
| SparkTTS          | 0.5B       | ✅           | 3.14    | 57.3   | 1.54    | 66.0   | -         | -      |
| FireRedTTS        | 0.5B       | ✅           | 3.82    | 46.0   | 1.51    | 63.5   | 17.45     | 62.1   |
| FireRedTTS-2      | 1.5B       | ✅           | 1.95    | 66.5   | 1.14    | 73.6   | -         | -      |
| Qwen2.5-Omni      | 7B         | ✅           | 2.72    | 63.2   | 1.70    | 75.2   | 7.97      | 74.7   |
| Qwen3-Omni        | 30B-A3B    | ✅           | 1.39    | -      | 1.07    | -      | -         | -      |
| OpenAudio-s1-mini | 0.5B       | ✅           | 1.94    | 55.0   | 1.18    | 68.5   | 23.37     | 64.3   |
| IndexTTS2         | 1.5B       | ✅           | 2.23    | 70.6   | 1.03    | 76.5   | 7.12      | 75.5   |
| VibeVoice         | 1.5B       | ✅           | 3.04    | 68.9   | 1.16    | 74.4   | -         | -      |
| HiggsAudio-v2     | 3B         | ✅           | 2.44    | 67.7   | 1.50    | 74.0   | 55.07     | 65.6   |
| VoxCPM-0.5B       | 0.6B       | ✅           | 1.85    | 72.9   | 0.93    | 77.2   | 8.87      | 73.0   |
| VoxCPM1.5         | 0.8B       | ✅           | 2.12    | 71.4   | 1.18    | 77.0   | 7.74      | 73.1   |
| MOSS-TTS          |            | ✅           | 1.85    | 73.4   | 1.20    | 78.8   | -         | -      |
| Qwen3-TTS         | 1.7B       | ✅           | 1.23    | 71.7   | 1.22    | 77.0   | 6.76      | 74.8   |
| FishAudio S2      | 4B         | ✅           | 0.99    | -      | 0.54    | -      | 5.99      | -      |
| LongCat-Audio-DiT | 3.5B       | ✅           | 1.50    | 78.6   | 1.09    | 81.8   | 6.04      | 79.7   |
| **VoxCPM2**       | 2B         | ✅           | 1.84    | 75.3   | 0.97    | 79.5   | 8.13      | 75.3   |




### CV3-eval

**CV3-eval の多言語 WER/CER(⬇) の結果（クリックで展開）**


| モデル | zh   | en   | hard-zh | hard-en | ja   | ko   | de   | es   | fr   | it   | ru   |
| --------------- | ---- | ---- | ------- | ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CosyVoice2      | 4.08 | 6.32 | 12.58   | 11.96   | 9.13 | 19.7 | -    | -    | -    | -    | -    |
| CosyVoice3-1.5B | 3.91 | 4.99 | 9.77    | 10.55   | 7.57 | 5.69 | 6.43 | 4.47 | 11.8 | 10.5 | 6.64 |
| Fish Audio S2   | 2.65 | 2.43 | 9.10    | 4.40    | 3.96 | 2.76 | 2.22 | 2.00 | 6.26 | 2.04 | 2.78 |
| **VoxCPM2**     | 3.65 | 5.00 | 8.55    | 8.48    | 5.96 | 5.69 | 4.77 | 3.80 | 9.85 | 4.25 | 5.21 |




### MiniMax-Multilingual-Test

**Minimax-MLS-test の WER(⬇) の結果（クリックで展開）**


| 言語 | Minimax   | ElevenLabs | Qwen3-TTS | FishAudio S2 | **VoxCPM2** |
| ---------- | --------- | ---------- | --------- | ------------ | ----------- |
| アラビア語 | **1.665** | 1.666      | –         | 3.500        | 13.046      |
| 広東語 | 34.111    | 51.513     | –         | **30.670**   | 38.584      |
| 中国語 | 2.252     | 16.026     | 0.928     | **0.730**    | 1.136       |
| チェコ語 | 3.875     | **2.108**  | –         | 2.840        | 24.132      |
| オランダ語 | 1.143     | **0.803**  | –         | 0.990        | 0.913       |
| 英語 | 2.164     | 2.339      | **0.934** | 1.620        | 2.289       |
| フィンランド語 | 4.666     | 2.964      | –         | 3.330        | **2.632**   |
| フランス語 | 4.099     | 5.216      | **2.858** | 3.050        | 4.534       |
| ドイツ語 | 1.906     | 0.572      | 1.235     | **0.550**    | 0.679       |
| ギリシャ語 | 2.016     | **0.991**  | –         | 5.740        | 2.844       |
| ヒンディー語 | 6.962     | **5.827**  | –         | 14.640       | 19.699      |
| インドネシア語 | 1.237     | **1.059**  | –         | 1.460        | 1.084       |
| イタリア語 | 1.543     | 1.743      | **0.948** | 1.270        | 1.563       |
| 日本語 | 3.519     | 10.646     | 3.823     | **2.760**    | 4.628       |
| 韓国語 | 1.747     | 1.865      | 1.755     | **1.180**    | 1.962       |
| ポーランド語 | 1.415     | **0.766**  | –         | 1.260        | 1.141       |
| ポルトガル語 | 1.877     | 1.331      | 1.526     | **1.140**    | 1.938       |
| ルーマニア語 | 2.878     | **1.347**  | –         | 10.740       | 21.577      |
| ロシア語 | 4.281     | 3.878      | 3.212     | **2.400**    | 3.634       |
| スペイン語 | 1.029     | 1.084      | 1.126     | **0.910**    | 1.438       |
| タイ語 | 2.701     | 73.936     | –         | 4.230        | 2.961       |
| トルコ語 | 1.52      | 0.699      | –         | 0.870        | 0.817       |
| ウクライナ語 | 1.082     | **0.997**  | –         | 2.300        | 6.316       |
| ベトナム語 | **0.88**  | 73.415     | –         | 7.410        | 3.307       |




**Minimax-MLS-test の SIM(⬆) の結果（クリックで展開）**


| 言語 | Minimax  | ElevenLabs | Qwen3-TTS | FishAudio S2 | **VoxCPM2** |
| ---------- | -------- | ---------- | --------- | ------------ | ----------- |
| アラビア語 | 73.6     | 70.6       | –         | 75.0         | **79.1**    |
| 広東語 | 77.8     | 67.0       | –         | 80.5         | **83.5**    |
| 中国語 | 78.0     | 67.7       | 79.9      | 81.6         | **82.5**    |
| チェコ語 | 79.6     | 68.5       | –         | **79.8**     | 78.3        |
| オランダ語 | 73.8     | 68.0       | –         | 73.0         | **80.8**    |
| 英語 | 75.6     | 61.3       | 77.5      | 79.7         | **85.4**    |
| フィンランド語 | 83.5     | 75.9       | –         | 81.9         | **89.0**    |
| フランス語 | 62.8     | 53.5       | 62.8      | 69.8         | **73.5**    |
| ドイツ語 | 73.3     | 61.4       | 77.5      | 76.7         | **80.3**    |
| ギリシャ語 | 82.6     | 73.3       | –         | 79.5         | **86.0**    |
| ヒンディー語 | 81.8     | 73.0       | –         | 82.1         | **85.6**    |
| インドネシア語 | 72.9     | 66.0       | –         | 76.3         | **80.0**    |
| イタリア語 | 69.9     | 57.9       | 81.7      | 74.7         | **78.0**    |
| 日本語 | 77.6     | 73.8       | 78.8      | 79.6         | **82.8**    |
| 韓国語 | 77.6     | 70.0       | 79.9      | 81.7         | **83.3**    |
| ポーランド語 | 80.2     | 72.9       | –         | 81.9         | **88.4**    |
| ポルトガル語 | 80.5     | 71.1       | 81.7      | 78.1         | **83.7**    |
| ルーマニア語 | **80.9** | 69.9       | –         | 73.3         | 79.7        |
| ロシア語 | 76.1     | 67.6       | 79.2      | 79.0         | **81.1**    |
| スペイン語 | 76.2     | 61.5       | 81.4      | 77.6         | **83.1**    |
| タイ語 | 80.0     | 58.8       | –         | 78.6         | **84.0**    |
| トルコ語 | 77.9     | 59.6       | –         | 83.5         | **87.1**    |
| ウクライナ語 | 73.0     | 64.7       | –         | 74.7         | **79.8**    |
| ベトナム語 | 74.3     | 36.9       | –         | 74.0         | **80.6**    |




### Internal 30-Language ASR Benchmark

さらに、**30 言語 × 500 サンプル**による社内の多言語明瞭度ベンチマークも実施しています。ASR による書き起こしは **Gemini 3.1 Flash Lite API** で評価しています。

**社内 30 言語 ASR ベンチマーク（クリックで展開）**


| 言語 | 指標 | VoxCPM2   | Fish S2-Pro |
| ---------------------- | ------ | --------- | ----------- |
| ar (アラビア語) | CER    | 1.23%     | 0.30%       |
| da (デンマーク語) | WER    | 2.70%     | 3.52%       |
| de (ドイツ語) | WER    | 0.96%     | 0.64%       |
| el (ギリシャ語) | WER    | 3.17%     | 4.61%       |
| en (英語) | WER    | 0.42%     | 1.03%       |
| es (スペイン語) | WER    | 1.33%     | 0.64%       |
| fi (フィンランド語) | WER    | 2.24%     | 2.80%       |
| fr (フランス語) | WER    | 2.16%     | 2.34%       |
| he (ヘブライ語) | CER    | 2.98%     | 15.27%      |
| hi (ヒンディー語) | CER    | 0.79%     | 0.91%       |
| id (インドネシア語) | WER    | 1.36%     | 1.68%       |
| it (イタリア語) | WER    | 1.65%     | 1.08%       |
| ja (日本語) | CER    | 2.40%     | 1.82%       |
| km (クメール語) | CER    | 2.05%     | 75.15%      |
| ko (韓国語) | CER    | 0.95%     | 0.29%       |
| lo (ラオ語) | CER    | 1.90%     | 87.40%      |
| ms (マレー語) | WER    | 1.75%     | 1.41%       |
| my (ビルマ語) | CER    | 1.42%     | 85.27%      |
| nl (オランダ語) | WER    | 1.25%     | 1.68%       |
| no (ノルウェー語) | WER    | 2.49%     | 3.76%       |
| pl (ポーランド語) | WER    | 1.90%     | 1.65%       |
| pt (ポルトガル語) | WER    | 1.48%     | 1.49%       |
| ru (ロシア語) | WER    | 0.90%     | 0.86%       |
| sv (スウェーデン語) | WER    | 2.22%     | 2.63%       |
| sw (スワヒリ語) | CER    | 1.07%     | 2.02%       |
| th (タイ語) | CER    | 0.94%     | 1.92%       |
| tl (タガログ語) | WER    | 2.63%     | 4.00%       |
| tr (トルコ語) | WER    | 1.65%     | 1.65%       |
| vi (ベトナム語) | WER    | 1.56%     | 5.56%       |
| zh (中国語) | CER    | 0.92%     | 1.02%       |
| 平均（30 言語） |        | **1.68%** | -           |




### InstructTTSEval

**指示に基づくボイスデザインの結果（クリックで展開）**


| モデル | InstructTTSEval-ZH |          |          | InstructTTSEval-EN |          |          |
| ---------------------- | ------------------ | -------- | -------- | ------------------ | -------- | -------- |
|                        | APS⬆               | DSD⬆     | RP⬆      | APS⬆               | DSD⬆     | RP⬆      |
| Hume                   | –                  | –        | –        | 83.0               | 75.3     | 54.3     |
| VoxInstruct            | 47.5               | 52.3     | 42.6     | 54.9               | 57.0     | 39.3     |
| Parler-tts-mini        | –                  | –        | –        | 63.4               | 48.7     | 28.6     |
| Parler-tts-large       | –                  | –        | –        | 60.0               | 45.9     | 31.2     |
| PromptTTS              | –                  | –        | –        | 64.3               | 47.2     | 31.4     |
| PromptStyle            | –                  | –        | –        | 57.4               | 46.4     | 30.9     |
| VoiceSculptor          | 75.7               | 64.7     | 61.5     | –                  | –        | –        |
| Mimo-Audio-7B-Instruct | 75.7               | 74.3     | 61.5     | 80.6               | 77.6     | 59.5     |
| Qwen3TTS-12Hz-1.7B-VD  | **85.2**           | **81.1** | **65.1** | 82.9               | 82.4     | 68.4     |
| **VoxCPM2**            | **85.2**           | 71.5     | 60.8     | **84.2**           | **83.2** | **71.4** |




---

## ⚙️ ファインチューニング

VoxCPM は**フルファインチューニング（SFT）**と **LoRA ファインチューニング**の両方に対応しています。わずか **5〜10 分**の音声で、特定の話者、言語、ドメインに適応させることができます。

```bash
# LoRA fine-tuning (parameter-efficient, recommended)
python scripts/train_voxcpm_finetune.py \
    --config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml

# Full fine-tuning
python scripts/train_voxcpm_finetune.py \
    --config_path conf/voxcpm_v2/voxcpm_finetune_all.yaml

# WebUI for training & inference
python lora_ft_webui.py   # then open http://localhost:7860
```

> **完全なガイド →** [ファインチューニングガイド](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html)（データの準備、設定、学習、LoRA のホットスワップ、FAQ）

---

## 📚 ドキュメント

完全なドキュメント：**[voxcpm.readthedocs.io](https://voxcpm.readthedocs.io/en/latest/)**


| トピック | リンク |
| -------------------------- | ------------------------------------------------------------------------------------- |
| クイックスタートとインストール | [クイックスタート](https://voxcpm.readthedocs.io/en/latest/quickstart.html) |
| 使い方ガイドと Cookbook | [ユーザーガイド](https://voxcpm.readthedocs.io/en/latest/usage_guide.html) |
| VoxCPM シリーズ | [モデル](https://voxcpm.readthedocs.io/en/latest/models/version_history.html) |
| ファインチューニング（SFT と LoRA） | [ファインチューニングガイド](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html) |
| FAQ とトラブルシューティング | [FAQ](https://voxcpm.readthedocs.io/en/latest/faq.html) |


---

## 🌟 エコシステムとコミュニティ


| プロジェクト | 説明 |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **[Nano-vLLM](https://github.com/a710128/nanovllm-voxcpm)** | 高スループットで高速な GPU サービング |
| **[vLLM-Omni](https://github.com/vllm-project/vllm-omni)** | VoxCPM2 向けの公式 vLLM オムニモーダルサービング — PagedAttention、OpenAI 互換 API |
| **[llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni)** | 全二重のオムニ推論エンジン — CPU / Metal / CUDA / Vulkan 上で VoxCPM2 GGUF |
| **[VoxCPM.cpp](https://github.com/bluryar/VoxCPM.cpp)** | GGML/GGUF：CPU、CUDA、Vulkan での推論 |
| **[audio.cpp](https://github.com/0xShug0/audio.cpp)** | ggml ベースの統合 C++ 推論フレームワーク — CPU/CUDA/Vulkan/Metal、CLI とサーバー、Python 不要 |
| **[VoxCPM-ONNX](https://github.com/bluryar/VoxCPM-ONNX)** | CPU 推論用の ONNX エクスポート |
| **[VoxCPMANE](https://github.com/0seba/VoxCPMANE)** | Apple Neural Engine バックエンド |
| **[voxcpm_rs](https://github.com/madushan1000/voxcpm_rs)** | Rust による再実装 |
| **[ComfyUI-VoxCPM](https://github.com/wildminder/ComfyUI-VoxCPM)** | ComfyUI のノードベースのワークフロー |
| **[ComfyUI_RH_VoxCPM](https://github.com/HM-RunningHub/ComfyUI_RH_VoxCPM)** | 複数話者の生成、LoRA、自動 ASR を備えた、VoxCPM 2 向けの機能が揃った ComfyUI ワークフロー |
| **[ComfyUI-VoxCPMTTS](https://github.com/1038lab/ComfyUI-VoxCPMTTS)** | ComfyUI の TTS 拡張機能 |
| **[TTS WebUI](https://github.com/rsxdalv/tts_webui_extension.vox_cpm)** | ブラウザベースの TTS 拡張機能 |
| **[VoxaDub](https://github.com/kadirb4rut/voxadub)** | ローカルファーストの動画吹き替え：話者を考慮した VoxCPM2 による吹き替え、完全ローカルの Hy-MT2 翻訳、尺を考慮した同期、実験的なライブ吹き替え |


> エコシステムの全体は、ドキュメントの[エコシステム](https://voxcpm.readthedocs.io/en/latest/)を参照してください。コミュニティのプロジェクトは OpenBMB が公式にメンテナンスしているものではありません。何か面白いものを作りましたか？ 追加するには [issue または PR を作成](https://github.com/OpenBMB/VoxCPM/issues)してください！

---

## ⚠️ リスクと制限事項

- **悪用の可能性：** VoxCPM のボイスクローンは非常にリアルな合成音声を生成できます。なりすまし、詐欺、偽情報のために VoxCPM を使用することは**固く禁じられています**。AI が生成したコンテンツには、明確に表示することを強くおすすめします。
- **制御可能な生成の安定性：** ボイスデザインと制御可能なボイスクローンの結果は実行ごとに異なることがあります。望みの声やスタイルを得るには 1〜3 回生成してみてください。制御の一貫性の改善に積極的に取り組んでいます。
- **言語のカバー範囲：** VoxCPM2 は公式に 30 言語をサポートしています。リストにない言語についても、直接試したり、独自のデータでファインチューニングを試したりしていただいてかまいません。今後のリリースで対応言語を拡大する予定です。
- **利用について：** このモデルは Apache-2.0 ライセンスで公開されています。本番環境へのデプロイでは、ユースケースに合わせた十分なテストと安全性の評価を行うことをおすすめします。

---

## 📖 引用

VoxCPM が役に立った場合は、私たちの研究の引用と、リポジトリへのスター ⭐ をご検討ください！

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

## 📄 ライセンス

VoxCPM のモデルの重みとコードは [Apache-2.0](LICENSE) ライセンスでオープンソース化されています。

## 🙏 謝辞

- 拡散自己回帰のバックボーンを提供してくれた [DiTAR](https://arxiv.org/abs/2502.03930)
- 言語モデルの基盤を提供してくれた [MiniCPM-4](https://github.com/OpenBMB/MiniCPM)
- Flow Matching ベースの LocDiT 実装を提供してくれた [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- Audio VAE のバックボーンを提供してくれた [DAC](https://github.com/descriptinc/descript-audio-codec)
- VoxCPM を試し、問題を報告し、アイデアを共有し、貢献してくださったコミュニティのユーザーの皆さん — 皆さんのサポートがプロジェクトの継続的な改善を支えています

## 機関

[ModelBest](https://modelbest.cn/)     [THUHCSI](https://github.com/thuhcsi)

## ⭐ Star 履歴

[Star History Chart](https://star-history.com/#OpenBMB/VoxCPM&Date)
