# VoxCPM on AMD GPUs (ROCm) — Podman / Docker

Runs the VoxCPM2 WebUI (`app.py`: TTS, voice design, cloning) or the LoRA
training UI (`lora_ft_webui.py`) on AMD Radeon GPUs. Tested on an
**RX 6700 XT (gfx1031, 12 GB)** with rootless Podman 5.4 on Debian 13.

| File | Purpose |
|---|---|
| `Containerfile` | python 3.11-slim + PyTorch 2.9.1 ROCm 6.4 wheels + VoxCPM |
| `launch.py` | Runs the upstream `app.py` unchanged, with ROCm/multilingual tweaks via env vars |
| `build.sh` | Builds `localhost/voxcpm:rocm`, optionally tags and pushes to a registry |
| `run.sh` | `podman run` / `docker run` with GPU passthrough and volume mounts |

## Quick start

```bash
# from the project root
docker/rocm/build.sh                        # -> localhost/voxcpm:rocm
docker/rocm/run.sh                          # WebUI -> http://localhost:8808
docker/rocm/run.sh python lora_ft_webui.py  # LoRA training UI
```

The host user needs access to `/dev/kfd` and `/dev/dri` (the `video` and
`render` groups). No ROCm install is needed on the host, only the `amdgpu`
kernel driver.

The first generation downloads `openbmb/VoxCPM2` (~5 GB) to `models/hf`,
ZipEnhancer (the reference denoiser) to `models/modelscope`, and Whisper on
first ASR use. All of these persist in the `models/` volume.

## build.sh settings

| Variable | Default | |
|---|---|---|
| `IMAGE_NAME`, `TAG` | `voxcpm`, `rocm` | local image is `localhost/IMAGE_NAME:TAG` |
| `REGISTRY` | — | e.g. `ghcr.io/<user>`: also tags `REGISTRY/IMAGE_NAME:TAG` and `:TAG-<git sha>` |
| `PUSH` | `0` | `1` pushes the registry tags (log in first: `podman login ghcr.io`) |
| `SHA_TAG` | `1` | `0` skips the extra `:TAG-<git sha>` tag |
| `LOCAL_TAG` | `1` | `0` (with `REGISTRY`) keeps only the registry name locally |
| `SOURCE_URL` | `origin` remote | OCI source label/annotation; GHCR links the package to this repo |

Extra arguments go to `podman build`, e.g. `docker/rocm/build.sh --no-cache`.

```bash
REGISTRY=ghcr.io/<user> PUSH=1 docker/rocm/build.sh
```

## run.sh settings

| Variable | Default | |
|---|---|---|
| `IMAGE` | `voxcpm:rocm` | e.g. `ghcr.io/<user>/voxcpm:rocm` to run a pulled image |
| `ENGINE` | `podman` if installed, else `docker` | |
| `PORT` | `8808` | host and container port |
| `DATA_ROOT` | project root | holds `models/ data/ lora/ output/` |
| `NETWORK`, `IP` | — | optional container network / static IP |
| `HSA_OVERRIDE_GFX_VERSION` | auto | gfx1031–1036 → `10.3.0`, gfx1101–1103 → `11.0.0`; `none` disables it |

## Runtime settings (env, passed through by run.sh)

| Variable | Default | Notes |
|---|---|---|
| `VOXCPM_MODEL` | `openbmb/VoxCPM2` | HF repo id or a local path under `/app/models` |
| `VOXCPM_DTYPE` | `float16` | RDNA2 has no fast bf16 path, so fp16 is used. Empty = checkpoint dtype (bf16) |
| `VOXCPM_OPTIMIZE` | `0` | `1` = `torch.compile`. `reduce-overhead` (HIP graphs) crashes gfx1031 with a GPU memory access fault |
| `ASR_BACKEND` | `whisper` | transcribes the reference audio in any language. `sensevoice` = upstream default (zh/en/ja/ko/yue only) |
| `ASR_MODEL` | `openai/whisper-large-v3-turbo` | |
| `ASR_LANGUAGE` | auto | force a language, e.g. `id` |
| `ASR_DEVICE` | same GPU | `cpu` if VRAM is tight |
| `TEXT_NUMBER_LANG` | off | expand digits to words with `num2words`, e.g. `id`: `2026` → "dua ribu dua puluh enam", `Rp 25.000` → "dua puluh lima ribu rupiah", `09.30` → "sembilan tiga puluh". The built-in text normalizer only handles zh/en |
| `MIOPEN_FIND_MODE` | `FAST` | Skips MIOpen's per-shape kernel benchmarking. In the default mode every new text length triggers it, so long or varied text runs 4–5× slower |
| `AUTO_SPLIT_WORDS` | `150` | Longer text is split into paragraph/sentence chunks, generated with one seed and a locked voice (chunk 1's first 12 s become the reference), and joined. Long results are also saved to `/app/output/webui_<time>.wav/.mp3`. `0` = off: a single call fails beyond ~1,400 words |
| `AUTO_SPLIT_PAUSE` | `0.7` | pause between paragraphs, seconds |
| `VOICES_DIR` | `/app/data/voices` | voice presets: every audio file there appears in a *Voice preset* dropdown above the reference audio. Picking one fills the reference, enables Ultimate Cloning and loads `<name>.txt` as transcript (created once by Whisper if missing, then editable). The list refreshes on page reload |
| `DIT_STEPS` | `10` | starting value of the *LocDiT flow-matching steps* slider. Fewer steps are faster but give rougher audio; it can still be changed per request |

## Measured on an RX 6700 XT

| Indonesian, 10 s of audio, warm | bf16 | **fp16 (default)** |
|---|---|---|
| RTF | 4.28 | **1.60** |
| Peak VRAM | 5.8 GB | 5.8 GB |

A small-batch linear layer runs 5× faster in fp16 than in bf16 on this card
(51 µs vs 251 µs), and the audio output is equivalent (no NaNs, same RMS).
The first generation after start-up is 2–3× slower while kernels warm up.
Without `MIOPEN_FIND_MODE=FAST` (the image default), *every* new text length
is also slow, because MIOpen benchmarks its convolution kernels per input shape.

## Tips for non-English / non-Chinese text

- Turn **Text Normalization** off in the UI (it uses the English normalizer for
  non-Chinese text). Use `TEXT_NUMBER_LANG` for numbers instead.
- For the most natural result, use **Ultimate Cloning**: a clean 5–15 s
  reference plus its exact transcript. Whisper fills in the transcript; check
  it by hand.
- Voice design works with descriptions in the target language, e.g.
  `(Perempuan muda, suara lembut dan hangat)Halo, selamat datang.`

## Troubleshooting

- **`HIP out of memory` with plenty of VRAM free on paper**: another process
  (an LLM server, for example) is holding VRAM. VoxCPM2 needs about 6 GB.
- **`torch.cuda.is_available()` is False**: check `ls -l /dev/kfd` and that
  your user is in the `render` group. With Podman, keep `--group-add keep-groups`.
  Don't add `--userns=keep-id`: rootless container root already maps to your
  user, and keep-id makes Podman copy the 18 GB image with remapped IDs on first run.
- **Permission errors under `/app`**: everything writable lives in the mounted
  volumes, and `HOME` is `/app/models/.home`.
