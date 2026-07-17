# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

### Install (editable dev mode)
```bash
pip install -e ".[dev]"
```

### Run tests
```bash
pytest tests/
pytest tests/test_cli.py::TestFunctionName  # single test
pytest --cov=voxcpm tests/
```

### Lint / format
```bash
flake8 src/
black src/
```

### Run Gradio demo
```bash
python app.py
```

### CLI
```bash
voxcpm design --text "Hello world" --output out.wav
voxcpm clone --reference-wav ref.wav --text "Hello" --output out.wav
```

---

## Architecture

The library lives in `src/voxcpm/` and is installed as the `voxcpm` package.

### Entry points

- **`VoxCPM`** (`core.py`) — public facade. `VoxCPM.from_pretrained(hf_model_id)` downloads weights from HF Hub and dispatches to the correct internal model class based on the `"architecture"` field in `config.json`.
  - `"voxcpm"` → `VoxCPMModel` (VoxCPM1.5 / 0.5B)
  - `"voxcpm2"` → `VoxCPM2Model` (VoxCPM2)
- **CLI** (`cli.py`) — `voxcpm` entrypoint. Subcommands: `design`, `clone`, `legacy`. Imports are lazy so the CLI loads fast without pulling in torch.

### Model pipeline (4-stage)

```
Text → LocEnc → TSLM (MiniCPM4 LM) → RALM → LocDiT (diffusion) → AudioVAE decode → waveform
```

- `modules/locenc/` — Local Encoder: converts text/reference audio into continuous embeddings.
- `modules/minicpm4/` — MiniCPM4 language model backbone (TSLM + RALM stages).
- `modules/locdit/` — Local DiT: flow-matching diffusion decoder (`UnifiedCFM`). V1 uses `local_dit.py`, V2 uses `local_dit_v2.py`.
- `modules/audiovae/` — AudioVAE encoder/decoder. V1 uses 16kHz (`audio_vae.py`), V2 uses asymmetric encode-16kHz/decode-48kHz (`audio_vae_v2.py`).
- `modules/layers/lora.py` — LoRA injection helpers applied to named linear modules.
- `zipenhancer.py` — Optional post-processing denoiser (ModelScope ZipEnhancer, 16kHz).

### Model versions in this repo

| Class | File | Weights |
|---|---|---|
| `VoxCPMModel` | `model/voxcpm.py` | `openbmb/VoxCPM1.5`, `openbmb/VoxCPM-0.5B` |
| `VoxCPM2Model` | `model/voxcpm2.py` | `openbmb/VoxCPM2` |

VoxCPM1.5 supports only **continuation cloning** (requires `prompt_wav_path` + `prompt_text`). VoxCPM2 adds Voice Design and Controllable Cloning.

### Fine-tuning

- `training/` — accelerate-based SFT trainer. Entry point: `scripts/train_voxcpm_finetune.py`.
- LoRA training and inference: `scripts/test_voxcpm_lora_infer.py`, `scripts/test_voxcpm_ft_infer.py`.
- `lora_ft_webui.py` — Gradio UI for LoRA fine-tuning.

### Key conventions

- Black line length: 120 chars, target Python 3.10.
- Tests stub out `voxcpm.core.VoxCPM` to avoid loading torch/model weights — keep CLI tests fast by maintaining that pattern.
- `optimize=True` calls `torch.compile` and runs a warmup inference on init; set `optimize=False` when debugging.
- Device selection: pass `device=None` (auto) or explicit `"cuda"`, `"mps"`, `"cpu"`.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
