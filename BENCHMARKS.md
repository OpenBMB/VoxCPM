# VoxCPM performance benchmarks

The benchmark runner measures the same public `VoxCPM.from_pretrained()` and
`generate()` path used by applications. It reports model-load time, each
generation's real-time factor (RTF), aggregate audio and utterance throughput,
and peak CUDA memory allocated during measured inference.

## Run the benchmark

Install VoxCPM and execute the runner from the repository root:

```bash
python benchmarks/benchmark_voxcpm.py --device cuda --iterations 3 --output benchmark.json
```

Two built-in English prompts are used by default. Repeat `--text` to benchmark
a custom batch, or provide a UTF-8 file containing one prompt per line:

```bash
python benchmarks/benchmark_voxcpm.py \
  --input-file prompts.txt \
  --iterations 3 \
  --warmup-runs 1 \
  --device cuda \
  --output benchmark.json
```

Each warm-up run generates every configured prompt before memory tracking and
measured iterations begin.

Use `--no-optimize` when intentionally measuring eager CUDA inference. VoxCPM
currently compiles only on the unindexed `cuda` device, so the runner records
whether optimization was requested and attempted, plus the compile-wrapper
status of every inference component. The status can be `none`, `partial`, or
`full`, so a partially failed compilation is not mislabeled as fully optimized.
Wrapper detection describes the configured model; a backend failure during lazy
compilation causes the benchmark run itself to fail. Compilation is not attempted
for CPU, MPS, or indexed devices such as `cuda:1`. Use `--denoiser` when its
model-load cost should be part of the comparison. Run `--help` for model,
revision, generation, seed, and output options.

For Hugging Face models, `--revision` accepts a branch, tag, or commit. The
runner downloads that revision once, loads the resulting local snapshot, and
records its immutable commit in `model_commit`. Snapshot resolution and download
happen before the model-load timer starts. Reuse that commit when reproducing a
report. Local model directories have no Hub commit and record `null` instead.

## Metrics

- `model_load_seconds`: wall time for `VoxCPM.from_pretrained()`, including its
  built-in warm-up when optimization is enabled.
- `rtf`: generation wall time divided by generated audio duration. Lower is
  better; values below 1 mean faster-than-real-time generation.
- `audio_seconds_per_second`: generated audio duration divided by generation
  wall time. Higher is better and is the reciprocal of aggregate RTF.
- `utterances_per_second`: measured prompts divided by generation wall time.
- `peak_cuda_memory_mb`: peak CUDA memory allocated after warm-up. It is `null`
  on CPU and MPS because PyTorch does not expose the corresponding CUDA metric.

CUDA is synchronized around timed regions so asynchronous kernels are included.
Audio files are not written during the benchmark, avoiding storage performance
in the generation measurements.

## Hardware comparison matrix

Keep the model revision, prompts, generation settings, and software versions
identical when comparing machines. The JSON report records every prompt plus
platform, processor, Python, PyTorch, CUDA, model, requested/resolved/actual
device, iteration, warm-up, denoiser, requested/attempted/per-component
optimization, model snapshot, CFG, timestep, and seed metadata.

The table below is intentionally an unpopulated results template: this project
does not publish unverified or simulated performance numbers. Add a row only
from a saved JSON report produced by the command above, and link or include the
report so other users can reproduce the measurement.

| Model | Device | PyTorch / CUDA | Optimize | Mean RTF | Audio sec/sec | Utterances/sec | Peak CUDA MiB |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| _Verified report required_ | _GPU / CPU_ | _versions_ | _yes / no_ | _value_ | _value_ | _value_ | _value / N/A_ |

For stable comparisons, run the benchmark on an otherwise idle machine at least
three times and report the median result. Include the exact command and JSON
report when publishing a matrix row.
