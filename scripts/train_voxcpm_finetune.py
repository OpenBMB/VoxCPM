#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import contextlib
import os
import signal

import argbind
import torch
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from safetensors.torch import save_file

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available, will use pytorch format", file=sys.stderr)

import json

from voxcpm.core import resolve_model_path
from voxcpm.model import VoxCPM2Model, VoxCPMModel
from voxcpm.model.voxcpm import LoRAConfig as LoRAConfigV1
from voxcpm.model.voxcpm2 import LoRAConfig as LoRAConfigV2
from voxcpm.training import (
    Accelerator,
    BatchProcessor,
    TrainingTracker,
    build_dataloader,
    load_audio_text_datasets,
)


@argbind.bind(without_prefix=True)
def train(
    pretrained_path: str,
    train_manifest: str,
    val_manifest: str = "",
    sample_rate: int = 16_000,
    out_sample_rate: int = 0,  # AudioVAE decoder output rate; used for TensorBoard audio logging
    batch_size: int = 1,
    grad_accum_steps: int = 1,
    num_workers: int = 2,
    preprocessing_num_workers: int = 1,
    num_iters: int = 100_000,
    log_interval: int = 100,
    valid_interval: int = 1_000,
    save_interval: int = 10_000,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    warmup_steps: int = 1_000,
    max_steps: int = 100_000,
    max_batch_tokens: int = 0,
    save_path: str = "checkpoints",
    tensorboard: str = "",
    lambdas: dict[str, float] = {"loss/diff": 1.0, "loss/stop": 1.0},
    lora: dict = None,
    config_path: str = "",
    max_grad_norm: float = 0.0,  # gradient clipping; 0 = disabled (backward compat)
    # Distribution options (for LoRA checkpoints)
    hf_model_id: str = "",  # HuggingFace model ID (e.g., "openbmb/VoxCPM1.5")
    distribute: bool = False,  # If True, save hf_model_id as base_model; otherwise save pretrained_path
):
    _ = config_path

    # Validate distribution options
    if lora is not None and distribute and not hf_model_id:
        raise ValueError("hf_model_id is required when distribute=True")

    accelerator = Accelerator(amp=True)

    save_dir = Path(save_path)
    tb_dir = Path(tensorboard) if tensorboard else save_dir / "logs"

    # Only create directories on rank 0 to avoid race conditions
    if accelerator.rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
    accelerator.barrier()  # Wait for directory creation

    writer = SummaryWriter(log_dir=str(tb_dir)) if accelerator.rank == 0 else None
    tracker = TrainingTracker(writer=writer, log_file=str(save_dir / "train.log"), rank=accelerator.rank)

    # Resolve HuggingFace Hub IDs to local paths (e.g. "openbmb/VoxCPM1.5")
    pretrained_path = resolve_model_path(pretrained_path)

    # Auto-detect model architecture from config.json
    with open(os.path.join(pretrained_path, "config.json"), "r", encoding="utf-8") as _f:
        _arch = json.load(_f).get("architecture", "voxcpm").lower()
    _model_cls = VoxCPM2Model if _arch == "voxcpm2" else VoxCPMModel
    LoRAConfig = LoRAConfigV2 if _arch == "voxcpm2" else LoRAConfigV1
    if accelerator.rank == 0:
        print(f"Detected architecture: {_arch} -> {_model_cls.__name__}", file=sys.stderr)
    base_model = _model_cls.from_local(
        pretrained_path, optimize=False, training=True, lora_config=LoRAConfig(**lora) if lora else None
    )
    tokenizer = base_model.text_tokenizer

    expected_sr = base_model.audio_vae.sample_rate
    assert sample_rate == expected_sr, (
        f"sample_rate mismatch: config says {sample_rate}, but the AudioVAE encoder expects {expected_sr}. "
        f"Please set sample_rate: {expected_sr} in your training config. "
    )

    def tokenize(batch):
        text_list = batch["text"]
        text_ids = [tokenizer(text) for text in text_list]
        return {"text_ids": text_ids}

    def prepare_datasets():
        prepared_train_ds, prepared_val_ds = load_audio_text_datasets(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            sample_rate=sample_rate,
            num_proc=preprocessing_num_workers,
            prepare_durations=max_batch_tokens > 0,
        )
        prepared_train_ds = prepared_train_ds.map(
            tokenize,
            batched=True,
            remove_columns=["text"],
            num_proc=preprocessing_num_workers,
            desc="Tokenizing training text",
        )
        prepared_val_texts = None
        if prepared_val_ds is not None:
            prepared_val_texts = list(prepared_val_ds["text"])
            prepared_val_ds = prepared_val_ds.map(
                tokenize,
                batched=True,
                remove_columns=["text"],
                num_proc=preprocessing_num_workers,
                desc="Tokenizing validation text",
            )
        return prepared_train_ds, prepared_val_ds, prepared_val_texts

    # Build the Hugging Face cache once. Other ranks wait and then reuse it.
    if accelerator.rank == 0:
        train_ds, val_ds, val_texts = prepare_datasets()
    accelerator.barrier()
    if accelerator.rank != 0:
        train_ds, val_ds, val_texts = prepare_datasets()

    dataset_cnt = int(max(train_ds["dataset_id"])) + 1 if "dataset_id" in train_ds.column_names else 1
    num_train_samples = len(train_ds)

    # ------------------------------------------------------------------ #
    # Optional: filter samples by estimated token count to avoid OOM
    # Enabled when max_batch_tokens > 0:
    #   max_sample_len = max_batch_tokens // batch_size
    #   Samples exceeding this length will be dropped
    # ------------------------------------------------------------------ #
    if max_batch_tokens and max_batch_tokens > 0:
        from voxcpm.training.data import compute_sample_lengths

        audio_vae_fps = base_model.audio_vae.sample_rate / base_model.audio_vae.hop_length
        est_lengths = compute_sample_lengths(
            train_ds,
            audio_vae_fps=audio_vae_fps,
            patch_size=base_model.config.patch_size,
        )
        max_sample_len = max_batch_tokens // batch_size if batch_size > 0 else max(est_lengths)
        keep_indices = [i for i, L in enumerate(est_lengths) if L <= max_sample_len]

        if len(keep_indices) < len(train_ds) and accelerator.rank == 0:
            tracker.print(
                f"Filtering {len(train_ds) - len(keep_indices)} / {len(train_ds)} "
                f"training samples longer than {max_sample_len} tokens "
                f"(max_batch_tokens={max_batch_tokens})."
            )
        train_ds = train_ds.select(keep_indices)

    train_loader = build_dataloader(
        train_ds,
        accelerator=accelerator,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = (
        build_dataloader(
            val_ds,
            accelerator=accelerator,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
        )
        if val_ds is not None
        else None
    )

    batch_processor = BatchProcessor(
        config=base_model.config,
        audio_vae=base_model.audio_vae,
        dataset_cnt=dataset_cnt,
        device=accelerator.device,
    )
    # Save audio_vae and output sample rate for audio generation.
    # Prefer model's actual output rate; fall back to YAML out_sample_rate or encode rate.
    audio_vae_for_gen = base_model.audio_vae
    out_sr = base_model.sample_rate  # decoder output rate (e.g. 48000 for V2)
    if out_sr == 0 and out_sample_rate > 0:
        out_sr = out_sample_rate
    del base_model.audio_vae
    model = accelerator.prepare_model(base_model)
    unwrapped_model = accelerator.unwrap(model)
    unwrapped_model.train()

    # Only print param info on rank 0 to avoid cluttered output
    if accelerator.rank == 0:
        for name, param in model.named_parameters():
            print(name, param.requires_grad, file=sys.stderr)

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Cosine + warmup scheduler from transformers:
    # - num_warmup_steps: warmup steps
    # - num_training_steps: total training steps (outer step count)
    total_training_steps = max_steps if max_steps > 0 else num_iters
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # All ranks load the same checkpoint to keep model and optimizer state in sync.
    start_step = load_checkpoint(model, optimizer, scheduler, save_dir, rank=accelerator.rank)
    accelerator.barrier()

    if start_step > 0 and accelerator.rank == 0:
        tracker.print(f"Resuming training from step {start_step}")

    # Resume tracker for signal handler to read current step
    resume = {"step": start_step}

    # Register signal handler to save checkpoint on termination (SIGTERM/SIGINT)
    def _signal_handler(
        signum,
        frame,
        _model=model,
        _optim=optimizer,
        _sched=scheduler,
        _save_dir=save_dir,
        _pretrained=pretrained_path,
        _hf_id=hf_model_id,
        _dist=distribute,
        _resume=resume,
        _rank=accelerator.rank,
    ):
        try:
            cur_step = int(_resume.get("step", start_step))
        except Exception:
            cur_step = start_step
        if _rank == 0:
            print(f"Signal {signum} received. Saving checkpoint at step {cur_step} ...", file=sys.stderr)
            try:
                save_checkpoint(_model, _optim, _sched, _save_dir, cur_step, _pretrained, _hf_id, _dist)
                print("Checkpoint saved. Exiting.", file=sys.stderr)
            except Exception as e:
                print(f"Error saving checkpoint on signal: {e}", file=sys.stderr)
        os._exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Manual epoch management instead of itertools.cycle to support DistributedSampler.set_epoch()
    grad_accum_steps = max(int(grad_accum_steps), 1)
    data_epoch = 0
    train_iter = iter(train_loader)

    def get_next_batch():
        """Get next batch, handles epoch boundary and DistributedSampler."""
        nonlocal train_iter, data_epoch
        try:
            return next(train_iter)
        except StopIteration:
            data_epoch += 1
            # Key: set DistributedSampler epoch to ensure different data order each epoch
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(data_epoch)
            train_iter = iter(train_loader)
            return next(train_iter)

    with tracker.live():
        for step in range(start_step, num_iters):
            # update resume step so signal handler can save current progress
            resume["step"] = step
            tracker.step = step
            optimizer.zero_grad(set_to_none=True)

            # Gradient accumulation: accumulate gradients over micro-batches before optimizer step
            loss_dict = {}
            for micro_step in range(grad_accum_steps):
                batch = get_next_batch()
                processed = batch_processor(batch)

                # Only sync gradients on the last micro-batch
                # Use no_sync() for intermediate steps to reduce communication overhead
                is_last_micro_step = micro_step == grad_accum_steps - 1
                sync_context = contextlib.nullcontext() if is_last_micro_step else accelerator.no_sync()

                with sync_context:
                    with accelerator.autocast(dtype=torch.bfloat16):
                        outputs = model(
                            processed["text_tokens"],
                            processed["text_mask"],
                            processed["audio_feats"],
                            processed["audio_mask"],
                            processed["loss_mask"],
                            processed["position_ids"],
                            processed["labels"],
                            progress=step / max(1, num_iters),
                        )

                    total_loss = 0.0
                    for key, value in outputs.items():
                        if key.startswith("loss/"):
                            weight = lambdas.get(key, 1.0)
                            loss_value = value * weight / grad_accum_steps
                            total_loss = total_loss + loss_value
                            # Record raw loss from last micro-batch for logging
                            loss_dict[key] = value.detach()

                    # Accumulate gradients (normalized by grad_accum_steps)
                    accelerator.backward(total_loss)

            # After all micro-batches, do unscale / grad_norm / step
            scaler = getattr(accelerator, "scaler", None)
            if scaler is not None:
                scaler.unscale_(optimizer)
            effective_max_norm = max_grad_norm if max_grad_norm > 0 else 1e9
            grad_norm = torch.nn.utils.clip_grad_norm_(unwrapped_model.parameters(), max_norm=effective_max_norm)

            accelerator.step(optimizer)
            accelerator.update()
            scheduler.step()

            if step % log_interval == 0 or step == num_iters - 1:
                loss_values = {k: v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in loss_dict.items()}
                loss_values["lr"] = float(optimizer.param_groups[0]["lr"])
                # Account for all GPUs when converting steps to epochs.
                epoch = (step * grad_accum_steps * batch_size * accelerator.world_size) / max(1, num_train_samples)
                loss_values["epoch"] = float(epoch)
                loss_values["grad_norm"] = float(grad_norm)
                tracker.log_metrics(loss_values, split="train")

            if val_loader is not None and (step % valid_interval == 0 or step == num_iters - 1):
                validate(
                    model,
                    val_loader,
                    batch_processor,
                    accelerator,
                    tracker,
                    lambdas,
                    writer=writer,
                    step=step,
                    val_ds=val_ds,
                    audio_vae=audio_vae_for_gen,
                    sample_rate=sample_rate,
                    out_sample_rate=out_sr,
                    val_texts=val_texts,
                    tokenizer=tokenizer,
                    valid_interval=valid_interval,
                )

            if (step % save_interval == 0 or step == num_iters - 1) and accelerator.rank == 0:
                save_checkpoint(model, optimizer, scheduler, save_dir, step, pretrained_path, hf_model_id, distribute)

    if accelerator.rank == 0:
        save_checkpoint(model, optimizer, scheduler, save_dir, num_iters, pretrained_path, hf_model_id, distribute)
    if writer:
        writer.close()


def validate(
    model,
    val_loader,
    batch_processor,
    accelerator,
    tracker,
    lambdas,
    writer=None,
    step=0,
    val_ds=None,
    audio_vae=None,
    sample_rate=22050,
    out_sample_rate=0,
    val_texts=None,
    tokenizer=None,
    valid_interval=1000,
):
    """Validate and generate sample audio"""
    from collections import defaultdict

    import numpy as np  # noqa: F401

    model.eval()
    total_losses = []
    sub_losses = defaultdict(list)  # Track individual sub-losses
    num_batches = 0
    max_val_batches = 10

    with torch.no_grad():
        for batch in val_loader:
            if num_batches >= max_val_batches:
                break
            processed = batch_processor(batch)
            with accelerator.autocast(dtype=torch.bfloat16):
                outputs = model(
                    processed["text_tokens"],
                    processed["text_mask"],
                    processed["audio_feats"],
                    processed["audio_mask"],
                    processed["loss_mask"],
                    processed["position_ids"],
                    processed["labels"],
                    progress=0.0,
                    sample_generate=False,
                )
            total = 0.0
            for key, value in outputs.items():
                if key.startswith("loss/"):
                    weighted_loss = lambdas.get(key, 1.0) * value
                    total += weighted_loss
                    sub_losses[key].append(value.detach())
            total_losses.append(total.detach())
            num_batches += 1

    if total_losses:
        # Compute mean total loss
        mean_total_loss = torch.stack(total_losses).mean()
        accelerator.all_reduce(mean_total_loss)

        # Compute mean of each sub-loss
        val_metrics = {"loss/total": mean_total_loss.item()}
        for key, values in sub_losses.items():
            mean_sub_loss = torch.stack(values).mean()
            accelerator.all_reduce(mean_sub_loss)
            val_metrics[key] = mean_sub_loss.item()

        tracker.log_metrics(val_metrics, split="val")

    # Generate sample audio for TensorBoard display
    if writer is not None and val_ds is not None and audio_vae is not None and accelerator.rank == 0:
        try:
            generate_sample_audio(
                model,
                val_ds,
                audio_vae,
                writer,
                step,
                accelerator,
                sample_rate,
                out_sample_rate=out_sample_rate,
                val_texts=val_texts,
                tokenizer=tokenizer,
                valid_interval=valid_interval,
                tracker=tracker,
            )
        except Exception as e:
            tracker.print(f"[Warning] Failed to generate sample audio: {e}")
            import io
            import traceback

            buf = io.StringIO()
            traceback.print_exc(file=buf)
            tracker.print(buf.getvalue())
    else:
        # Log why audio generation was skipped
        missing = []
        if writer is None:
            missing.append("writer")
        if val_ds is None:
            missing.append("val_ds")
        if audio_vae is None:
            missing.append("audio_vae")
        if missing and accelerator.rank == 0:
            tracker.print(f"[Warning] Skip audio generation: missing {', '.join(missing)}")

    model.train()


def compute_mel_spectrogram(audio_np, sample_rate, n_mels=128):
    """Compute Mel Spectrogram (dB) using librosa"""
    import librosa
    import numpy as np

    audio_np = audio_np.flatten().astype(np.float32)
    mel = librosa.feature.melspectrogram(y=audio_np, sr=sample_rate, n_mels=n_mels, fmax=sample_rate // 2)
    return librosa.power_to_db(mel, ref=np.max)


def create_mel_figure(gen_audio_np, gen_mel, sample_rate, step=None, ref_audio_np=None, ref_mel=None):
    """
    Create mel spectrogram figure: show comparison if reference audio exists, otherwise show generated only
    """
    import matplotlib

    matplotlib.use("Agg")
    import librosa.display
    import matplotlib.pyplot as plt

    fmax = sample_rate // 2
    step_str = f" @ Step {step}" if step is not None else ""

    if ref_audio_np is not None and ref_mel is not None:
        # Comparison mode: reference vs generated
        fig, (ax_ref, ax_gen) = plt.subplots(2, 1, figsize=(12, 8))

        img_ref = librosa.display.specshow(
            ref_mel, sr=sample_rate, x_axis="time", y_axis="mel", fmax=fmax, cmap="viridis", ax=ax_ref
        )
        ax_ref.set_title(
            f"Reference (GT) - {len(ref_audio_np)/sample_rate:.2f}s{step_str}",
            fontsize=10,
            fontweight="bold",
            color="#28A745",
        )
        plt.colorbar(img_ref, ax=ax_ref, format="%+2.0f dB", pad=0.02)

        img_gen = librosa.display.specshow(
            gen_mel, sr=sample_rate, x_axis="time", y_axis="mel", fmax=fmax, cmap="viridis", ax=ax_gen
        )
        ax_gen.set_title(
            f"Generated - {len(gen_audio_np)/sample_rate:.2f}s", fontsize=10, fontweight="bold", color="#DC3545"
        )
        plt.colorbar(img_gen, ax=ax_gen, format="%+2.0f dB", pad=0.02)
    else:
        # Single figure mode: show generated only
        fig, ax = plt.subplots(figsize=(12, 4))
        img = librosa.display.specshow(
            gen_mel, sr=sample_rate, x_axis="time", y_axis="mel", fmax=fmax, cmap="viridis", ax=ax
        )
        ax.set_title(f"Generated - {len(gen_audio_np)/sample_rate:.2f}s{step_str}", fontsize=11, fontweight="bold")
        plt.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)

    plt.tight_layout()
    return fig


def normalize_audio(audio_np):
    """Normalize audio to [-0.9, 0.9]"""
    import numpy as np

    max_val = np.abs(audio_np).max()
    return audio_np / max_val * 0.9 if max_val > 0 else audio_np


def generate_sample_audio(
    model,
    val_ds,
    audio_vae,
    writer,
    step,
    accelerator,
    sample_rate=22050,
    out_sample_rate=0,
    val_texts=None,
    tokenizer=None,
    pretrained_path=None,
    valid_interval=1000,
    tracker=None,
):
    """Select 2 fixed validation samples, generate audio and log to TensorBoard"""
    import numpy as np

    log = tracker.print if tracker else print
    num_samples = min(2, len(val_ds))
    log(f"[Audio] Starting audio generation for {num_samples} samples at step {step}")

    unwrapped_model = accelerator.unwrap(model)
    # Determine the correct output sample rate for generated audio.
    # out_sample_rate is the decoder output rate (e.g. 48kHz for V2);
    # sample_rate is the encoder input rate (e.g. 16kHz for V2).
    gen_sr = out_sample_rate if out_sample_rate > 0 else sample_rate

    for i in range(num_samples):
        sample = val_ds[i]
        text = val_texts[i] if val_texts and i < len(val_texts) else "Hello, this is a test."

        # Load reference audio
        ref_audio_np = None
        try:
            if "audio" in sample and isinstance(sample["audio"], dict) and "array" in sample["audio"]:
                ref_audio_np = np.array(sample["audio"]["array"], dtype=np.float32)
                ref_sr = sample["audio"].get("sampling_rate", sample_rate)
                if ref_sr != sample_rate:
                    import torchaudio.functional as F

                    ref_audio_np = (
                        F.resample(torch.from_numpy(ref_audio_np).unsqueeze(0), ref_sr, sample_rate).squeeze(0).numpy()
                    )
                log(f"[Audio] Loaded reference audio for sample {i}: duration={len(ref_audio_np)/sample_rate:.2f}s")
        except Exception as e:
            log(f"[Warning] Failed to load reference audio: {e}")

        # Preserve the original mode so validation failures do not leak into training.
        prev_training = unwrapped_model.training
        try:
            # Inference setup
            unwrapped_model.eval()
            # unwrapped_model.to(torch.bfloat16)
            unwrapped_model.audio_vae = audio_vae.to(torch.float32)

            log(f"[Audio] Generating sample {i} with text: '{text[:50]}...'")
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available()
                else contextlib.nullcontext()
            )
            with torch.no_grad(), autocast_ctx:
                generated = unwrapped_model.generate(
                    target_text=text, inference_timesteps=10, cfg_value=2.0, seed=42
                )

            # Restore training setup
            # unwrapped_model.to(torch.float32)
            # unwrapped_model.audio_vae = None

            if generated is None or len(generated) == 0:
                log(f"[Warning] Generated audio is empty for sample {i}")
                continue

            # Process generated audio
            gen_audio_np = (
                generated.cpu().float().numpy().flatten()
                if isinstance(generated, torch.Tensor)
                else np.array(generated, dtype=np.float32).flatten()
            )
            gen_audio_np = normalize_audio(gen_audio_np)

            tag = f"val_sample_{i}"
            writer.add_audio(f"{tag}/generated_audio", gen_audio_np, global_step=step, sample_rate=gen_sr)
            log(f"[Audio] Generated audio for sample {i}: duration={len(gen_audio_np)/gen_sr:.2f}s")

            # Log reference audio (at encoder input rate, which is what val_ds provides)
            if ref_audio_np is not None:
                writer.add_audio(
                    f"{tag}/reference_audio", normalize_audio(ref_audio_np), global_step=step, sample_rate=sample_rate
                )

            # Generate mel spectrogram figure
            try:
                mel_gen = compute_mel_spectrogram(gen_audio_np, gen_sr)
                mel_ref = compute_mel_spectrogram(ref_audio_np, sample_rate) if ref_audio_np is not None else None
                fig = create_mel_figure(gen_audio_np, mel_gen, gen_sr, step, ref_audio_np, mel_ref)
                writer.add_figure(f"{tag}/mel_spectrogram", fig, global_step=step)
                log(f"[Audio] Created mel spectrogram figure for sample {i}")
            except Exception as e:
                log(f"[Warning] Failed to create mel spectrogram: {e}")

        except Exception as e:
            log(f"[Warning] Failed to generate audio for sample {i}: {e}")
            import traceback

            traceback.print_exc()

        finally:
            # Always restore the training state, even if generation fails.
            try:
                # unwrapped_model.to(torch.float32)
                unwrapped_model.audio_vae = None
                if prev_training:
                    unwrapped_model.train()
                else:
                    unwrapped_model.eval()
            except Exception as e:
                log(f"[Warning] Failed to restore model state: {e}")


def load_checkpoint(model, optimizer, scheduler, save_dir: Path, rank: int = 0):
    """
    Load the latest checkpoint if it exists.
    Called by all ranks so that distributed state stays aligned.
    Returns the step number to resume from, or 0 if no checkpoint found.
    """
    latest_folder = save_dir / "latest"
    if not latest_folder.exists():
        return 0

    unwrapped = model.module if hasattr(model, "module") else model
    lora_cfg = unwrapped.lora_config

    # Load model weights
    if lora_cfg is not None:
        # LoRA: load lora_weights
        lora_weights_path = latest_folder / "lora_weights.safetensors"
        if not lora_weights_path.exists():
            lora_weights_path = latest_folder / "lora_weights.ckpt"

        if lora_weights_path.exists():
            if lora_weights_path.suffix == ".safetensors":
                from safetensors.torch import load_file

                state_dict = load_file(str(lora_weights_path))
            else:
                ckpt = torch.load(lora_weights_path, map_location="cpu", weights_only=True)
                state_dict = ckpt.get("state_dict", ckpt)

            unwrapped.load_state_dict(state_dict, strict=False)
            if rank == 0:
                print(f"Loaded LoRA weights from {lora_weights_path}", file=sys.stderr)
    else:
        # Full finetune: load model.safetensors or pytorch_model.bin
        model_path = latest_folder / "model.safetensors"
        if not model_path.exists():
            model_path = latest_folder / "pytorch_model.bin"

        if model_path.exists():
            if model_path.suffix == ".safetensors":
                from safetensors.torch import load_file

                state_dict = load_file(str(model_path))
            else:
                ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
                state_dict = ckpt.get("state_dict", ckpt)

            unwrapped.load_state_dict(state_dict, strict=False)
            if rank == 0:
                print(f"Loaded model weights from {model_path}", file=sys.stderr)

    # Load optimizer state
    optimizer_path = latest_folder / "optimizer.pth"
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=True))
        if rank == 0:
            print(f"Loaded optimizer state from {optimizer_path}", file=sys.stderr)

    # Load scheduler state
    scheduler_path = latest_folder / "scheduler.pth"
    if scheduler_path.exists():
        scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu", weights_only=True))
        if rank == 0:
            print(f"Loaded scheduler state from {scheduler_path}", file=sys.stderr)

    state_path = latest_folder / "training_state.json"
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        resume_step = int(state.get("step", 0))
        if rank == 0:
            print(f"Resuming from step {resume_step}", file=sys.stderr)
        return resume_step

    # Fallback for older checkpoints without metadata.
    step_folders = [d for d in save_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
    if step_folders:
        steps = [int(d.name.split("_")[1]) for d in step_folders]
        resume_step = max(steps)
        if rank == 0:
            print(f"Resuming from step {resume_step}", file=sys.stderr)
        return resume_step

    return 0


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    save_dir: Path,
    step: int,
    pretrained_path: str = None,
    hf_model_id: str = "",
    distribute: bool = False,
):
    """
    Save checkpoint with different strategies for full finetune vs LoRA:
    - Full finetune: save non-vae weights to model.safetensors (or pytorch_model.bin if safetensors unavailable)
    - LoRA: save only lora weights to lora_weights.safetensors (or lora_weights.ckpt if safetensors unavailable)
    """
    import shutil

    save_dir.mkdir(parents=True, exist_ok=True)
    tag = f"step_{step:07d}"
    folder = save_dir / tag
    folder.mkdir(parents=True, exist_ok=True)

    unwrapped = model.module if hasattr(model, "module") else model
    full_state = unwrapped.state_dict()
    lora_cfg = unwrapped.lora_config

    if lora_cfg is not None:
        # LoRA finetune: save only lora_A/lora_B weights
        state_dict = {k: v for k, v in full_state.items() if "lora_" in k}
        if SAFETENSORS_AVAILABLE:
            save_file(state_dict, folder / "lora_weights.safetensors")
        else:
            torch.save({"state_dict": state_dict}, folder / "lora_weights.ckpt")

        # Save LoRA config and base model path to a separate JSON file
        # If distribute=True, save hf_model_id; otherwise save local pretrained_path
        base_model_to_save = hf_model_id if distribute else (str(pretrained_path) if pretrained_path else None)
        lora_info = {
            "base_model": base_model_to_save,
            "lora_config": lora_cfg.model_dump() if hasattr(lora_cfg, "model_dump") else vars(lora_cfg),
        }
        with open(folder / "lora_config.json", "w", encoding="utf-8") as f:
            json.dump(lora_info, f, indent=2, ensure_ascii=False)
    else:
        # Full finetune: save non-vae weights to model.safetensors
        state_dict = {k: v for k, v in full_state.items() if not k.startswith("audio_vae.")}
        if SAFETENSORS_AVAILABLE:
            save_file(state_dict, folder / "model.safetensors")
        else:
            torch.save({"state_dict": state_dict}, folder / "pytorch_model.bin")

        # Copy config files from pretrained path
        if pretrained_path:
            pretrained_dir = Path(pretrained_path)
            files_to_copy = [
                "config.json",
                "audiovae.pth",
                "audiovae.safetensors",
                "tokenizer.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
            ]
            for fname in files_to_copy:
                src = pretrained_dir / fname
                if src.exists():
                    shutil.copy2(src, folder / fname)

    torch.save(optimizer.state_dict(), folder / "optimizer.pth")
    torch.save(scheduler.state_dict(), folder / "scheduler.pth")
    with open(folder / "training_state.json", "w", encoding="utf-8") as f:
        json.dump({"step": int(step)}, f)

    # Update (or create) a `latest` folder by copying the most recent checkpoint
    latest_link = save_dir / "latest"
    try:
        if latest_link.exists():
            shutil.rmtree(latest_link)
        shutil.copytree(folder, latest_link)
    except Exception:
        print(f"Warning: failed to update latest checkpoint at {latest_link}", file=sys.stderr)


if __name__ == "__main__":
    from voxcpm.training.config import load_yaml_config

    args = argbind.parse_args()
    config_file = args.get("config_path")
    # If YAML config provided, use YAML args to call train
    if config_file:
        yaml_args = load_yaml_config(config_file)
        train(**yaml_args)
    else:
        # Otherwise use command line args (parsed by argbind)
        with argbind.scope(args):
            train()
