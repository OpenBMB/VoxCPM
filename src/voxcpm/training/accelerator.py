from __future__ import annotations

import contextlib
import os
import random
import typing

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel

from voxcpm.model.utils import current_accelerator_device_type

# Device types torch.amp knows how to autocast / grad-scale. Accelerators outside
# this set keep the pre-existing behaviour (no AMP).
_AMP_DEVICE_TYPES = frozenset({"cuda", "xpu", "npu", "hpu"})


def _distributed_backend(device_type: str) -> str:
    """Pick the collective backend that matches the active accelerator."""
    default_backend = getattr(dist, "get_default_backend_for_device", None)
    if default_backend is not None:
        try:
            return default_backend(torch.device(device_type))
        except Exception:
            pass
    return {"cuda": "nccl", "xpu": "xccl", "npu": "hccl"}.get(device_type, "gloo")


def _device_context(device_type: str, index: int):
    """Context manager binding the current process to device ``index``, if any."""
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None and hasattr(accelerator, "device_index"):
        try:
            return accelerator.device_index(index)
        except Exception:
            pass
    module = getattr(torch, device_type, None)
    device_ctx = getattr(module, "device", None)
    if callable(device_ctx):
        return device_ctx(index)
    return None


class Accelerator:
    """
    Simplified accelerator that mirrors the behaviour of the minicpm-audio
    training utilities. It initializes a distributed process group when
    ``torchrun`` is used and exposes helpers for AMP, gradient scaling and
    preparing models/dataloaders for DDP.

    The accelerator is resolved through PyTorch's device-agnostic runtime, so
    CUDA, Ascend NPU, XPU and CPU-only machines all take the matching code path.
    """

    def __init__(self, amp: bool = False, seed: int = 42):
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device_type = current_accelerator_device_type() or "cpu"

        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(_distributed_backend(self.device_type), init_method="env://")

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.amp = amp

        # Set random seed to ensure model initialization consistency
        self._set_seed(seed)

        class DummyScaler:
            def step(self, optimizer):
                optimizer.step()

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                return optimizer

            def update(self):
                pass

        amp_device_type = self._amp_device_type()
        self.scaler = torch.amp.GradScaler(amp_device_type) if (amp and amp_device_type) else DummyScaler()
        self.device_ctx = _device_context(self.device_type, self.local_rank) if self.device_type != "cpu" else None
        self._ddp_model = None  # For no_sync support

    def _amp_device_type(self) -> typing.Optional[str]:
        """Device type usable by torch.amp, or ``None`` when AMP is unavailable."""
        return self.device_type if self.device_type in _AMP_DEVICE_TYPES else None

    def _set_seed(self, seed: int):
        """Set random seed to ensure model initialization consistency across multiple GPUs"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def __enter__(self):
        if self.device_ctx is not None:
            self.device_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.device_ctx is not None:
            self.device_ctx.__exit__(exc_type, exc_value, traceback)

    def barrier(self):
        """Synchronize all processes"""
        if dist.is_initialized():
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.AVG):
        """All-reduce tensor across processes"""
        if dist.is_initialized():
            dist.all_reduce(tensor, op=op)
        return tensor

    # ------------------------------------------------------------------ #
    # Model helpers
    # ------------------------------------------------------------------ #
    def prepare_model(self, model: torch.nn.Module, **kwargs):
        if hasattr(model, "device"):  # make sure the matrix will be moved to the correct device
            model.device = self.device
        model = model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(model, device_ids=[self.local_rank], **kwargs)
            self._ddp_model = model  # Save DDP model reference for no_sync support
        return model

    @contextlib.contextmanager
    def no_sync(self):
        """
        Context manager to skip gradient synchronization during gradient accumulation.
        Only used outside the last micro-batch.
        """
        if self._ddp_model is not None:
            with self._ddp_model.no_sync():
                yield
        else:
            yield

    @property
    def device(self):
        if self.device_type == "cpu":
            return torch.device("cpu")
        if self.device_type == "mps":
            return torch.device("mps")
        return torch.device(self.device_type, self.local_rank)

    # ------------------------------------------------------------------ #
    # AMP helpers
    # ------------------------------------------------------------------ #
    def autocast(self, *args, **kwargs):
        amp_device_type = self._amp_device_type()
        if amp_device_type is None:
            return torch.amp.autocast("cpu", enabled=False, *args, **kwargs)
        return torch.amp.autocast(amp_device_type, enabled=self.amp, *args, **kwargs)

    def backward(self, loss: torch.Tensor):
        self.scaler.scale(loss).backward()

    def step(self, optimizer: torch.optim.Optimizer):
        self.scaler.step(optimizer)

    def update(self):
        self.scaler.update()

    # ------------------------------------------------------------------ #
    # Data helpers
    # ------------------------------------------------------------------ #
    def prepare_dataloader(
        self,
        dataset: typing.Iterable,
        *,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        collate_fn=None,
        drop_last: bool = False,
    ) -> torch.utils.data.DataLoader:
        if self.world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle
            )
            shuffle = False
        else:
            sampler = None

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
            pin_memory=True,
        )

    @staticmethod
    def unwrap(model: torch.nn.Module) -> torch.nn.Module:
        return model.module if hasattr(model, "module") else model
