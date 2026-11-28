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


def _has_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


class Accelerator:
    """
    Simplified accelerator that mirrors the behaviour of the minicpm-audio
    training utilities. It initializes a distributed process group when
    ``torchrun`` is used and exposes helpers for AMP, gradient scaling and
    preparing models/dataloaders for DDP.
    """

    def __init__(self, amp: bool = False, seed: int = 42):
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))

        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group("nccl", init_method="env://")

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
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

        if amp and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler("cuda")
        elif amp and _has_xpu():
            self.scaler = torch.amp.GradScaler("xpu")
        else:
            self.scaler = DummyScaler()

        if torch.cuda.is_available():
            self.device_ctx = torch.cuda.device(self.local_rank)
        elif _has_xpu():
            self.device_ctx = torch.xpu.device(self.local_rank)
        else:
            self.device_ctx = None
        self._ddp_model = None  # For no_sync support

    def _set_seed(self, seed: int):
        """Set random seed to ensure model initialization consistency across multiple GPUs"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if _has_xpu():
            torch.xpu.manual_seed_all(seed)

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
        if torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        if _has_xpu():
            return torch.device("xpu", self.local_rank)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ------------------------------------------------------------------ #
    # AMP helpers
    # ------------------------------------------------------------------ #
    def autocast(self, *args, **kwargs):
        device_type = "xpu" if (_has_xpu() and not torch.cuda.is_available()) else "cuda"
        return torch.amp.autocast(device_type, enabled=self.amp, *args, **kwargs)

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
            pin_memory=torch.cuda.is_available(),
        )

    @staticmethod
    def unwrap(model: torch.nn.Module) -> torch.nn.Module:
        return model.module if hasattr(model, "module") else model
