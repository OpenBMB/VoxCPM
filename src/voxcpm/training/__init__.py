"""
Training utilities for VoxCPM fine-tuning.

This package mirrors the training mechanics used in the minicpm-audio
tooling while relying solely on local audio-text datasets managed via
the HuggingFace ``datasets`` library.
"""

from .accelerator import Accelerator
from .data import (
    BatchProcessor,
    HFVoxCPMDataset,
    build_dataloader,
    load_audio_text_datasets,
)
from .state import TrainingState
from .tracker import TrainingTracker
from .validate import ValidationResult, validate_manifest

__all__ = [
    "Accelerator",
    "BatchProcessor",
    "HFVoxCPMDataset",
    "TrainingState",
    "TrainingTracker",
    "ValidationResult",
    "build_dataloader",
    "load_audio_text_datasets",
    "validate_manifest",
]
