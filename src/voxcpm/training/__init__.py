"""
Training utilities for VoxCPM fine-tuning.

This package mirrors the training mechanics used in the minicpm-audio
tooling while relying solely on local audio-text datasets managed via
the HuggingFace ``datasets`` library.
"""

from .accelerator import Accelerator
from .tracker import TrainingTracker
from .data import (
    load_audio_text_datasets,
    HFVoxCPMDataset,
    build_dataloader,
    BatchProcessor,
)
from .state import TrainingState
from .validate import validate_manifest, ValidationResult

__all__ = [
    "Accelerator",
    "TrainingTracker",
    "HFVoxCPMDataset",
    "BatchProcessor",
    "TrainingState",
    "load_audio_text_datasets",
    "build_dataloader",
    "validate_manifest",
    "ValidationResult",
]
