import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argbind
import torch
from datasets import Audio, Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from torch.utils.data import Dataset as TorchDataset

from ..model.voxcpm import VoxCPMConfig
from ..modules.audiovae import AudioVAE
from .packers import AudioFeatureProcessingPacker

DEFAULT_TEXT_COLUMN = "text"
DEFAULT_AUDIO_COLUMN = "audio"
DEFAULT_REF_AUDIO_COLUMN = "ref_audio"
DEFAULT_ID_COLUMN = "dataset_id"


def _manifest_features(
    *,
    manifest_paths: List[str],
    text_column: str,
    audio_column: str,
    ref_audio_column: str,
    dataset_id_column: str,
) -> Features:
    features = {
        text_column: Value("string"),
        audio_column: Value("string"),
        ref_audio_column: Value("string"),
        "duration": Value("float64"),
        "ref_duration": Value("float64"),
        "is_prompt": Value("bool"),
    }
    if dataset_id_column:
        features[dataset_id_column] = Value("int64")

    for manifest_path in manifest_paths:
        with Path(manifest_path).open("r", encoding="utf-8") as manifest:
            sample_lines = []
            for line in manifest:
                stripped = line.strip()
                if stripped:
                    sample_lines.append(stripped)
                    if len(sample_lines) >= 100:
                        break
        for line in sample_lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for column, value in record.items():
                if column not in features and value is not None:
                    features[column] = _feature_from_value(value)
    return Features(features)


def _feature_from_value(value):
    if isinstance(value, bool):
        return Value("bool")
    if isinstance(value, int):
        return Value("int64")
    if isinstance(value, float):
        return Value("float64")
    if isinstance(value, str):
        return Value("string")
    if isinstance(value, list):
        first_value = next((item for item in value if item is not None), "")
        return Sequence(_feature_from_value(first_value))
    if isinstance(value, dict):
        return {key: _feature_from_value(item) for key, item in value.items() if item is not None}
    raise ValueError(f"Unsupported manifest field type: {type(value).__name__}")


def _resolve_audio_path(path: str, manifest_dir: str) -> str:
    if not path:
        return path
    audio_path = Path(path).expanduser()
    if not audio_path.is_absolute():
        audio_path = Path(manifest_dir) / audio_path
    return str(audio_path.resolve())


def _read_audio_duration(path: str, column: str) -> float:
    if not path:
        raise ValueError(f"Cannot determine duration: '{column}' is empty.")

    import soundfile as sf

    try:
        return float(sf.info(path).duration)
    except Exception as exc:
        raise ValueError(f"Cannot read audio metadata for '{column}' file '{path}': {exc}") from exc


def _prepare_audio_metadata(
    example: Dict,
    *,
    audio_column: str,
    ref_audio_column: str,
    manifest_dir: str,
    prepare_durations: bool,
) -> Dict:
    audio_path = _resolve_audio_path(example.get(audio_column), manifest_dir)
    ref_audio_path = _resolve_audio_path(example.get(ref_audio_column), manifest_dir)

    duration = example.get("duration")
    if prepare_durations and duration is None:
        duration = _read_audio_duration(audio_path, audio_column)

    ref_duration = example.get("ref_duration")
    if prepare_durations and ref_audio_path and ref_duration is None:
        ref_duration = _read_audio_duration(ref_audio_path, ref_audio_column)

    return {
        audio_column: audio_path,
        ref_audio_column: ref_audio_path,
        "duration": duration,
        "ref_duration": ref_duration,
    }


@argbind.bind()
def load_audio_text_datasets(
    train_manifest: str,
    val_manifest: str = "",
    text_column: str = DEFAULT_TEXT_COLUMN,
    audio_column: str = DEFAULT_AUDIO_COLUMN,
    ref_audio_column: str = DEFAULT_REF_AUDIO_COLUMN,
    dataset_id_column: str = DEFAULT_ID_COLUMN,
    sample_rate: int = 16_000,
    num_proc: int = 1,
    prepare_durations: bool = False,
) -> Tuple[Dataset, Optional[Dataset]]:
    if num_proc < 1:
        raise ValueError(f"num_proc must be at least 1, got {num_proc}.")

    data_files = {"train": train_manifest}
    if val_manifest:
        data_files["validation"] = val_manifest

    dataset_dict: DatasetDict = load_dataset(
        "json",
        data_files=data_files,
        features=_manifest_features(
            manifest_paths=list(data_files.values()),
            text_column=text_column,
            audio_column=audio_column,
            ref_audio_column=ref_audio_column,
            dataset_id_column=dataset_id_column,
        ),
    )

    def prepare(ds: Dataset, manifest_path: str) -> Dataset:
        if audio_column not in ds.column_names:
            raise ValueError(f"Expected '{audio_column}' column in manifest.")
        ds = ds.map(
            _prepare_audio_metadata,
            fn_kwargs={
                "audio_column": audio_column,
                "ref_audio_column": ref_audio_column,
                "manifest_dir": str(Path(manifest_path).expanduser().resolve().parent),
                "prepare_durations": prepare_durations,
            },
            num_proc=num_proc,
            desc="Preparing audio metadata",
        )
        ds = ds.cast_column(audio_column, Audio(sampling_rate=sample_rate))
        if audio_column != DEFAULT_AUDIO_COLUMN:
            ds = ds.rename_column(audio_column, DEFAULT_AUDIO_COLUMN)
        if text_column != DEFAULT_TEXT_COLUMN:
            ds = ds.rename_column(text_column, DEFAULT_TEXT_COLUMN)

        # Explicit JSON features make ref_audio nullable even when omitted.
        ref_col = ref_audio_column if ref_audio_column in ds.column_names else DEFAULT_REF_AUDIO_COLUMN
        has_ref_audio = ref_col in ds.column_names and any(ds[ref_col])
        if has_ref_audio:
            ds = ds.cast_column(ref_col, Audio(sampling_rate=sample_rate))
            if ref_col != DEFAULT_REF_AUDIO_COLUMN:
                ds = ds.rename_column(ref_col, DEFAULT_REF_AUDIO_COLUMN)
        elif ref_col in ds.column_names:
            ds = ds.remove_columns(ref_col)

        if dataset_id_column and dataset_id_column in ds.column_names:
            if all(value is None for value in ds[dataset_id_column]):
                ds = ds.remove_columns(dataset_id_column)
                ds = ds.add_column(DEFAULT_ID_COLUMN, [0] * len(ds))
            elif dataset_id_column != DEFAULT_ID_COLUMN:
                ds = ds.rename_column(dataset_id_column, DEFAULT_ID_COLUMN)
        else:
            ds = ds.add_column(DEFAULT_ID_COLUMN, [0] * len(ds))
        return ds

    train_ds = prepare(dataset_dict["train"], train_manifest)
    val_ds = prepare(dataset_dict["validation"], val_manifest) if "validation" in dataset_dict else None
    return train_ds, val_ds


def compute_sample_lengths(
    ds: Dataset,
    audio_vae_fps: int = 25,
    patch_size: int = 1,
) -> List[int]:
    """
    预估每个样本经过 packer 之后的大致序列长度（text+audio），用于过滤超长样本。

    逻辑与 AudioFeatureProcessingPacker / AudioVAE 一致：
    - 文本长度: len(text_ids)
    - 音频长度:
        duration(s) * audio_vae_fps -> 近似 VAE 帧数 t_vae
        t_seq = ceil(t_vae / patch_size)
    - 无 ref_audio: text_len + t_seq + 2
    - 有 ref_audio: text_len + t_seq + ref_seq + 4

    Optimized: Use batch column access instead of iterating item by item.
    """
    text_ids_list = ds["text_ids"]
    text_lens = [len(t) for t in text_ids_list]

    if "duration" not in ds.column_names:
        raise ValueError("The dataset must contain a 'duration' column before length filtering.")
    durations = ds["duration"]
    missing_duration_count = sum(duration is None for duration in durations)
    if missing_duration_count:
        raise ValueError(
            f"The dataset contains {missing_duration_count} samples without duration metadata. "
            "Load it with prepare_durations=True before length filtering."
        )

    has_ref_audio = DEFAULT_REF_AUDIO_COLUMN in ds.column_names
    ref_durations = ds["ref_duration"] if "ref_duration" in ds.column_names else [None] * len(ds)

    lengths = []
    for text_len, duration, ref_dur in zip(text_lens, durations, ref_durations):
        t_vae = math.ceil(float(duration) * audio_vae_fps)
        t_seq = math.ceil(t_vae / patch_size)

        ref_seq = 0
        if has_ref_audio:
            if ref_dur is not None and float(ref_dur) > 0:
                ref_vae = math.ceil(float(ref_dur) * audio_vae_fps)
                ref_seq = math.ceil(ref_vae / patch_size)

        # +2 for 101/102; +2 more for 103/104 when ref_audio present
        overhead = 4 if ref_seq > 0 else 2
        total_len = text_len + t_seq + ref_seq + overhead
        lengths.append(total_len)

    return lengths


class HFVoxCPMDataset(TorchDataset):
    """
    Thin wrapper around a tokenized HuggingFace dataset that returns
    PyTorch-friendly samples.
    """

    _SENTINEL = [-100.0]

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.has_ref_audio = DEFAULT_REF_AUDIO_COLUMN in dataset.column_names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        audio = item[DEFAULT_AUDIO_COLUMN]
        sample = {
            "text_ids": item["text_ids"],
            "audio_array": audio["array"],
            "audio_sampling_rate": audio["sampling_rate"],
            "dataset_id": item.get(DEFAULT_ID_COLUMN, 0),
            "is_prompt": item.get("is_prompt", False),
        }
        if self.has_ref_audio:
            ref = item.get(DEFAULT_REF_AUDIO_COLUMN)
            sample["ref_audio_array"] = ref["array"] if ref else self._SENTINEL
        return sample

    @staticmethod
    def pad_sequences(seqs: List[torch.Tensor], pad_value: float):
        if not seqs:
            return torch.empty(0)
        max_len = max(seq.shape[0] for seq in seqs)
        padded = []
        for seq in seqs:
            if seq.shape[0] < max_len:
                pad_width = (0, max_len - seq.shape[0])
                seq = torch.nn.functional.pad(seq, pad_width, value=pad_value)
            padded.append(seq)
        return torch.stack(padded)

    @classmethod
    def collate_fn(cls, batch: List[Dict]):
        text_tensors = [torch.tensor(sample["text_ids"], dtype=torch.int32) for sample in batch]
        audio_tensors = [torch.tensor(sample["audio_array"], dtype=torch.float32) for sample in batch]
        dataset_ids = torch.tensor([sample["dataset_id"] for sample in batch], dtype=torch.int32)
        is_prompts = [bool(sample.get("is_prompt", False)) for sample in batch]

        text_padded = cls.pad_sequences(text_tensors, pad_value=-100)
        audio_padded = cls.pad_sequences(audio_tensors, pad_value=-100.0)
        task_ids = torch.ones(text_padded.size(0), dtype=torch.int32)

        result = {
            "text_tokens": text_padded,
            "audio_tokens": audio_padded,
            "task_ids": task_ids,
            "dataset_ids": dataset_ids,
            "is_prompts": is_prompts,
        }

        if "ref_audio_array" in batch[0]:
            ref_tensors = [torch.tensor(s["ref_audio_array"], dtype=torch.float32) for s in batch]
            result["ref_audio_tokens"] = cls.pad_sequences(ref_tensors, pad_value=-100.0)

        return result


class BatchProcessor:
    """
    Wraps ``AudioFeatureProcessingPacker`` so the training loop can mirror
    the minicpm-audio mechanics.
    """

    def __init__(
        self,
        *,
        config: VoxCPMConfig,
        audio_vae: AudioVAE,
        dataset_cnt: int,
        device: torch.device,
    ):
        self.device = device
        self.dataset_cnt = dataset_cnt
        self.audio_vae = audio_vae
        self.audio_vae.to(device)
        self.packer = AudioFeatureProcessingPacker(
            dataset_cnt=dataset_cnt,
            max_len=config.max_length,
            patch_size=config.patch_size,
            feat_dim=config.feat_dim,
            audio_vae=self.audio_vae,
        )

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        audio_tokens = batch["audio_tokens"].to(self.device)
        text_tokens = batch["text_tokens"].to(self.device)
        task_ids = batch["task_ids"].to(self.device)
        dataset_ids = batch["dataset_ids"].to(self.device)

        ref_audio_tokens = None
        if "ref_audio_tokens" in batch:
            ref_audio_tokens = batch["ref_audio_tokens"].to(self.device)

        packed = self.packer(
            audio_tokens=audio_tokens,
            text_tokens=text_tokens,
            task_ids=task_ids,
            dataset_ids=dataset_ids,
            is_prompts=batch["is_prompts"],
            ref_audio_tokens=ref_audio_tokens,
        )
        return packed


def build_dataloader(
    hf_dataset: Dataset,
    *,
    accelerator,
    batch_size: int,
    num_workers: int,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    torch_dataset = HFVoxCPMDataset(hf_dataset)
    # Standard padding-based batching; Accelerator will attach DistributedSampler if needed.
    return accelerator.prepare_dataloader(
        torch_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=HFVoxCPMDataset.collate_fn,
        drop_last=drop_last,
    )
