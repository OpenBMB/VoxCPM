import os
import sys
import re
import json
import tempfile
import numpy as np
from typing import Generator, Optional
from huggingface_hub import snapshot_download
from .model.voxcpm import VoxCPMModel, LoRAConfig
from .model.voxcpm2 import VoxCPM2Model
from .model.utils import next_and_close


def _split_long_text_for_tts(text: str, max_chars: int = 90) -> list[str]:
    """Split long spoken text on natural punctuation while keeping chunks short."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("target text must be a non-empty string")
    if max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text]

    pieces = re.findall(r"[^。！？!?；;.]+[。！？!?；;.]?", text)
    if not pieces:
        pieces = [text]

    segments = []
    current = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        subpieces = [piece]
        if len(piece) > max_chars:
            subpieces = re.findall(r"[^，,、：:]+[，,、：:]?", piece) or [piece]

        for subpiece in subpieces:
            subpiece = subpiece.strip()
            while len(subpiece) > max_chars:
                if current:
                    segments.append(current)
                    current = ""
                segments.append(subpiece[:max_chars])
                subpiece = subpiece[max_chars:].strip()
            if not subpiece:
                continue
            if current and len(current) + len(subpiece) > max_chars:
                segments.append(current)
                current = subpiece
            else:
                current += subpiece

    if current:
        segments.append(current)
    return segments or [text]


def _as_mono_float32(wav: np.ndarray) -> np.ndarray:
    return np.asarray(wav, dtype=np.float32).reshape(-1)


class VoxCPM:
    def __init__(
        self,
        voxcpm_model_path: str,
        zipenhancer_model_path: str | None = "iic/speech_zipenhancer_ans_multiloss_16k_base",
        enable_denoiser: bool = True,
        optimize: bool = True,
        device: str | None = None,
        lora_config: Optional[LoRAConfig] = None,
        lora_weights_path: Optional[str] = None,
    ):
        """Initialize VoxCPM TTS pipeline.

        Args:
            voxcpm_model_path: Local filesystem path to the VoxCPM model assets
                (weights, configs, etc.). Typically the directory returned by
                a prior download step.
            zipenhancer_model_path: ModelScope acoustic noise suppression model
                id or local path. If None, denoiser will not be initialized.
            enable_denoiser: Whether to initialize the denoiser pipeline.
            optimize: Whether to optimize the model with torch.compile. True by default, but can be disabled for debugging.
            device: Runtime device. If set to ``None`` or ``"auto"``, VoxCPM
                will choose automatically (preferring CUDA, then MPS, then CPU).
                If set explicitly, that device is used or a clear error is raised.
            lora_config: LoRA configuration for fine-tuning. If lora_weights_path is
                provided without lora_config, a default config will be created.
            lora_weights_path: Path to pre-trained LoRA weights (.pth file or directory
                containing lora_weights.ckpt). If provided, LoRA weights will be loaded.
        """
        print(
            f"voxcpm_model_path: {voxcpm_model_path}, zipenhancer_model_path: {zipenhancer_model_path}, enable_denoiser: {enable_denoiser}",
            file=sys.stderr,
        )

        # If lora_weights_path is provided but no lora_config, create a default one
        if lora_weights_path is not None and lora_config is None:
            lora_config = LoRAConfig(
                enable_lm=True,
                enable_dit=True,
                enable_proj=False,
            )
            print(f"Auto-created default LoRAConfig for loading weights from: {lora_weights_path}", file=sys.stderr)

        # Determine model type from config.json architecture field
        config_path = os.path.join(voxcpm_model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        arch = config.get("architecture", "voxcpm").lower()

        if arch == "voxcpm2":
            self.tts_model = VoxCPM2Model.from_local(
                voxcpm_model_path,
                optimize=optimize,
                device=device,
                lora_config=lora_config,
            )
            print("Loaded VoxCPM2Model", file=sys.stderr)
        elif arch == "voxcpm":
            self.tts_model = VoxCPMModel.from_local(
                voxcpm_model_path,
                optimize=optimize,
                device=device,
                lora_config=lora_config,
            )
            print("Loaded VoxCPMModel", file=sys.stderr)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        # Load LoRA weights if path is provided
        if lora_weights_path is not None:
            print(f"Loading LoRA weights from: {lora_weights_path}", file=sys.stderr)
            loaded_keys, skipped_keys = self.tts_model.load_lora_weights(lora_weights_path)
            print(f"Loaded {len(loaded_keys)} LoRA parameters, skipped {len(skipped_keys)}", file=sys.stderr)

        self.text_normalizer = None
        self.denoiser = None
        if enable_denoiser and zipenhancer_model_path is not None:
            from .zipenhancer import ZipEnhancer

            self.denoiser = ZipEnhancer(zipenhancer_model_path)
        else:
            self.denoiser = None
        if optimize:
            print("Warm up VoxCPMModel...", file=sys.stderr)
            self.tts_model.generate(
                target_text="Hello, this is the first test sentence.",
                max_len=10,
            )

    @classmethod
    def from_pretrained(
        cls,
        hf_model_id: str = "openbmb/VoxCPM2",
        load_denoiser: bool = True,
        zipenhancer_model_id: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
        cache_dir: str = None,
        local_files_only: bool = False,
        optimize: bool = True,
        device: str | None = None,
        lora_config: Optional[LoRAConfig] = None,
        lora_weights_path: Optional[str] = None,
        **kwargs,
    ):
        """Instantiate ``VoxCPM`` from a Hugging Face Hub snapshot.

        Args:
            hf_model_id: Explicit Hugging Face repository id (e.g. "org/repo") or local path.
            load_denoiser: Whether to initialize the denoiser pipeline.
            optimize: Whether to optimize the model with torch.compile. True by default, but can be disabled for debugging.
            zipenhancer_model_id: Denoiser model id or path for ModelScope
                acoustic noise suppression.
            cache_dir: Custom cache directory for the snapshot.
            local_files_only: If True, only use local files and do not attempt
                to download.
            device: Runtime device. Use ``None``/``"auto"`` for automatic
                fallback, or an explicit value such as ``"cpu"``, ``"mps"``,
                ``"cuda"``, or ``"cuda:0"``.
            lora_config: LoRA configuration for fine-tuning. If lora_weights_path is
                provided without lora_config, a default config will be created with
                enable_lm=True and enable_dit=True.
            lora_weights_path: Path to pre-trained LoRA weights (.pth file or directory
                containing lora_weights.ckpt). If provided, LoRA weights will be loaded
                after model initialization.
        Kwargs:
            Additional keyword arguments passed to the ``VoxCPM`` constructor.

        Returns:
            VoxCPM: Initialized instance whose ``voxcpm_model_path`` points to
            the downloaded snapshot directory.

        Raises:
            ValueError: If neither a valid ``hf_model_id`` nor a resolvable
                ``hf_model_id`` is provided.
        """
        repo_id = hf_model_id
        if not repo_id:
            raise ValueError("You must provide hf_model_id")

        # Load from local path if provided
        if os.path.isdir(repo_id):
            local_path = repo_id
        else:
            # Otherwise, try from_pretrained (Hub); exit on failure
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        return cls(
            voxcpm_model_path=local_path,
            zipenhancer_model_path=zipenhancer_model_id if load_denoiser else None,
            enable_denoiser=load_denoiser,
            optimize=optimize,
            device=device,
            lora_config=lora_config,
            lora_weights_path=lora_weights_path,
            **kwargs,
        )

    def generate(self, *args, **kwargs) -> np.ndarray:
        return next_and_close(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[np.ndarray, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    def generate_long_form(
        self,
        text: str,
        control: str = None,
        prompt_wav_path: str = None,
        prompt_text: str = None,
        reference_wav_path: str = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 4096,
        normalize: bool = False,
        denoise: bool = False,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        max_chars: int = 90,
        silence_ms: int = 300,
    ) -> np.ndarray:
        """Generate long-form speech as short anchored segments.

        The first generated segment becomes a stable reference voice for
        VoxCPM2 when no external reference is supplied. Later segments use the
        previous segment as the continuation prompt so long inputs do not rely
        on one very long autoregressive pass.
        """
        segments = _split_long_text_for_tts(text, max_chars=max_chars)
        control = (control or "").strip()

        first_text = f"({control}){segments[0]}" if control else segments[0]

        common_kwargs = {
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
            "min_len": min_len,
            "max_len": max_len,
            "normalize": normalize,
            "retry_badcase": retry_badcase,
            "retry_badcase_max_times": retry_badcase_max_times,
            "retry_badcase_ratio_threshold": retry_badcase_ratio_threshold,
        }

        if len(segments) == 1:
            return _as_mono_float32(
                self.generate(
                    text=first_text,
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=prompt_text,
                    reference_wav_path=reference_wav_path,
                    denoise=denoise,
                    **common_kwargs,
                )
            )

        import soundfile as sf

        sample_rate = self.tts_model.sample_rate
        silence_len = max(0, int(sample_rate * silence_ms / 1000.0))
        silence = np.zeros(silence_len, dtype=np.float32)
        outputs = []
        can_use_reference = isinstance(self.tts_model, VoxCPM2Model)

        with tempfile.TemporaryDirectory(prefix="voxcpm_long_form_") as tmp_dir:
            def write_segment(name: str, wav: np.ndarray) -> str:
                path = os.path.join(tmp_dir, name)
                sf.write(path, _as_mono_float32(wav), sample_rate)
                return path

            first_wav = _as_mono_float32(
                self.generate(
                    text=first_text,
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=prompt_text,
                    reference_wav_path=reference_wav_path,
                    denoise=denoise,
                    **common_kwargs,
                )
            )
            outputs.append(first_wav)

            stable_reference_path = reference_wav_path
            if stable_reference_path is None and can_use_reference:
                stable_reference_path = write_segment("seed_reference.wav", first_wav)

            seed_prompt_path = write_segment("segment_001.wav", first_wav)
            seed_prompt_text = first_text

            for index, segment in enumerate(segments[1:], start=2):
                wav = _as_mono_float32(
                    self.generate(
                        text=segment,
                        prompt_wav_path=seed_prompt_path,
                        prompt_text=seed_prompt_text,
                        reference_wav_path=stable_reference_path if can_use_reference else None,
                        denoise=False,
                        **common_kwargs,
                    )
                )
                if silence_len:
                    outputs.append(silence)
                outputs.append(wav)
                write_segment(f"segment_{index:03d}.wav", wav)

        return np.concatenate(outputs).astype(np.float32, copy=False)

    def _generate(
        self,
        text: str,
        prompt_wav_path: str = None,
        prompt_text: str = None,
        reference_wav_path: str = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 4096,
        normalize: bool = False,
        denoise: bool = False,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        streaming: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        """Synthesize speech for the given text and return a single waveform.

        Args:
            text: Input text to synthesize.
            prompt_wav_path: Path to prompt audio for continuation mode.
                Must be paired with ``prompt_text``.
            prompt_text: Text content corresponding to the prompt audio.
            reference_wav_path: Path to reference audio for voice cloning
                (structurally isolated via ref_audio tokens). Can be used
                alone or combined with ``prompt_wav_path`` + ``prompt_text``.
            cfg_value: Guidance scale for the generation model.
            inference_timesteps: Number of inference steps.
            min_len: Minimum audio length.
            max_len: Maximum token length during generation.
            normalize: Whether to run text normalization before generation.
            denoise: Whether to denoise the prompt/reference audio if a
                denoiser is available.
            retry_badcase: Whether to retry badcase.
            retry_badcase_max_times: Maximum number of times to retry badcase.
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio.
            streaming: Whether to return a generator of audio chunks.
        Returns:
            Generator of numpy.ndarray: 1D waveform array (float32) on CPU.
            Yields audio chunks for each generation step if ``streaming=True``,
            otherwise yields a single array containing the final audio.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("target text must be a non-empty string")

        if prompt_wav_path is not None:
            if not os.path.exists(prompt_wav_path):
                raise FileNotFoundError(f"prompt_wav_path does not exist: {prompt_wav_path}")

        if reference_wav_path is not None:
            if not os.path.exists(reference_wav_path):
                raise FileNotFoundError(f"reference_wav_path does not exist: {reference_wav_path}")

        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")

        is_v2 = isinstance(self.tts_model, VoxCPM2Model)
        if reference_wav_path is not None and not is_v2:
            raise ValueError("reference_wav_path is only supported with VoxCPM2 models")

        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        temp_files = []

        try:
            actual_prompt_path = prompt_wav_path
            actual_ref_path = reference_wav_path

            if denoise and self.denoiser is not None:
                if prompt_wav_path is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        temp_files.append(tmp.name)
                    self.denoiser.enhance(prompt_wav_path, output_path=temp_files[-1])
                    actual_prompt_path = temp_files[-1]
                if reference_wav_path is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        temp_files.append(tmp.name)
                    self.denoiser.enhance(reference_wav_path, output_path=temp_files[-1])
                    actual_ref_path = temp_files[-1]

            if actual_prompt_path is not None or actual_ref_path is not None:
                if is_v2:
                    fixed_prompt_cache = self.tts_model.build_prompt_cache(
                        prompt_text=prompt_text,
                        prompt_wav_path=actual_prompt_path,
                        reference_wav_path=actual_ref_path,
                    )
                else:
                    fixed_prompt_cache = self.tts_model.build_prompt_cache(
                        prompt_text=prompt_text,
                        prompt_wav_path=actual_prompt_path,
                    )
            else:
                fixed_prompt_cache = None

            if normalize:
                if self.text_normalizer is None:
                    from .utils.text_normalize import TextNormalizer

                    self.text_normalizer = TextNormalizer()
                text = self.text_normalizer.normalize(text)

            generate_result = self.tts_model._generate_with_prompt_cache(
                target_text=text,
                prompt_cache=fixed_prompt_cache,
                min_len=min_len,
                max_len=max_len,
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                retry_badcase=retry_badcase,
                retry_badcase_max_times=retry_badcase_max_times,
                retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                streaming=streaming,
            )

            if streaming:
                try:
                    for wav, _, _ in generate_result:
                        yield wav.squeeze(0).cpu().numpy()
                finally:
                    generate_result.close()
            else:
                wav, _, _ = next_and_close(generate_result)
                yield wav.squeeze(0).cpu().numpy()

        finally:
            for tmp_path in temp_files:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    # ------------------------------------------------------------------ #
    # LoRA Interface (delegated to VoxCPMModel)
    # ------------------------------------------------------------------ #
    def load_lora(self, lora_weights_path: str) -> tuple:
        """Load LoRA weights from a checkpoint file.

        Args:
            lora_weights_path: Path to LoRA weights (.pth file or directory
                containing lora_weights.ckpt).

        Returns:
            tuple: (loaded_keys, skipped_keys) - lists of loaded and skipped parameter names.

        Raises:
            RuntimeError: If model was not initialized with LoRA config.
        """
        if self.tts_model.lora_config is None:
            raise RuntimeError(
                "Cannot load LoRA weights: model was not initialized with LoRA config. "
                "Please reinitialize with lora_config or lora_weights_path parameter."
            )
        return self.tts_model.load_lora_weights(lora_weights_path)

    def unload_lora(self):
        """Unload LoRA by resetting all LoRA weights to initial state (effectively disabling LoRA)."""
        self.tts_model.reset_lora_weights()

    def set_lora_enabled(self, enabled: bool):
        """Enable or disable LoRA layers without unloading weights.

        Args:
            enabled: If True, LoRA layers are active; if False, only base model is used.
        """
        self.tts_model.set_lora_enabled(enabled)

    def get_lora_state_dict(self) -> dict:
        """Get current LoRA parameters state dict.

        Returns:
            dict: State dict containing all LoRA parameters (lora_A, lora_B).
        """
        return self.tts_model.get_lora_state_dict()

    @property
    def lora_enabled(self) -> bool:
        """Check if LoRA is currently configured."""
        return self.tts_model.lora_config is not None
