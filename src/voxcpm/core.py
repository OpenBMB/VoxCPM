import os
import re
import tempfile
import numpy as np
from typing import Generator
from huggingface_hub import snapshot_download
from .model.voxcpm import VoxCPMModel

class VoxCPM:
    def __init__(self,
            voxcpm_model_path : str,
            zipenhancer_model_path : str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
            enable_denoiser : bool = True,
            optimize: bool = True,
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
        """
        print(f"voxcpm_model_path: {voxcpm_model_path}, zipenhancer_model_path: {zipenhancer_model_path}, enable_denoiser: {enable_denoiser}")
        self.tts_model = VoxCPMModel.from_local(voxcpm_model_path, optimize=optimize)
        self.text_normalizer = None
        if enable_denoiser and zipenhancer_model_path is not None:
            from .zipenhancer import ZipEnhancer
            self.denoiser = ZipEnhancer(zipenhancer_model_path)
        else:
            self.denoiser = None
        print("Warm up VoxCPMModel...")
        self.tts_model.generate(
            target_text="Hello, this is the first test sentence.",
            max_len=10,
        )

    @classmethod
    def from_pretrained(cls,
            hf_model_id: str = "openbmb/VoxCPM-0.5B",
            load_denoiser: bool = True,
            zipenhancer_model_id: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
            cache_dir: str = None,
            local_files_only: bool = False,
            **kwargs,
        ):
        """Instantiate ``VoxCPM`` from a Hugging Face Hub snapshot.

        Args:
            hf_model_id: Explicit Hugging Face repository id (e.g. "org/repo") or local path.
            load_denoiser: Whether to initialize the denoiser pipeline.
            zipenhancer_model_id: Denoiser model id or path for ModelScope
                acoustic noise suppression.
            cache_dir: Custom cache directory for the snapshot.
            local_files_only: If True, only use local files and do not attempt
                to download.
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
            **kwargs,
        )

    def generate(self, *args, **kwargs) -> np.ndarray:
        return next(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[np.ndarray, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    def _generate(self, 
            text : str,
            prompt_wav_path : str = None,
            prompt_text : str = None,
            cfg_value : float = 2.0,    
            inference_timesteps : int = 10,
            max_length : int = 4096,
            normalize : bool = True,
            denoise : bool = True,
            retry_badcase : bool = True,
            retry_badcase_max_times : int = 3,
            retry_badcase_ratio_threshold : float = 6.0,
            streaming: bool = False,
        ) -> Generator[np.ndarray, None, None]:
        """Synthesize speech for the given text and return a single waveform.

        This method optionally builds and reuses a prompt cache. If an external
        prompt (``prompt_wav_path`` + ``prompt_text``) is provided, it will be
        used for all sub-sentences. Otherwise, the prompt cache is built from
        the first generated result and reused for the remaining text chunks.

        Args:
            text: Input text. Can include newlines; each non-empty line is
                treated as a sub-sentence.
            prompt_wav_path: Path to a reference audio file for prompting.
            prompt_text: Text content corresponding to the prompt audio.
            cfg_value: Guidance scale for the generation model.
            inference_timesteps: Number of inference steps.
            max_length: Maximum token length during generation.
            normalize: Whether to run text normalization before generation.
            denoise: Whether to denoise the prompt audio if a denoiser is
                available.
            retry_badcase: Whether to retry badcase.
            retry_badcase_max_times: Maximum number of times to retry badcase.
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio.
            streaming: Whether to return a generator of audio chunks.
        Returns:
            Generator of numpy.ndarray: 1D waveform array (float32) on CPU. 
            Yields audio chunks for each generations step if ``streaming=True``,
            otherwise yields a single array containing the final audio.
        """
        if not text.strip() or not isinstance(text, str):
            raise ValueError("target text must be a non-empty string")
        
        if prompt_wav_path is not None:
            if not os.path.exists(prompt_wav_path):
                raise FileNotFoundError(f"prompt_wav_path does not exist: {prompt_wav_path}")
        
        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")
        
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)
        temp_prompt_wav_path = None
        
        try:
            if prompt_wav_path is not None and prompt_text is not None:
                if denoise and self.denoiser is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        temp_prompt_wav_path = tmp_file.name
                    self.denoiser.enhance(prompt_wav_path, output_path=temp_prompt_wav_path)
                    prompt_wav_path = temp_prompt_wav_path
                fixed_prompt_cache = self.tts_model.build_prompt_cache(
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=prompt_text
                )
            else:
                fixed_prompt_cache = None  # will be built from the first inference
            
            if normalize:
                if self.text_normalizer is None:
                    from .utils.text_normalize import TextNormalizer
                    self.text_normalizer = TextNormalizer()
                text = self.text_normalizer.normalize(text)
            
            generate_result = self.tts_model._generate_with_prompt_cache(
                            target_text=text,
                            prompt_cache=fixed_prompt_cache,
                            min_len=2,
                            max_len=max_length,
                            inference_timesteps=inference_timesteps,
                            cfg_value=cfg_value,
                            retry_badcase=retry_badcase,
                            retry_badcase_max_times=retry_badcase_max_times,
                            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                            streaming=streaming,
                        )
        
            for wav, _, _ in generate_result:
                yield wav.squeeze(0).cpu().numpy()
        
        finally:
            if temp_prompt_wav_path and os.path.exists(temp_prompt_wav_path):
                try:
                    os.unlink(temp_prompt_wav_path)
                except OSError:
                    pass  