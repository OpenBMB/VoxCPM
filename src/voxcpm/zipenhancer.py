"""
ZipEnhancer Module - Audio Denoising Enhancer

Provides on-demand import ZipEnhancer functionality for audio denoising processing.
Related dependencies are imported only when denoising functionality is needed.
"""

import os
import tempfile
from typing import Optional
import torchaudio
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class ZipEnhancer:
    """ZipEnhancer Audio Denoising Enhancer"""

    def __init__(self, model_path: str = "iic/speech_zipenhancer_ans_multiloss_16k_base"):
        """
        Initialize ZipEnhancer
        Args:
            model_path: ModelScope model path or local path
        """
        self.model_path = model_path
        self._pipeline = pipeline(Tasks.acoustic_noise_suppression, model=self.model_path)

    def _normalize_loudness(self, wav_path: str):
        """
        Audio loudness normalization

        Args:
            wav_path: Audio file path
        """
        audio, sr = torchaudio.load(wav_path)
        loudness = torchaudio.functional.loudness(audio, sr)
        normalized_audio = torchaudio.functional.gain(audio, -20 - loudness)
        torchaudio.save(wav_path, normalized_audio, sr)

    def enhance(self, input_path: str, output_path: Optional[str] = None, normalize_loudness: bool = True) -> str:
        """
        Audio denoising enhancement
        Args:
            input_path: Input audio file path
            output_path: Output audio file path (optional, creates temp file by default)
            normalize_loudness: Whether to perform loudness normalization
        Returns:
            str: Output audio file path
        Raises:
            RuntimeError: If pipeline is not initialized or processing fails
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file does not exist: {input_path}")
        tmp_created = False
        # Create temporary file if no output path is specified
        if output_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                output_path = tmp_file.name
            tmp_created = True
        try:
            # Perform denoising processing
            self._pipeline(input_path, output_path=output_path)
            # Loudness normalization
            if normalize_loudness:
                self._normalize_loudness(output_path)
            return output_path
        except Exception as e:
            # Clean up temporary files only if we created them
            if tmp_created and output_path and os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except OSError:
                    pass
            raise RuntimeError(f"Audio denoising processing failed: {e}")
