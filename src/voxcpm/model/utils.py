import os
from typing import List, Optional
import torch
from transformers import PreTrainedTokenizer

_LOW_PRECISION_DTYPES = {"bfloat16", "bf16", "float16", "fp16"}
_VALID_DTYPE_OVERRIDES = {
    "bfloat16", "bf16",
    "float16", "fp16",
    "float32", "fp32",
}


# Ref: https://github.com/OpenBMB/VoxCPM/issues/256#issuecomment-4235252732
# Explicitly close partially-consumed generators so inference_mode cleanup
# does not get deferred to Python's GC/finalizer path.
def next_and_close(gen):
    try:
        return next(gen)
    finally:
        gen.close()


def mask_multichar_chinese_tokens(tokenizer: PreTrainedTokenizer):
    """Create a tokenizer wrapper that converts multi-character Chinese tokens to single characters.

    This function creates a wrapper around the provided tokenizer that automatically
    splits multi-character Chinese tokens into individual characters. This is useful
    for ensuring consistent tokenization of Chinese text.

    Args:
        tokenizer: The base tokenizer to wrap

    Returns:
        A CharTokenizerWrapper instance that handles multi-character Chinese tokens

    Example:
        >>> from transformers import LlamaTokenizerFast
        >>> tokenizer = LlamaTokenizerFast.from_pretrained("path/to/tokenizer")
        >>> wrapped_tokenizer = mask_multichar_chinese_tokens(tokenizer)
        >>> tokens = wrapped_tokenizer("你好世界")
    """
    # Pre-compute multi-character tokens (length >= 2, pure Chinese characters)
    multichar_tokens = {
        token for token in tokenizer.vocab.keys() if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
    }

    class CharTokenizerWrapper:
        """Wrapper class for tokenizers that handles multi-character Chinese tokens.

        This wrapper automatically splits multi-character Chinese tokens into
        individual characters while preserving the original tokenizer's interface.
        """

        def __init__(self, base_tokenizer: PreTrainedTokenizer) -> None:
            """Initialize the wrapper with a base tokenizer.

            Args:
                base_tokenizer: The tokenizer to wrap
            """
            self.tokenizer = base_tokenizer
            self.multichar_tokens = multichar_tokens

        def tokenize(self, text: str, **kwargs) -> List[str]:
            """Tokenize text and split multi-character Chinese tokens into single characters.

            Args:
                text: Input text to tokenize
                **kwargs: Additional arguments passed to the base tokenizer

            Returns:
                List of processed tokens with multi-character Chinese tokens split

            Example:
                >>> wrapper = CharTokenizerWrapper(tokenizer)
                >>> tokens = wrapper.tokenize("你好世界")
                >>> # Returns ["你", "好", "世", "界"] instead of ["你好", "世界"]
            """
            if not isinstance(text, str):
                raise TypeError(f"Expected string input, got {type(text)}")

            tokens = self.tokenizer.tokenize(text, **kwargs)
            processed = []

            for token in tokens:
                # Remove possible subword prefix
                clean_token = token.replace("▁", "")

                if clean_token in self.multichar_tokens:
                    # Split multi-character token into single characters
                    chars = list(clean_token)
                    processed.extend(chars)
                else:
                    processed.append(token)

            return processed

        def __call__(self, text: str, **kwargs) -> List[int]:
            """Call the tokenizer and return token IDs.

            This method provides the same interface as the original tokenizer
            but with multi-character Chinese token handling.

            Args:
                text: Input text to tokenize
                **kwargs: Additional arguments passed to the base tokenizer

            Returns:
                List of token IDs

            Raises:
                TypeError: If input is not a string
                ValueError: If tokenization fails
            """
            try:
                tokens = self.tokenize(text, **kwargs)
                result = self.tokenizer.convert_tokens_to_ids(tokens)
                return result
            except Exception as e:
                raise ValueError(f"Tokenization failed: {str(e)}") from e

    return CharTokenizerWrapper(tokenizer)


def get_dtype(dtype: str):
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "bf16":
        return torch.bfloat16
    elif dtype == "float16":
        return torch.float16
    elif dtype == "fp16":
        return torch.float16
    elif dtype == "float32":
        return torch.float32
    elif dtype == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _has_mps() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def pick_runtime_dtype(device: str, configured_dtype: str) -> str:
    """Pick a safe runtime dtype for the resolved device.

    On Apple Silicon (MPS), bfloat16/float16 produce enough numerical drift
    in the diffusion AR loop that the output is glitched and the model's
    badcase detector triggers infinite retries. float32 is the only stable
    option today. CUDA and CPU keep whatever the checkpoint was trained with.

    Users can override with ``VOXCPM_MPS_DTYPE`` (e.g. ``bfloat16``) when
    they want to test future MPS improvements.
    """
    if device != "mps":
        return configured_dtype

    override = os.environ.get("VOXCPM_MPS_DTYPE", "").strip().lower()
    if override:
        if override not in _VALID_DTYPE_OVERRIDES:
            raise ValueError(
                f"VOXCPM_MPS_DTYPE='{override}' is not one of "
                f"{sorted(_VALID_DTYPE_OVERRIDES)}"
            )
        return override

    if (configured_dtype or "").lower() in _LOW_PRECISION_DTYPES:
        return "float32"
    return configured_dtype


def auto_select_device(preferred_device: Optional[str] = "cuda") -> str:
    """
    Choose a runtime device automatically.

    Preference order:
    - if the preferred device is available, use it
    - otherwise fall back to CUDA -> MPS -> CPU
    """
    preferred = (preferred_device or "cuda").strip().lower()

    if preferred.startswith("cuda") and torch.cuda.is_available():
        return preferred
    if preferred == "mps" and _has_mps():
        return "mps"
    if preferred == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if _has_mps():
        return "mps"
    return "cpu"


def resolve_runtime_device(device: Optional[str], configured_device: str = "cuda") -> str:
    """
    Resolve the actual runtime device.

    Semantics:
    - ``device`` is ``None`` or ``"auto"``: use automatic fallback selection
    - otherwise: treat it as an explicit user choice and validate availability
    """
    explicit = None if device is None else device.strip().lower()

    if explicit is None or explicit == "auto":
        return auto_select_device(configured_device)

    if explicit.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                f"Requested device '{device}', but CUDA is not available. "
                "Use device='auto' for automatic fallback."
            )
        return explicit
    if explicit == "mps":
        if not _has_mps():
            raise ValueError(
                "Requested device 'mps', but MPS is not available. "
                "Use device='auto' for automatic fallback."
            )
        return "mps"
    if explicit == "cpu":
        return "cpu"

    raise ValueError(
        f"Unsupported device '{device}'. Supported values are 'auto', 'cpu', 'mps', "
        "'cuda', or indexed CUDA devices like 'cuda:0'."
    )
