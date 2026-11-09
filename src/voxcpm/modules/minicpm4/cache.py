from typing import List, Tuple
import torch


def _prompt_length(kv_caches: List[Tuple[torch.Tensor, torch.Tensor]], max_length: int) -> int:
    """Length of a prompt cache, rejected when it cannot fit in the buffer."""
    if not kv_caches:
        raise ValueError("KV cache fill requires at least one layer")
    seq_len = int(kv_caches[0][0].size(2))
    if seq_len > max_length:
        raise ValueError(f"KV cache prompt length {seq_len} exceeds max_length {max_length}")
    return seq_len


class StaticKVCache:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        dim_kv_head: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        max_length: int = 8192,
    ):
        self.max_length = max_length
        self.num_layers = num_layers

        self.kv_cache = torch.zeros(
            2,
            num_layers,
            batch_size,
            num_kv_heads,
            max_length,
            dim_kv_head,
            device=device,
            dtype=dtype,
        )
        self.current_length = 0

    def get_layer_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx]

    def step(self) -> int:
        if self.current_length >= self.max_length:
            raise ValueError("KV cache is full")

        ret = self.current_length
        self.current_length += 1
        return ret

    def fill_caches(self, kv_caches: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.current_length = _prompt_length(kv_caches, self.max_length)
        self.kv_cache.zero_()
        for i in range(self.num_layers):
            self.kv_cache[0, i, :, :, : self.current_length, :] = kv_caches[i][0]
            self.kv_cache[1, i, :, :, : self.current_length, :] = kv_caches[i][1]
