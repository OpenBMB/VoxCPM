from typing import List, Tuple
import torch


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
        self.batch_size = batch_size

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
        # Track current position per batch item
        self.current_length = torch.zeros(batch_size, dtype=torch.long, device=device)

    def get_layer_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx]

    def step(self, batch_idx: int | None = None) -> int | torch.Tensor:
        """
        Advance the position counter and return the position index to use.

        Args:
            batch_idx: If provided, step only that batch item and return its position.
                       If None, step all batch items and return a tensor of positions.

        Returns:
            Position index (int if batch_idx provided, tensor of shape [batch_size] otherwise)
        """
        if batch_idx is not None:
            if self.current_length[batch_idx] >= self.max_length:
                raise ValueError(f"KV cache is full for batch item {batch_idx}")
            ret = self.current_length[batch_idx].item()
            self.current_length[batch_idx] += 1
            return ret
        else:
            if (self.current_length >= self.max_length).any():
                raise ValueError("KV cache is full for one or more batch items")
            ret = self.current_length.clone()
            self.current_length += 1
            return ret

    def fill_caches(self, kv_caches: List[Tuple[torch.Tensor, torch.Tensor]]):
        # kv_caches[i] = (key, value) with shape [batch_size, num_kv_heads, seq_len, dim_kv_head]
        seq_len = kv_caches[0][0].size(2)
        self.current_length.fill_(seq_len)
        self.kv_cache.zero_()
        for i in range(self.num_layers):
            self.kv_cache[0, i, :, :, :seq_len, :] = kv_caches[i][0]
            self.kv_cache[1, i, :, :, :seq_len, :] = kv_caches[i][1]

    def reset(self, batch_idx: int | None = None):
        """Reset position counter(s) to zero."""
        if batch_idx is not None:
            self.current_length[batch_idx] = 0
        else:
            self.current_length.zero_()
