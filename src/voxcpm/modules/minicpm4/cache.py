import os
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
        # Buffer preallocated para construir la mascara de atencion sin un
        # torch.arange nuevo por capa y por paso.
        self.position_arange = torch.arange(max_length, device=device)
        self.current_length = 0
        # Opt-in: atender solo sobre la ventana de posiciones validas en vez de
        # todo max_length. Es mas rapido pero NO bit-exact (el tiling de SDPA
        # cambia con la longitud de K, alterando el orden de reduccion) — el
        # efecto es equivalente a cambiar de seed, sin cambio de calidad en
        # distribucion. Desactivado por defecto.
        self.window_enabled = os.environ.get("VOXCPM_KV_WINDOW", "") == "1"

    def window_length(self, bucket: int = 256) -> int:
        """Longitud de atencion redondeada a multiplos de `bucket` (o max_length si esta desactivado)."""
        if not self.window_enabled:
            return self.max_length
        return min(self.max_length, -(-self.current_length // bucket) * bucket)

    def get_layer_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx]

    def step(self) -> int:
        if self.current_length >= self.max_length:
            raise ValueError("KV cache is full")

        ret = self.current_length
        self.current_length += 1
        return ret

    def fill_caches(self, kv_caches: List[Tuple[torch.Tensor, torch.Tensor]]):
        # No hace falta zero_(): las posiciones > current_length quedan siempre
        # excluidas por la mascara de atencion, y las <= se sobreescriben aqui.
        self.current_length = kv_caches[0][0].size(2)
        for i in range(self.num_layers):
            self.kv_cache[0, i, :, :, : self.current_length, :] = kv_caches[i][0]
            self.kv_cache[1, i, :, :, : self.current_length, :] = kv_caches[i][1]
