import math

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    """
    LoRA 线性层：直接持有 weight/bias，保持与 nn.Linear 相同的 state_dict key 结构。

    state_dict 结构：
        - weight: 原始权重（与 nn.Linear 一致）
        - bias: 原始偏置（与 nn.Linear 一致）
        - lora_A: LoRA 低秩矩阵 A
        - lora_B: LoRA 低秩矩阵 B

    这样设计的好处：加载预训练权重时无需做 key 转换。
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert isinstance(base, nn.Linear), "LoRALinear only supports wrapping nn.Linear."

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self._base_scaling = alpha / r if r > 0 else 0.0

        # 使用 buffer 存储 scaling，这样修改值不会触发 torch.compile 重编译
        # persistent=False 表示不保存到 state_dict，避免加载时 missing key
        self.register_buffer("scaling", torch.tensor(self._base_scaling), persistent=False)

        # 直接持有 weight 和 bias（从原始 Linear 转移过来）
        self.weight = base.weight
        self.bias = base.bias  # 可能是 None

        # LoRA 参数
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基础 Linear 计算
        result = F.linear(x, self.weight, self.bias)
        if self.r <= 0 or self.lora_A is None:
            return result
        # LoRA: result + dropout(x @ A^T @ B^T) * scaling
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return result + self.dropout(lora_out) * self.scaling

    def reset_lora_parameters(self):
        """重置 LoRA 参数到初始状态"""
        if self.r > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def set_enabled(self, enabled: bool):
        """启用/禁用 LoRA（通过 scaling 控制，兼容 torch.compile）"""
        # 使用 fill_ 原地修改 buffer 值，不会触发重编译
        self.scaling.fill_(self._base_scaling if enabled else 0.0)

    @property
    def enabled(self) -> bool:
        return self.scaling.item() != 0.0


def _get_parent_module(root: nn.Module, name: str) -> nn.Module | None:
    """
    根据类似 'layers.0.self_attn.q_proj' 的全名，返回 parent module（即 q_proj 的上一级）。
    """
    parts = name.split(".")
    if len(parts) == 1:
        return root
    parent = root
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return None
        parent = getattr(parent, p)
    return parent


def apply_lora_to_named_linear_modules(
    root: nn.Module,
    *,
    target_submodule_names: list[str],
    r: int,
    alpha: float,
    dropout: float,
) -> None:
    """
    在给定模块及其子模块中，对名字以 target_submodule_names 结尾的 Linear 层注入 LoRA。

    例如 target_submodule_names=["q_proj", "v_proj"] 时，
    会在所有名为 *.q_proj / *.v_proj 的 nn.Linear 上替换为 LoRALinear。
    """
    for full_name, module in list(root.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        short_name = full_name.split(".")[-1]
        if short_name not in target_submodule_names:
            continue

        parent = _get_parent_module(root, full_name)
        if parent is None:
            continue

        # 用 LoRALinear 替换原始 Linear
        lora_layer = LoRALinear(
            base=module,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )
        setattr(parent, short_name, lora_layer)
