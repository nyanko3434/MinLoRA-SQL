"""From-scratch LoRA adapter injection, shape-compatible with peft's storage convention."""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear (or bitsandbytes Linear4bit) with a low-rank adapter."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        for p in base_layer.parameters():
            p.requires_grad_(False)
        self.base_layer = base_layer

        # Never read base_layer.weight.shape: under bitsandbytes Linear4bit the weight
        # is packed and its shape does not reflect the logical in/out dimensions.
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        dtype = dtype or getattr(base_layer, "compute_dtype", base_layer.weight.dtype)
        device = device or base_layer.weight.device

        # Shapes match peft's nn.Linear(in_features, r) / nn.Linear(r, out_features)
        # storage convention exactly, so checkpoint weights can be copied with no transpose.
        self.lora_A = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

    def forward(self, x):
        base_out = self.base_layer(x)
        h = self.lora_dropout(x).to(self.lora_A.dtype)
        # Keep the matmuls grouped this way: the other grouping materializes an
        # (out_features x in_features) matrix, exactly what LoRA exists to avoid.
        lora_out = (h @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out.to(base_out.dtype)

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling}"


def inject_lora(
    model: nn.Module,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> nn.Module:
    """Replace nn.Linear children named in target_modules with LoRALinear wrappers, in place."""
    # Freeze the whole model up front. LoRALinear only freezes the base_layer it wraps, so
    # without this, every non-targeted param (embeddings, lm_head, norms, untargeted
    # projections) stays trainable from from_pretrained's default and silently trains alongside
    # the adapters.
    for p in model.parameters():
        p.requires_grad_(False)

    candidates = []
    linear_names_seen = []
    for _, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                linear_names_seen.append(child_name)
            if child_name in target_modules and isinstance(child, nn.Linear):
                candidates.append((parent, child_name, child))

    if not candidates:
        raise ValueError(
            f"inject_lora found no nn.Linear children matching target_modules={target_modules!r}. "
            f"nn.Linear child names found in model: {sorted(set(linear_names_seen))}"
        )

    # Mutate only after the full walk above; mutating the tree mid-iteration is fragile.
    for parent, child_name, child in candidates:
        setattr(parent, child_name, LoRALinear(child, r=rank, alpha=alpha, dropout=dropout))

    return model


def lora_param_stats(model: nn.Module) -> dict:
    """Aggregate parameter/layer counts for reporting adapter size and trainability."""
    total = 0
    trainable = 0
    lora = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        if "lora_" in name:
            lora += n

    n_lora_layers = sum(1 for m in model.modules() if isinstance(m, LoRALinear))

    return {
        "total": total,
        "trainable": trainable,
        "lora": lora,
        "n_lora_layers": n_lora_layers,
    }
