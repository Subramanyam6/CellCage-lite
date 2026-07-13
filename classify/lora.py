"""LoRA adapters for the optional DINOv2 fine-tune.

By default DINOv2 stays frozen and only the head is trained. When enough labeled
cells are available to justify adjusting the backbone, LoRA adds a small pair of
low-rank matrices beside each target linear layer and trains only those (well
under 1% of the model), so the general-purpose features underneath are preserved.

A LoRA-wrapped linear layer computes ``W x + (B A) x * (alpha / rank)``, where
only ``A`` and ``B`` are trainable and ``W`` (the original weight) is frozen.

torch is imported lazily; importing this module does not require it.
"""

from __future__ import annotations


def make_lora_linear(base_linear, rank: int = 8, alpha: float = 16.0):
    """Wrap a ``torch.nn.Linear`` with a trainable low-rank adapter.

    The original layer is frozen; only the two low-rank matrices are trained.
    """
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad = False
            self.A = nn.Parameter(torch.zeros(rank, base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, rank))
            nn.init.kaiming_uniform_(self.A, a=5**0.5)
            self.scaling = alpha / rank

        def forward(self, x):
            return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scaling

    return LoRALinear(base_linear, rank, alpha)


def inject_lora(model, target_substrings=("qkv", "proj"), rank: int = 8, alpha: float = 16.0) -> int:
    """Replace matching linear layers in ``model`` with LoRA-wrapped versions.

    Returns the number of layers wrapped. Only layers whose qualified name
    contains one of ``target_substrings`` (the attention projections, by default)
    are adapted.
    """
    from torch import nn

    wrapped = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(s in full for s in target_substrings):
                setattr(module, child_name, make_lora_linear(child, rank, alpha))
                wrapped += 1
    return wrapped
