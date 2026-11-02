"""Opt-in frozen LoRA candidates for measurement, never training or loading.

Prepare only after strict checkpoint loading and dtype conversion. Merging and
low-rank evaluation change rounding; neither is an exact BF16 replacement.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PreparedLoRALinear(nn.Module):
    def __init__(self, source, mode):
        super().__init__()
        if mode not in ("merged", "lowrank", "frozen_delta"):
            raise ValueError(f"Unknown LoRA candidate: {mode}")
        if source.training or source.weight.device.type != "cpu":
            raise ValueError("Prepare evaluated CPU weights before device upload")
        self.mode = mode
        self.scaling = source.scaling
        with torch.no_grad():
            if mode == "merged":
                # Preserve the source dtype's B@A rounding before merging.
                delta = source.lora_B @ source.lora_A
                weight = (source.weight.float() + delta.float() * self.scaling).to(source.weight.dtype)
            else:
                weight = source.weight.detach().clone()
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.bias = (nn.Parameter(source.bias.detach().clone(), requires_grad=False)
                         if source.bias is not None else None)
            if mode == "lowrank":
                self.lora_A = nn.Parameter(source.lora_A.detach().clone(), requires_grad=False)
                self.lora_B = nn.Parameter(source.lora_B.detach().clone(), requires_grad=False)
            elif mode == "frozen_delta":
                self.delta = nn.Parameter(source.lora_B @ source.lora_A, requires_grad=False)
        self.eval()

    def forward(self, x):
        if self.training:
            raise RuntimeError("Prepared LoRA is inference-only")
        base = F.linear(x, self.weight, self.bias)
        if self.mode == "merged":
            return base
        delta = (F.linear(F.linear(x, self.lora_A), self.lora_B)
                 if self.mode == "lowrank" else F.linear(x, self.delta))
        return base + delta * self.scaling


def prepare_lora(model, mode):
    """Replace patched linears in place; return the number prepared.

    The input model must be disposable: keep checkpoints and the stock reference
    separate. Prepared weights cannot be used to reload a stock state dictionary.
    """
    if mode == "stock":
        return 0
    if mode not in ("merged", "lowrank", "frozen_delta"):
        raise ValueError(f"Unknown LoRA candidate: {mode}")
    if model.training:
        raise ValueError("LoRA preparation requires model.eval()")
    count = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and hasattr(child, "lora_A"):
                setattr(parent, name, PreparedLoRALinear(child, mode))
                count += 1
    if not count:
        raise ValueError("No stock LoRA linears found; refusing an ineffective experiment")
    return count
