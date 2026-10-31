"""Temporarily promote a model for a practical CPU reference on non-BF16 hosts."""
from contextlib import contextmanager
import torch


@contextmanager
def float32_reference(model):
    # Mutate in place: deepcopy would leave the repo's LoRA function closures
    # pointing at the original Linear objects. Restore every original dtype.
    tensors = [("parameter",n,t.dtype) for n,t in model.named_parameters()]
    tensors += [("buffer",n,t.dtype) for n,t in model.named_buffers()]
    attributes = [(m,m.dtype) for m in model.modules()
                  if isinstance(getattr(m,"dtype",None),torch.dtype)]
    try:
        model.float()  # Unlike .to(float32), this preserves complex RoPE buffers.
        for module,_ in attributes:
            module.dtype = torch.float32
        yield model
    finally:
        for kind,name,dtype in tensors:
            tensor = model.get_parameter(name) if kind == "parameter" else model.get_buffer(name)
            tensor.data = tensor.data.to(dtype)
        for module,dtype in attributes:
            module.dtype = dtype
