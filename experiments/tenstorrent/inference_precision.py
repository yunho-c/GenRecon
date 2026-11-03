"""Promote a disposable CPU eval model for an FP32 device diagnostic.

Apply after strict loading, stock references and other preparation. Preserve
the loaded quantized values and parameter identities used by LoRA closures.
Complex phases and integer buffers retain their original types and values.
"""
from collections import Counter

import torch


def prepare_float32_device(model):
    if model.training:
        raise ValueError('FP32 device preparation requires model.eval()')
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    tensors = parameters + buffers
    if any(value.device.type != 'cpu' for value in tensors):
        raise ValueError('Prepare FP32 on CPU before device transfer')
    original_dtypes = Counter(str(value.dtype) for value in parameters)
    original_bytes = sum(value.numel()*value.element_size() for value in parameters)
    changed_parameters = changed_buffers = 0
    with torch.no_grad():
        for group, values in [('parameter', parameters), ('buffer', buffers)]:
            for value in values:
                if value.is_floating_point() and value.dtype != torch.float32:
                    value.data = value.data.float()
                    if group == 'parameter':
                        changed_parameters += 1
                    else:
                        changed_buffers += 1
    attributes = 0
    for module in model.modules():
        if isinstance(getattr(module, 'dtype', None), torch.dtype):
            module.dtype = torch.float32
            attributes += 1
    return {'changed_parameters': changed_parameters, 'changed_buffers': changed_buffers,
            'dtype_attributes': attributes,
            'parameter_dtype_counts_before': dict(original_dtypes),
            'parameter_dtype_counts_after': dict(Counter(str(p.dtype) for p in parameters)),
            'parameter_bytes_before': original_bytes,
            'parameter_bytes_after': sum(p.numel()*p.element_size() for p in parameters),
            'note': 'Original stock CPU reference unchanged. Promote already loaded/quantized values, not original full-precision checkpoint values. Complex and integer buffers preserved.'}
