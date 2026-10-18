"""Quantized copies of LeWM for the Week 3 study.

- int8: PyTorch dynamic quantization of every nn.Linear (int8 weights, activations quantized
  on the fly), the CPU deployment path. CPU only.
- w6, w4, w3: weight-only, symmetric, per-output-channel round-to-nearest of every nn.Linear,
  computed in float. Runs on any device.
Linear layers hold 99% of the parameters; the two conv layers stay in float in every variant.
"""

import copy

import torch
from torch import nn

PRECISIONS = ('fp32', 'int8', 'w6', 'w4', 'w3')


def fake_quantize_weights(model, bits):
    model = copy.deepcopy(model)
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                w = mod.weight
                scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
                w.copy_((w / scale).round().clamp(-qmax, qmax) * scale)
    return model


def quantize(model, precision):
    if precision == 'fp32':
        return model
    if precision == 'int8':
        torch.backends.quantized.engine = 'qnnpack' if 'qnnpack' in torch.backends.quantized.supported_engines \
            else torch.backends.quantized.engine
        return torch.ao.quantization.quantize_dynamic(copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8)
    if precision.startswith('w'):
        return fake_quantize_weights(model, int(precision[1:]))
    raise ValueError(f'unknown precision {precision!r}; expected one of {PRECISIONS}')
