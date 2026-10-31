"""
int8 weight quantization, written from scratch.

Normally every weight is a float32 (4 bytes). Quantization stores each weight
as an int8 (1 byte, a whole number from -127 to 127) plus ONE float "scale"
per output row:

    scale  = max(|row|) / 127
    q      = round(w / scale)         # int8
    w_back = q * scale                # what we use at run time (tiny rounding error)

That makes the Linear layers ~4x smaller in memory/on disk, at the cost of a
little rounding error. This is "weight-only, per-channel, symmetric" int8,
the same basic idea behind LLM.int8 / GPTQ / llama.cpp's Q8_0 (they add tricks
on top for bigger models).
"""

import torch
import torch.nn as nn


class Int8Linear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        w = linear.weight.detach().float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0  # one scale per output row
        self.register_buffer("qweight", torch.round(w / scale).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale)
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x):
        # dequantize on the fly; real int8 kernels would multiply int8 x int8 directly
        w = self.qweight.to(x.dtype) * self.scale.to(x.dtype)
        return x @ w.T


def quantize_model(model, skip=("lm_head",)):
    """swap every nn.Linear (except the tied output layer) for an Int8Linear, in place"""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name not in skip:
                setattr(module, name, Int8Linear(child))
    return model


def model_bytes(model):
    """bytes used by all weights (parameters + buffers like int8 weights and scales)"""
    seen, total = set(), 0
    for t in model.state_dict().values():  # saved tensors only (skips the RoPE cos/sin tables)
        if t.data_ptr() in seen:
            continue  # tied weights counted once
        seen.add(t.data_ptr())
        total += t.numel() * t.element_size()
    return total


def quantization_error(model_fp, model_q):
    """mean relative error of the weights, averaged over the quantized layers"""
    errs = []
    fp = dict(model_fp.named_modules())
    for name, m in model_q.named_modules():
        if isinstance(m, Int8Linear):
            w = fp[name].weight.detach().float()
            wq = m.qweight.float() * m.scale
            errs.append(((w - wq).norm() / w.norm()).item())
    return sum(errs) / len(errs)
