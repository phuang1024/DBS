import torch
from torch import nn


def get_residual(weights):
    """Get the order of the first significant digit of the tensors"""
    signs = torch.sign(weights)
    exps = torch.round(torch.log2(torch.abs(weights)))
    pow_weights = signs * torch.pow(2, exps)
    return pow_weights, exps


def rf8(model, n=4):
    """Residual Float-Point 8-bit Model Quantization"""
    with torch.no_grad():
        for param in model.parameters():
            data1, exps1 = get_residual(param.data)
            data2, exps2 = get_residual(param.data - data1)
            flags = (exps1-exps2 <= n)
            param.data = data1 + flags * data2