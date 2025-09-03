"""
Wrapper around eg_coding.cpp
Exposes Python bindings.

Functions:
encode(x) -> y:
    x: 1D int32 tensor.
    y: 1D uint8 tensor.
decode(y) -> x:
    y: 1D uint8 tensor.
    x: 1D int32 tensor.
"""

import torch
from torch.utils.cpp_extension import load

_eg_coding = load(
    name="eg_coding",
    sources=["eg_coding.cpp"],
    verbose=True,
)

encode = _eg_coding.encode
decode = _eg_coding.decode

# Warm up
encode(torch.randint(-1000, 1000, (10,), dtype=torch.int32))
