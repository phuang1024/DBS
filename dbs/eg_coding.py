"""
Wrapper around eg_coding.cpp
Exposes Python bindings.

Functions:
encode(x) -> y:
    x: 1D uint32 tensor.
    y: 1D uint8 tensor.
decode(y) -> x:
    y: 1D uint8 tensor.
    x: 1D uint32 tensor.
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
