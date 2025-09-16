"""
Wrapper around eg_coding.cpp
Exposes Python bindings.

Functions:
encode(x) -> y:
    x: 1D int64 tensor.
    y: 1D uint64 tensor.
decode(y) -> x:
    y: 1D uint64 tensor.
    x: 1D int64 tensor.
"""

import torch
from torch.utils.cpp_extension import load

_eg_coding = load(
    name="eg_coding",
    sources=["eg_coding.cpp"],
    extra_cflags=["-std=c++20"],
    verbose=True,
)

encode = _eg_coding.encode
decode = _eg_coding.decode

# Warm up
encode(torch.randint(-1000, 1000, (10,), dtype=torch.int64))


## Tests

def find_error():
    def erroneous(t):
        return not torch.equal(decode(encode(t)), t)

    while True:
        values = torch.randint(-1000, 1000, (1000,), dtype=torch.int64)
        if erroneous(values):
            break

    a = 0
    b = 1000
    while True:
        mid = (a + b) // 2
        decision = None
        if erroneous(values[a:mid]):
            decision = "LOWER"
        elif erroneous(values[mid:b]):
            decision = "UPPER"
        else:
            v = values[a:b]
            print(v)
            print("encode(v):", encode(v))
            print("decode(encode(v)):", decode(encode(v)))
            return v
        if decision == "LOWER":
            b = mid
        else:
            a = mid


def find_error_shortest(iters=100):
    best = None
    for _ in range(iters):
        error = find_error()
        if best is None or error.numel() < best.numel():
            best = error

    print("Shortest error:", best)


find_error_shortest()
