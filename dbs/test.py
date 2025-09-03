"""
Test EG coding accuracy and performance.
"""

import time

import torch

from eg_coding import encode, decode

LENGTH = 100000
MAX_VALUE = 1000
ITERS = 100

numbers = torch.randint(0, MAX_VALUE, (LENGTH,), dtype=torch.uint32)

# Warmup
for _ in range(10):
    y = encode(numbers)
    x = decode(y)

# Test
time_start = time.time()
correct = True
for _ in range(ITERS):
    y = encode(numbers)
    x = decode(y)
    if not torch.equal(x, numbers):
        correct = False
elapse = time.time() - time_start

time_per_iter = elapse / ITERS
time_per_num = time_per_iter / LENGTH
print("Exponential Golomb coding performance test")
print(f"  Length: {LENGTH}")
print(f"  Max value: {MAX_VALUE}")
print(f"  Iterations: {ITERS}")
print("Test results")
print(f"  Accurate encoding and decoding: {correct}")
print(f"  Time elapsed: {elapse:.2f} seconds")
print(f"  Time per iter: {time_per_iter * 1e3:.1f} ms")
print(f"  Time per number: {time_per_num * 1e9:.1f} ns")
