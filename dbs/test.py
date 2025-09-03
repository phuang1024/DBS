"""
Test EG coding accuracy and performance.
"""

import time

import torch

from eg_coding import encode, decode

ITERS = 100


def test_eg(numbers, iters=ITERS):
    # Warmup
    for _ in range(10):
        y = encode(numbers)
        x = decode(y)

    # Test
    time_start = time.time()
    for i in range(iters):
        y = encode(numbers)
        x = decode(y)

        if i == 0:
            correct = torch.equal(x, numbers)
            compressed_size = y.numel() * y.element_size()

    elapse = time.time() - time_start

    length = numbers.numel()
    data_size = numbers.numel() * numbers.element_size()
    time_per_iter = elapse / iters
    time_per_num = time_per_iter / length
    print("Exponential Golomb coding performance test")
    print(f"  Length: {length}")
    print(f"  Iterations: {iters}")
    print(f"  Original size: {data_size} bytes")
    print("Test results")
    print(f"  Accurate encoding and decoding: {correct}")
    print(f"  Compressed size: {compressed_size} bytes")
    print(f"  Compression ratio: {data_size / compressed_size:.2f}x")
    print(f"  Time elapsed: {elapse:.2f} seconds")
    print(f"  Time per iter: {time_per_iter * 1e3:.1f} ms")
    print(f"  Time per number: {time_per_num * 1e9:.1f} ns")
    print()


LENGTH = 100000
MAX_VALUE = 1000
def test_uniform(length=LENGTH):
    print("Testing uniform distribution")
    print(f"  Length: {length}")
    print(f"  Max value: {MAX_VALUE}")
    numbers = torch.randint(-MAX_VALUE, MAX_VALUE, (LENGTH,), dtype=torch.int32)
    test_eg(numbers)


def test_gradient_dist(length=LENGTH):
    print("Testing distribution of gradients during training")
    print(f"  Loading from gradients.pt")
    print(f"  Length (truncated): {length}")

    data = torch.load("gradients.pt")
    data = data[:length]

    # Quantize
    data = (data * 1000).to(torch.int32)

    test_eg(data)


def main():
    test_uniform()
    test_gradient_dist()


if __name__ == "__main__":
    main()
