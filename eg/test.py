import time

import torch
from torch.utils.cpp_extension import load

# Python implementation
import python

# C++ implementation
cpp = load(
    name="cpp",
    sources=["cpp.cpp"],
    verbose=True,
)

torch.manual_seed(0)

TEST_ITERS = 100

NUM_NUMS = 10000
nums = torch.randint(0, 10000, (NUM_NUMS,), dtype=torch.uint32)
#nums = torch.tensor([7, 8, 9], dtype=torch.uint32)


def test_functions(encode, decode, iters=TEST_ITERS):
    time_start = time.time()
    for i in range(iters):
        data = encode(nums)
        nums_dec = decode(data)
        #if i == 0:
            #print(nums, data, nums_dec, sep="\n")
    elapse = time.time() - time_start
    elapse /= iters

    equal = torch.equal(nums, nums_dec)

    print(f"  Avg time per iter: {elapse*1000:.1f}ms")
    print(f"  Input and output match: {equal}")
    print()


def main():
    print("Performance test of Exp-Golomb coding:")
    print(f"  Number of numbers: {NUM_NUMS}")
    print(f"  Number of iterations: {TEST_ITERS}")
    print()

    print("Testing Python implementation:")
    test_functions(python.encode, python.decode)

    print("Testing C++ implementation:")
    test_functions(cpp.encode, cpp.decode)


if __name__ == "__main__":
    main()
