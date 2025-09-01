import time

import torch

import python

torch.manual_seed(0)

TEST_ITERS = 100

NUM_NUMS = 10000
nums = torch.randint(0, 100, (NUM_NUMS,))


def test_functions(encode, decode, iters=TEST_ITERS):
    time_start = time.time()
    for _ in range(iters):
        data = encode(nums.tolist())
        nums_dec = decode(data)
    elapse = time.time() - time_start
    elapse /= iters

    equal = torch.equal(nums, nums_dec)

    print(f"  Elapse: {elapse:.2f}s")
    print(f"  Input and output match: {equal}")
    print()


def main():
    print("Performance test of Exp-Golomb coding:")
    print(f"  Number of numbers: {NUM_NUMS}")
    print(f"  Number of iterations: {TEST_ITERS}")
    print()

    print("Testing Python implementation:")
    test_functions(python.encode, python.decode)


if __name__ == "__main__":
    main()
