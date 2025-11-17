"""
Script to analyze gradient distribution.
"""

import torch

grad = torch.load("../dbs/gradients.pt")
sign = torch.sign(grad)


sparsity = (grad == 0).sum().item() / grad.numel()
print(f"Sparsity: {sparsity:.4f}")


def find_runs(tensor, value):
    start = None
    runs = []
    for i in range(len(tensor)):
        if start is not None and tensor[i] != value:
            runs.append((start, i - 1))
            start = None
        if start is None and tensor[i] == value:
            start = i
    if start is not None:
        runs.append((start, len(tensor) - 1))
    return runs

def total_run_length(runs):
    length = 0
    for start, end in runs:
        length += end - start + 1
    return length

for value in (0, 1, -1):
    runs = find_runs(sign, value)
    total_len = total_run_length(runs)
    avg_len = total_len / len(runs) if runs else 0
    print(f"value={value}: num_runs={len(runs)}, total_length={total_len}, avg_length={avg_len:.2f}")
