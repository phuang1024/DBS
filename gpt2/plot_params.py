"""
Plot parameter distributions.
"""

import matplotlib.pyplot as plt
import torch


yes_bs = torch.load("yes_backslash_params.pt")
no_bs = torch.load("no_backslash_params.pt")

bins = 100
x_range = (-0.1, 0.1)

plt.hist(no_bs.numpy(), bins=bins, alpha=0.5, label="No Backslash", color='orange', range=x_range)
plt.hist(yes_bs.numpy(), bins=bins, alpha=0.5, label="Yes Backslash", color='blue', range=x_range)

plt.title("Parameter Distribution")
plt.legend()
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.xlim(-0.1, 0.1)

plt.grid()

plt.tight_layout()
plt.show()
