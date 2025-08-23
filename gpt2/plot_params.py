"""
Plot parameter distributions.
"""

import matplotlib.pyplot as plt
import torch


yes_bs = torch.load("yes_backslash_params.pt")
no_bs = torch.load("no_backslash_params.pt")

plt.hist(no_bs.numpy(), bins=500, alpha=0.5, label="No Backslash", color='orange')
plt.hist(yes_bs.numpy(), bins=500, alpha=0.5, label="Yes Backslash", color='blue')

plt.title("Parameter Distribution")
plt.legend()
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.xlim(-0.1, 0.1)

plt.grid()

plt.tight_layout()
plt.show()
