"""
Show stats of distribution of gradients.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gennorm
from tqdm import tqdm

#data = torch.load("../dbs/gradients.pt").numpy()
with open("../data/grads_bert.npy", "rb") as f:
    data = np.frombuffer(f.read(), dtype=np.float32)

print(f"Data shape: {data.shape}")

max_x = 1e-3

# Calculate GG distribution
gg_shape = 0.15
gg_std = 5.824e-4
x = np.linspace(-max_x, max_x, 100)
pdf = gennorm.pdf(x, gg_shape, scale=gg_std)

hist_bars, _, _ = plt.hist(data, bins=100, alpha=0.5, color='blue', range=(-max_x, max_x * 1.01))
hist_bars = sorted(hist_bars, reverse=True)
pdf = pdf / np.max(pdf) * hist_bars[1]  # Scale PDF to histogram
plt.plot(x, pdf, 'r-', lw=2)
#plt.title("Gradient Distribution")
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.tight_layout()
#plt.show()
plt.savefig("grad_dist.png")
stop


# Plot proportion of values where abs(v) < x
x_values = np.linspace(0, max_x, 30)
proportions = []

for x in x_values:
    proportion = np.sum(np.abs(data) <= x) / data.size
    proportions.append(proportion)

plt.clf()
plt.plot(x_values, proportions, marker='o')
#plt.title("Proportion of Parameters with |value| <= x")
plt.xlabel("x")
plt.ylabel("Proportion")
plt.grid()
plt.tight_layout()
#plt.show()
plt.savefig("proportion_below_x.png")


# Print proportion of zeros
proportion_zeros = np.sum(data == 0) / data.size
print(f"Proportion of zeros: {proportion_zeros:.4f}")


# Print average zero run length
lengths = []
current_length = 0
for v in tqdm(data):
    if v == 0:
        current_length += 1
    else:
        if current_length > 0:
            lengths.append(current_length)
            current_length = 0
if current_length > 0:
    lengths.append(current_length)

print(f"Average zero run length: {np.mean(lengths):.2f}")

# Print stats
print(f"Mean: {data.mean().item():.6f}")
print(f"Std: {data.std().item():.6f}")
