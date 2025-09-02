import matplotlib.pyplot as plt
import numpy as np
import torch

data = torch.load("params.pt")

plt.hist(data.numpy(), bins=100, alpha=0.5, color='orange', range=(-0.007, 0.007))
plt.title("Parameter Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# Plot proportion of values where abs(v) < x
x_values = np.linspace(0, 0.01, 30)
proportions = []

for x in x_values:
    proportion = (data.abs() < x).float().mean().item()
    proportions.append(proportion)

plt.plot(x_values, proportions, marker='o')
plt.title("Proportion of Parameters with |value| < x")
plt.xlabel("x")
plt.ylabel("Proportion")
plt.grid()
plt.tight_layout()
plt.show()


# Print proportion of zeros
proportion_zeros = (data == 0).float().mean().item()
print(f"Proportion of zeros: {proportion_zeros:.4f}")


# Print average zero run length
lengths = []
current_length = 0
for v in data[:500000]:
    if v == 0:
        current_length += 1
    else:
        if current_length > 0:
            lengths.append(current_length)
            current_length = 0
if current_length > 0:
    lengths.append(current_length)

print(f"Average zero run length: {np.mean(lengths):.2f}")
