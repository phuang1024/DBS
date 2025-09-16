import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 63, 1)
y = np.zeros_like(x)

for i in range(len(x)):
    v = x[i] + 1
    bits = bin(v)[2:]
    y[i] = len(bits) * 2 - 1

plt.scatter(x, y)
plt.xlabel("Value")
plt.ylabel("Number of bits")
plt.show()
