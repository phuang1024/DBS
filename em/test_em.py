import matplotlib.pyplot as plt
import numpy as np
import torch

data = torch.load("gradients.pt")
data = data.numpy()
print(data.shape)

#plt.figure()
#plt.hist(data, bins=100, range=(-0.01, 0.01))
#plt.show()


def cost_fn(centers):
    centers = np.sort(centers)

    # Classify each point
    distances = np.empty((data.shape[0], len(centers)))
    for cls in range(len(centers)):
        distances[:, cls] = np.abs(data - centers[cls])
    labels = np.argmin(distances, axis=1)

    cost = 0

    # Compute cost bits per value.
    bits_per_cls = np.abs(np.arange(len(centers)) - len(centers) // 2) + 1
    for cls in range(len(centers)):
        cost += np.sum(labels == cls) * bits_per_cls[cls]
    cost /= data.shape[0]

    # Compute cost distance
    for cls in range(len(centers) - 1):
        cost += (centers[cls + 1] - centers[cls]) * 5e1

    return cost


# KNN
k = 5
centers = np.random.uniform(-0.01, 0.01, size=(5,))

steps = 100
for step in range(steps):
    centers = np.sort(centers)
    cost = cost_fn(centers)

    # Numerical derivative
    grad = np.zeros_like(centers)
    for i in range(k):
        delta = 1e-6
        centers_pos = centers.copy()
        centers_neg = centers.copy()
        centers_pos[i] += delta
        centers_neg[i] -= delta
        cost_pos = cost_fn(centers_pos)
        cost_neg = cost_fn(centers_neg)
        grad[i] = (cost_pos - cost_neg) / (2 * delta)

    lr = 1e-6
    centers -= lr * grad

    print("Step", step, "centers:", centers, "grads", grad, "cost:", cost)

plt.figure()
plt.hist(data, bins=100, range=(-0.01, 0.01))
for cls in range(k):
    plt.axvline(centers[cls], color=f"C{cls}", linestyle="--")
plt.xlim(-0.01, 0.01)
plt.show()
