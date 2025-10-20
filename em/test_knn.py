import matplotlib.pyplot as plt
import numpy as np
import torch

data = torch.load("gradients.pt")
data = data.numpy()
print(data.shape)

#plt.figure()
#plt.hist(data, bins=100, range=(-0.01, 0.01))
#plt.show()


# KNN
k = 5
centers = np.random.uniform(-0.01, 0.01, size=(5,))

steps = 100
for i in range(steps):
    # Classify each point
    distances = np.empty((data.shape[0], k))
    for cls in range(k):
        distances[:, cls] = np.abs(data - centers[cls])
    labels = np.argmin(distances, axis=1)

    # Update centers
    new_centers = np.zeros_like(centers)
    for cls in range(k):
        if np.sum(labels == cls) > 0:
            new_centers[cls] = np.mean(data[labels == cls])
        else:
            new_centers[cls] = centers[cls]

    centers = new_centers

    print("Step", i, "centers:", centers)

plt.figure()
plt.hist(data, bins=100, range=(-0.01, 0.01))
for cls in range(k):
    plt.axvline(centers[cls], color=f"C{cls}", linestyle="--")
plt.xlim(-0.01, 0.01)
plt.show()
