"""
Analysis of encoding run times.
"""

import matplotlib.pyplot as plt


# Pairs of (run_time, data_transfer)
# Data transfer in MB, run time in seconds.
data = [
    (66, 48.5),
    (62.6, 36.7),
    #(84.8, 94.2),
    (67.9, 57),
    (70, 49.4),
    (61.2, 44.2),
    (69.4, 39.5),
    (69.7, 36.8),
    (71.1, 34.9),
]


x = [d[0] for d in data]
y = [d[1] for d in data]

plt.scatter(x, y)
plt.xlabel("Data Transfer (MB)")
plt.ylabel("Run Time (s)")
plt.show()
