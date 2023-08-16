import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def gaussian_distribution(x, y, sigma=1):
    return np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

x = np.linspace(-3, 3, 9)
y = np.linspace(-3, 3, 9)
X, Y = np.meshgrid(x, y)
Z = gaussian_distribution(X, Y)

# plot only Z in 2d
plt.imshow(Z, cmap='viridis')
plt.colorbar()
#write values in pixels
for i in range(9):
    for j in range(9):
        plt.text(i, j, round(Z[i, j], 2), ha="center", va="center", color="w")

# make axis be -4 to 4
plt.xticks(np.arange(9), np.arange(-4, 5))
plt.yticks(np.arange(9), np.arange(-4, 5))

# put frame around central 3 pixels
plt.gca().add_patch(plt.Rectangle((2.49, 2.49), 3, 3, fill=False, color='black', linewidth=2))

plt.show()


