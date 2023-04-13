import numpy as np
import matplotlib.pyplot as plt

sparsity = np.array([0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9])
sizes = np.loadtxt("sizes.txt").reshape(-1, 3)

sizes[:, 1] /= sizes[:, 0]
sizes[:, 2] /= sizes[:, 0]
sizes[:, 0] /= sizes[:, 0]

plt.plot(sparsity, sizes[:, 0], marker='o', label="Dense")
plt.plot(sparsity, sizes[:, 1], marker='o', label="Block")
plt.plot(sparsity, sizes[:, 2], marker='o', label="CSR")
plt.legend()
plt.xticks(sparsity, sparsity.astype(np.str_))
plt.xlabel("sparsity")
plt.ylabel("relative matrix size")
plt.savefig("sizes_comparison", dpi=500)
plt.show()
