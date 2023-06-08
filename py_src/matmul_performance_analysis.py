import matplotlib.pyplot as plt

block_sparse = [ 728.576, 517.008, 309.861, 253.621, 208.577 ]
bitmap_sparse = [ 1682.46, 1243.84, 795.228, 680.767, 567.75 ]
KinN_sparse = [ 1282.62, 636.783, 204.891, 129.125, 81.4703 ]

# plot results for samples
plt.figure(figsize=(10, 5))
plt.xlabel("Sparsity [percentage]")
plt.ylabel("Latency [ms]")
# increase the marker size
plt.rcParams['lines.markersize'] = 10
plt.plot([1, 2, 3, 4, 5], block_sparse, label="Block Sparse", marker="o")
plt.plot([1, 2, 3, 4, 5], bitmap_sparse, label="Bitmap Sparse", marker="o")
plt.plot([1, 2, 3, 4, 5], KinN_sparse, label="KinN Sparse", marker="o")
# rename values on x axis
plt.xticks([1, 2, 3, 4, 5], [0.25, 0.5, 0.75, 0.8125, 0.875])
plt.legend()
plt.tight_layout()
plt.savefig(f"plots/matmul_performance.png", dpi=500)
plt.show()
