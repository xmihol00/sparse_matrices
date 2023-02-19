import numpy as np

sparsity = 0.5
array = np.random.rand(100, 100)
array *= np.random.rand(100, 100) > sparsity
np.savetxt(f"square_{sparsity}_saparse.csv", array, delimiter=',', fmt="%.7f")

A1 = np.random.rand(2, 4)
A2 = np.random.rand(4, 2)
print(A1.dot(A2))
print(A2.T.dot(A1.T).T)
