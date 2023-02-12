import numpy as np

sparsity = 0.5
array = np.random.rand(100, 100)
array *= np.random.rand(100, 100) > sparsity
np.savetxt(f"sprase_{int(sparsity * 100)}_percent.csv", array, delimiter=',', fmt="%.7f")
print(array)
