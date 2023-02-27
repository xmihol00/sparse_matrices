import numpy as np

ROWS = 1000
COLUMNS = 1000
sparsity = 0.95
array = np.random.rand(ROWS, COLUMNS)
array *= np.random.rand(ROWS, COLUMNS) > sparsity
array = array.astype(dtype=np.float32)
np.savetxt(f"{sparsity}_saparse.csv", array, delimiter=',', fmt="%.7f")
np.savetxt(f"squared_{sparsity}_saparse.csv", array.dot(array), delimiter=',', fmt="%.1f")
