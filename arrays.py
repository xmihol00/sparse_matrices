import numpy as np

ROWS = 100
COLUMNS = 100
sparsity = 0.25
array = np.random.rand(ROWS, COLUMNS)
array *= np.random.rand(ROWS, COLUMNS) > sparsity
array = array.astype(dtype=np.float32)
np.savetxt(f"{sparsity}_saparse.csv", array, delimiter=',', fmt="%.7f")
np.savetxt(f"squared_{sparsity}_saparse.csv", array.dot(array), delimiter=',', fmt="%.1f")
