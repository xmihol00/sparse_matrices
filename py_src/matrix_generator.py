import numpy as np

ROWS = 1024
COLUMNS = 1024
sparsity = 0.75
#array = np.random.rand(ROWS, COLUMNS)
#array *= np.random.rand(ROWS, COLUMNS) > sparsity
#array = array.astype(dtype=np.float32)
#np.savetxt(f"generated_matrices/{sparsity}_saparse.csv", array, delimiter=',', fmt="%.7f")
array1 = np.loadtxt(f"weights/weights_l0.csv", delimiter=',')
array2 = np.random.rand(ROWS, COLUMNS)
np.savetxt(f"generated_matrices/random.csv", array2, delimiter=',', fmt="%.7f")
np.savetxt(f"generated_matrices/reference.csv", np.matmul(array1, array2), delimiter=',', fmt="% .7f")
