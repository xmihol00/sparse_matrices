import time
import numpy as np
import math

ROWS = 1024
COLUMNS = 1024
sparsity = 0.8
array = np.random.rand(1, COLUMNS)
array *= np.random.rand(1, COLUMNS) > sparsity
array = array.astype(dtype=np.float32)

# reshape so that row 1 has elements from column 1, column 16, column 32, etc.
# row 2 has elements from column 2, column 17, column 33, etc.

#reshaped = np.zeros((32, 32))
#for i in range(32):
#    reshaped[i, :] = array[0, i::32]
#
#np.savetxt(f"generated_matrices/skip_rows_saparse.csv", reshaped, delimiter=',', fmt="%.7f")
#np.savetxt(f"generated_matrices/skip_rows_dense.csv", array, delimiter=',', fmt="%.7f")

array1 = np.loadtxt(f"weights/weights_l0.csv", delimiter=',')
array2 = np.random.rand(ROWS, COLUMNS)
np.savetxt(f"generated_matrices/random.csv", array2, delimiter=',', fmt="%.7f")
np.savetxt(f"generated_matrices/random_column.csv", array2[:, 0], delimiter=',', fmt="%.7f")
np.savetxt(f"generated_matrices/reference.csv", np.matmul(array1, array2), delimiter=',', fmt="% .7f")
np.savetxt(f"generated_matrices/reference_column.csv", np.matmul(array1, array2[:, 0]), delimiter=',', fmt="% .7f")

