import numpy as np

ROWS = 1024
COLUMNS = 1024

array1 = np.loadtxt(f"weights/weights_l0.csv", delimiter=',')
array2 = np.random.rand(ROWS, COLUMNS)
np.savetxt(f"generated_matrices/random.csv", array2, delimiter=',', fmt="%.7f")
np.savetxt(f"generated_matrices/reference.csv", np.matmul(array1, array2), delimiter=',', fmt="% .7f")

SPARSITY = 13/16
ENTRIES = ROWS * COLUMNS
REMOVED_ENTRIES = int(ENTRIES * SPARSITY)

indices = np.random.choice(ENTRIES, REMOVED_ENTRIES, replace=False)
array3 = np.random.rand(ROWS, COLUMNS)
array3[indices // COLUMNS, indices % COLUMNS] = 0
np.savetxt(f"generated_matrices/random_{SPARSITY}_sparse.csv", array3, delimiter=',', fmt="%.7f")
np.savetxt(f"generated_matrices/reference_{SPARSITY}_sparse.csv", np.matmul(array3, array2), delimiter=',', fmt="% .7f")
