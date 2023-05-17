import numpy as np

weights = np.loadtxt(f"weights/weights_l0.csv", delimiter=',')
random = np.loadtxt(f"generated_matrices/random.csv", delimiter=',')

first_row = weights[0, :]
second_row = weights[1, :]
random_row = random[0, :]

reshaped_first_row = np.zeros((32, 32))
reshaped_second_row = np.zeros((32, 32))
reshaped_random_row = np.zeros((32, 32))
for i in range(32):
    reshaped_first_row[i, :] = first_row[i::32]
    reshaped_second_row[i, :] = second_row[i::32]
    reshaped_random_row[i, :] = random_row[i::32]

first_nonzero_rows = np.any(reshaped_first_row != 0, axis=1)
second_nonzero_rows = np.any(reshaped_second_row != 0, axis=1)

np.savetxt(f"generated_matrices/skip_rows_dense1.csv", reshaped_first_row, delimiter=',', fmt="% .7f")
np.savetxt(f"generated_matrices/skip_rows_sparse1.csv", reshaped_first_row[first_nonzero_rows], delimiter=',', fmt="% .7f")
np.savetxt(f"generated_matrices/skip_rows_metadata1.csv", np.where(first_nonzero_rows == 1)[0], delimiter=',', fmt="%d")
np.savetxt(f"generated_matrices/skip_rows_dense2.csv", reshaped_second_row, delimiter=',', fmt="% .7f")
np.savetxt(f"generated_matrices/skip_rows_sparse2.csv", reshaped_second_row[second_nonzero_rows], delimiter=',', fmt="% .7f")
np.savetxt(f"generated_matrices/skip_rows_metadata2.csv", np.where(second_nonzero_rows == 1)[0], delimiter=',', fmt="%d")

print("Dense dot product row 1:", np.dot(first_row, random_row))
print("Sparse dot product row 1:", np.dot(reshaped_first_row[first_nonzero_rows].flatten(), reshaped_random_row[first_nonzero_rows].flatten()))
print("Dense dot product row 2:", np.dot(second_row, random_row))
print("Sparse dot product row 2:", np.dot(reshaped_second_row[second_nonzero_rows].flatten(), reshaped_random_row[second_nonzero_rows].flatten()))
