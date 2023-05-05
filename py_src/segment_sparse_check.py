import numpy as np

array = np.loadtxt(f"weights/weights_l0.csv", delimiter=',')[0, :]

reshaped = np.zeros((32, 32))
for i in range(32):
    reshaped[i, :] = array[i::32]

np.savetxt(f"generated_matrices/skip_rows_saparse.csv", reshaped, delimiter=',', fmt="% .7f")
