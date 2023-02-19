import numpy as np

array = np.loadtxt("square_0.5_saparse.csv", delimiter=',')
np.savetxt(f"squared.csv", array.dot(array), delimiter=',', fmt="%.2f")
