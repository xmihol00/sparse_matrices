import glob
import numpy as np

weight_files = sorted(glob.glob("weights/weights_l*.csv"))
weight_files.pop()

patterns = set()

more_than_4 = 0
total_blocks = 1024 * 1024 * 4 // 16

for file in weight_files:
    weights = np.loadtxt(file, delimiter=",")
    for i in range(weights.shape[0] // 16):
        for j in range(weights.shape[1]):
            block = weights[i * 16 : (i + 1) * 16, j]
            non_zero_indices = np.nonzero(block)[0]
            pattern = 0
            for index in non_zero_indices:
                pattern |= 1 << index
            patterns.add(pattern)
            if len(non_zero_indices) > 4:
                more_than_4 += 1
            

print(patterns, len(patterns), more_than_4, total_blocks)
