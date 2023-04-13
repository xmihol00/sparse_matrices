import pandas as pd
import numpy as np
import glob

csv_files = sorted(glob.glob("weights/layer_*_weights.csv"))

total_elements = 0
zero_elements = 0

for file in csv_files:
    weights_df = pd.read_csv(file)
    
    weights = weights_df.to_numpy()
    
    print(f"layer: {file} sparsity: {np.count_nonzero(weights == 0) / np.size(weights) * 100:.2f} %")
    total_elements += np.size(weights)
    zero_elements += np.count_nonzero(weights == 0)

sparsity = (zero_elements / total_elements) * 100

print(f"total elements: {total_elements}")
print(f"zero elements: {zero_elements}")
print(f"sparsity: {sparsity:.2f} %")
