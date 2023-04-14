import os
import re

source_dir = "cpp_src/"
output_dir = "cpp_merged/"
output_cpp = output_dir + "merged.cpp"
output_h = output_dir + "merged.h"

h_files = []
cpp_files = []

for filename in os.listdir(source_dir):
    filename = source_dir + filename
    if filename.endswith(".h") and not filename.startswith("main"):
        h_files.append(filename)
    elif filename.endswith(".cpp") and not filename.startswith("main"):
        cpp_files.append(filename)

with open(output_h, 'w') as merged_h:
    for h_file in h_files:
        with open(h_file, 'r') as f:
            for line in f:
                if not re.match(r"\s*#include\s+\"", line):
                    merged_h.write(line)
        merged_h.write('\n')

with open(output_cpp, 'w') as merged_cpp:
    merged_cpp.write('#include "merged.h"\n')
    for cpp_file in cpp_files:
        with open(cpp_file, 'r') as f:
            for line in f:
                if not re.match(r'\s*#include\s+"', line):
                    merged_cpp.write(line)
        merged_cpp.write('\n')
