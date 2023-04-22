import csv
import sys
import argparse

def read_csv(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            matrix.append([float(x) for x in row])
    return matrix

def compare_matrices(matrix1, matrix2, epsilon):
    different_cells = []
    for i in range(len(matrix1)):
        for j in range(len(matrix1[i])):
            if abs(matrix1[i][j] - matrix2[i][j]) > epsilon:
                different_cells.append((i, j, matrix1[i][j], matrix2[i][j]))
    return different_cells

def main(file1, file2, epsilon, print_differences=False):
    matrix1 = read_csv(file1)
    matrix2 = read_csv(file2)

    different_cells = compare_matrices(matrix1, matrix2, epsilon)
    
    if different_cells:
        if print_differences:
            for i, j, value1, value2 in different_cells:
                print(f"({i}, {j}) : {value1} : {value2}")
        print(f"Found {len(different_cells)} differences with epsilon =", epsilon)
    else:
        print("No differences found with epsilon =", epsilon)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two matrices in CSV format")
    parser.add_argument("file1", help="Path to the first CSV file")
    parser.add_argument("file2", help="Path to the second CSV file")
    parser.add_argument("-e", "--epsilon", type=float, default=0.01, help="Tolerance value for comparing elements")
    parser.add_argument("-p", "--print", type=bool, default=False, help="Print the different elements")

    args = parser.parse_args()

    main(args.file1, args.file2, args.epsilon, args.print)
