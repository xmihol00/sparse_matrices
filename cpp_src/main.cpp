#include <chrono>

#include "enums.h"
#include "dense.h"
#include "bitmap_sparse.h"
#include "block_sparse.h"
#include "CSR_sparse.h"
#include "models.h"
#include "block4in16_sparse.h"
#include "blockKinN_sparse.h"
#include "activations.h"

using namespace std;
using namespace Matrix;
using namespace Models;

constexpr bool BLOCK_SPARSE = true;
constexpr bool BITMAP_SPARSE = true;
constexpr bool KINN_SPARSE = true;

int main()
{   
    Dense colMat = Dense("generated_matrices/random.csv", COLUMN_MAJOR);

    if (BLOCK_SPARSE)
    {
        BlockSparse rowMat05 = BlockSparse("generated_matrices/random_0.5_sparse.csv", 10, ROW_MAJOR);
        BlockSparse rowMat075 = BlockSparse("generated_matrices/random_0.75_sparse.csv", 10, ROW_MAJOR);
        BlockSparse rowMat08125 = BlockSparse("generated_matrices/random_0.8125_sparse.csv", 10, ROW_MAJOR);
        BlockSparse rowMat0875 = BlockSparse("generated_matrices/random_0.875_sparse.csv", 10, ROW_MAJOR);
        
        auto start = chrono::high_resolution_clock::now();
        Dense result = rowMat05.dot(colMat);
        auto end = chrono::high_resolution_clock::now();
        cout << "Block Sparse 0.5 sparsity: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
        
        start = chrono::high_resolution_clock::now();
        result = rowMat075.dot(colMat);
        end = chrono::high_resolution_clock::now();
        cout << "Block Sparse 0.75 sparsity: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;

        start = chrono::high_resolution_clock::now();
        result = rowMat08125.dot(colMat);
        end = chrono::high_resolution_clock::now();
        cout << "Block Sparse 0.8125 sparsity: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;

        start = chrono::high_resolution_clock::now();
        result = rowMat0875.dot(colMat);
        end = chrono::high_resolution_clock::now();
        cout << "Block Sparse 0.875 sparsity: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    }

    if (BITMAP_SPARSE)
    {
        BitmapSparse rowMat05 = BitmapSparse("generated_matrices/random_0.5_sparse.csv", ROW_MAJOR);
        BitmapSparse rowMat075 = BitmapSparse("generated_matrices/random_0.75_sparse.csv", ROW_MAJOR);
        BitmapSparse rowMat08125 = BitmapSparse("generated_matrices/random_0.8125_sparse.csv", ROW_MAJOR);
        BitmapSparse rowMat0875 = BitmapSparse("generated_matrices/random_0.875_sparse.csv", ROW_MAJOR);

        auto start = chrono::high_resolution_clock::now();
        Dense result = rowMat05.dot(colMat);
        auto end = chrono::high_resolution_clock::now();
        cout << "Bitmap Sparse 0.5 sparsity: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;

        start = chrono::high_resolution_clock::now();
        result = rowMat075.dot(colMat);
        end = chrono::high_resolution_clock::now();
        cout << "Bitmap Sparse 0.75 sparsity: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;

        start = chrono::high_resolution_clock::now();
        result = rowMat08125.dot(colMat);
        end = chrono::high_resolution_clock::now();
        cout << "Bitmap Sparse 0.8125 sparsity: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;

        start = chrono::high_resolution_clock::now();
        result = rowMat0875.dot(colMat);
        end = chrono::high_resolution_clock::now();
        cout << "Bitmap Sparse 0.875 sparsity: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    }

    if (KINN_SPARSE)
    {
        BlockKinNSparse<8, 16, 1024, 1024, ROW_MAJOR> rowMat05("weights/weights_l0.csv", false);
        BlockKinNSparse<4, 16, 1024, 1024, ROW_MAJOR> rowMat075("weights/weights_l0.csv", false);
        BlockKinNSparse<3, 16, 1024, 1024, ROW_MAJOR> rowMat08125("weights/weights_l0.csv", false);
        BlockKinNSparse<2, 16, 1024, 1024, ROW_MAJOR> rowMat0875("weights/weights_l0.csv", false);

        auto start = chrono::high_resolution_clock::now();
        Dense result = rowMat05.dot(colMat);
        auto end = chrono::high_resolution_clock::now();
        cout << "K in N Sparse 0.5 sparsity: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;

        start = chrono::high_resolution_clock::now();
        result = rowMat075.dot(colMat);
        end = chrono::high_resolution_clock::now();
        cout << "K in N Sparse 0.75 sparsity: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;

        start = chrono::high_resolution_clock::now();
        result = rowMat08125.dot(colMat);
        end = chrono::high_resolution_clock::now();
        cout << "K in N Sparse 0.8125 sparsity: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;

        start = chrono::high_resolution_clock::now();
        result = rowMat0875.dot(colMat);
        end = chrono::high_resolution_clock::now();
        cout << "K in N Sparse 0.875 sparsity: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    }

#define NETWORK_TEST 0
#if NETWORK_TEST
    Dense groundTruth("datasets/mnist_y_test.csv", COLUMN_MAJOR);

    bool metadataFirst = true;
    Dense input("datasets/mnist_X_test_T.csv", COLUMN_MAJOR);

    Mnist32x32_4L_KinMSparse<4, 16, 8> model4in16Sparse;
    model4in16Sparse.load("weights/weights_", "weights/biases_", metadataFirst);
    auto start = chrono::high_resolution_clock::now();
    Dense output4in16Sparse = model4in16Sparse.predictThreadsMatrix(input);
    auto end = chrono::high_resolution_clock::now();
    cerr << "Time of prediction (4 in 16 template): " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    cerr << "Accuracy: " << output4in16Sparse.percentageDifference(groundTruth) << endl;

    Mnist32x32_4L_KinMSparse<3, 16, 8> model3in16Sparse;
    model3in16Sparse.load("weights/weights_", "weights/biases_", metadataFirst);
    start = chrono::high_resolution_clock::now();
    Dense output3in16Sparse = model3in16Sparse.predictThreadsMatrix(input);
    end = chrono::high_resolution_clock::now();
    cerr << "Time of prediction (3 in 16 template): " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    cerr << "Accuracy: " << output3in16Sparse.percentageDifference(groundTruth) << endl;
    
    Mnist32x32_4L_KinMSparse<2, 16, 8> model2in16Sparse;
    model2in16Sparse.load("weights/weights_", "weights/biases_", metadataFirst);
    start = chrono::high_resolution_clock::now();
    Dense output2in16Sparse = model2in16Sparse.predictThreadsMatrix(input);
    end = chrono::high_resolution_clock::now();
    cerr << "Time of prediction (2 in 16 template): " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    cerr << "Accuracy: " << output2in16Sparse.percentageDifference(groundTruth) << endl;
#endif
    return 0;
}
