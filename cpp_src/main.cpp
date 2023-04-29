#include <chrono>

#include "enums.h"
#include "dense.h"
#include "in_data_bitmap_sparse.h"
#include "block_sparse.h"
#include "CSR_sparse.h"
#include "models.h"
#include "block4in16_sparse.h"
#include "blockKinN_sparse.h"

using namespace std;
using namespace Matrix;
using namespace Models;

const char DEFAULT_FILENAME[] = "generated_matrices/0.75_saparse.csv";

int main(int argc, char *argv[])
{   
    const char *fileName;
    if (argc > 1)
    {
        fileName = argv[1];        
    }
    else
    {
        fileName = DEFAULT_FILENAME;
    }

#define MUL_TEST 0
#if MUL_TEST
    //Block4in16Sparse rowMat = Block4in16Sparse("weights/weights_l0.csv", ROW_MAJOR);
    //Dense rowMat = Dense("weights/weights_l0.csv", ROW_MAJOR);
    BlockKinNSparse<4, 16, 1024, 1024, ROW_MAJOR> rowMat("weights/weights_l0.csv");
    Dense colMat = Dense("generated_matrices/random.csv", COLUMN_MAJOR);
    Dense result(rowMat.getRows(), colMat.getColumns(), COLUMN_MAJOR, 9 * sizeof(float));
    
    // measure time of multiplication
    auto start = chrono::high_resolution_clock::now();
    rowMat.dot(colMat, result);
    auto end = chrono::high_resolution_clock::now();
    cerr << "Time of multiplication: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    result.printMatrix(7);
#endif

#define NETWORK_TEST 1
#if NETWORK_TEST
    Dense groundTruth("datasets/mnist_y_test.csv", COLUMN_MAJOR);

    Dense input("datasets/mnist_X_test_T.csv", COLUMN_MAJOR);
    Mnist32x32_4L model("weights/weights_", "weights/biases_");
    auto start = chrono::high_resolution_clock::now();
    Dense output = model.predict(input);
    auto end = chrono::high_resolution_clock::now();
    cerr << "Time of prediction: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    Dense results = output.argmax(0);
    cerr << "Accuracy: " << results.percentageDifference(groundTruth) << endl;
    
    Dense inputSparse("datasets/mnist_X_test_T.csv", COLUMN_MAJOR);
    Mnist32x32_4L_4in16Sparse modelSparse("weights/weights_", "weights/biases_");
    start = chrono::high_resolution_clock::now();
    Dense outputSparse = modelSparse.predict(inputSparse);
    end = chrono::high_resolution_clock::now();
    cerr << "Time of prediction: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    Dense resultsSparse = outputSparse.argmax(0);
    cerr << "Accuracy: " << resultsSparse.percentageDifference(groundTruth) << endl;

    Dense inputSparse2("datasets/mnist_X_test_T.csv", COLUMN_MAJOR);
    Mnist32x32_4L_KinMSparse<4, 16> modelSparse2("weights/weights_", "weights/biases_");
    start = chrono::high_resolution_clock::now();
    Dense outputSparse2 = modelSparse2.predict(inputSparse2);
    end = chrono::high_resolution_clock::now();
    cerr << "Time of prediction: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    Dense resultsSparse2 = outputSparse2.argmax(0);
    cerr << "Accuracy: " << resultsSparse2.percentageDifference(groundTruth) << endl;
#endif

    return 0;
}
