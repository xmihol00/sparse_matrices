#include <chrono>

#include "enums.h"
#include "dense.h"
#include "in_data_bitmap_sparse.h"
#include "block_sparse.h"
#include "CSR_sparse.h"
#include "models.h"
#include "block4in16_sparse.h"
#include "blockKinN_sparse.h"
#include "activations.h"

using namespace std;
using namespace Matrix;
using namespace Models;

float same(float x)
{
    return x;
}

int main()
{   

#define MUL_TEST 0
#if MUL_TEST
    Dense rowMat = Dense("weights/weights_l0.csv", ROW_MAJOR);
    //Block4in16Sparse rowMat = Block4in16Sparse("weights/weights_l0.csv", ROW_MAJOR);
    //BlockKinNSparse<2, 16, 1024, 1024, ROW_MAJOR> rowMat("weights/weights_l0.csv");
    Dense colMat = Dense("generated_matrices/random_column.csv", COLUMN_MAJOR);
    Dense result(rowMat.getRows(), colMat.getColumns(), COLUMN_MAJOR, (uint16_t)(9 * sizeof(float)));
    Dense bias = Dense(1024, 1, COLUMN_MAJOR);
    rowMat.dotAddActivateRowThreads(colMat, bias, result, same, 8);

    /*auto start = chrono::high_resolution_clock::now();
    rowMat.dot(colMat, result);
    auto end = chrono::high_resolution_clock::now();
    cerr << "Time of multiplication: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;

    start = chrono::high_resolution_clock::now();
    rowMat.dotNEON(colMat, result);
    end = chrono::high_resolution_clock::now();
    cerr << "Time of multiplication: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;

    start = chrono::high_resolution_clock::now();
    rowMat.dotNEONThreads(colMat, result, 8);
    end = chrono::high_resolution_clock::now();
    cerr << "Time of multiplication: " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;*/

    result.printMatrix(7);
#endif

#define NETWORK_TEST 1
#if NETWORK_TEST
    Dense groundTruth("datasets/mnist_y_test.csv", COLUMN_MAJOR);

    /*Dense input("datasets/mnist_X_test_T.csv", COLUMN_MAJOR);
    Mnist32x32_4L model("weights/weights_", "weights/biases_");
    auto start = chrono::high_resolution_clock::now();
    Dense output = model.predictOptimized(input);
    auto end = chrono::high_resolution_clock::now();
    cerr << "Time of prediction (dense): " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    cerr << "Accuracy: " << output.percentageDifference(groundTruth) << endl;

    input = Dense("datasets/mnist_X_test_T.csv", COLUMN_MAJOR);
    Mnist32x32_4L_Threads<8> modelThreads;
    modelThreads.load("weights/weights_", "weights/biases_");
    start = chrono::high_resolution_clock::now();
    output = modelThreads.predictMatrix(input);
    end = chrono::high_resolution_clock::now();
    cerr << "Time of prediction (dense): " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    //Dense results = output.argmax(0);
    cerr << "Accuracy: " << output.percentageDifference(groundTruth) << endl;*/

    Dense input("datasets/mnist_X_test_T.csv", COLUMN_MAJOR);

    Mnist32x32_4L_KinMSparse<4, 16, 8> model4in16Sparse;
    model4in16Sparse.load("weights/weights_", "weights/biases_");
    auto start = chrono::high_resolution_clock::now();
    Dense output4in16Sparse = model4in16Sparse.predictThreadsMatrix(input);
    auto end = chrono::high_resolution_clock::now();
    cerr << "Time of prediction (4 in 16 template): " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    cerr << "Accuracy: " << output4in16Sparse.percentageDifference(groundTruth) << endl;

    Mnist32x32_4L_KinMSparse<3, 16, 8> model3in16Sparse;
    model3in16Sparse.load("weights/weights_", "weights/biases_");
    start = chrono::high_resolution_clock::now();
    Dense output3in16Sparse = model3in16Sparse.predictThreadsMatrix(input);
    end = chrono::high_resolution_clock::now();
    cerr << "Time of prediction (4 in 16 template): " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    cerr << "Accuracy: " << output3in16Sparse.percentageDifference(groundTruth) << endl;
    
    Mnist32x32_4L_KinMSparse<2, 16, 8> model2in16Sparse;
    model2in16Sparse.load("weights/weights_", "weights/biases_");
    start = chrono::high_resolution_clock::now();
    Dense output2in16Sparse = model2in16Sparse.predictThreadsMatrix(input);
    end = chrono::high_resolution_clock::now();
    cerr << "Time of prediction (2 in 16 template): " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    cerr << "Accuracy: " << output2in16Sparse.percentageDifference(groundTruth) << endl;

    /*start = chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++)
    {
        output = model.predictOptimized(input);
    }
    end = chrono::high_resolution_clock::now();
    cerr << "Time of prediction (dense optimized): " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    //results = output.argmax(0);
    //cerr << "Accuracy: " << results.percentageDifference(groundTruth) << endl;

    float *rawPtr = new float[1024];
    Mnist32x32_4L_Threads<4> modelThreads;
    modelThreads.load("weights/weights_", "weights/biases_");
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++)
    {
        modelThreads.predictRawSample(rawPtr);
    }
    end = chrono::high_resolution_clock::now();
    cerr << "Time of prediction (dense threads): " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    //modelThreads.~Mnist32x32_4L_Threads();
    delete[] rawPtr;*/
    
    /*Dense inputSparse("datasets/mnist_X_test_T.csv", COLUMN_MAJOR);
    Mnist32x32_4L_4in16Sparse modelSparse("weights/weights_", "weights/biases_");
    start = chrono::high_resolution_clock::now();
    Dense outputSparse = modelSparse.predict(inputSparse);
    end = chrono::high_resolution_clock::now();
    cerr << "Time of prediction (4 in 16): " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    Dense resultsSparse = outputSparse.argmax(0);
    cerr << "Accuracy: " << resultsSparse.percentageDifference(groundTruth) << endl;
    //modelSparse.~Mnist32x32_4L_4in16Sparse();

    Dense inputSparse2("datasets/mnist_X_test_T.csv", COLUMN_MAJOR);
    Mnist32x32_4L_KinMSparse<4, 16> modelSparse2;
    modelSparse2.load("weights/weights_", "weights/biases_");
    start = chrono::high_resolution_clock::now();
    Dense outputSparse2 = modelSparse2.predict(inputSparse2);
    end = chrono::high_resolution_clock::now();
    cerr << "Time of prediction (4 in 16 template): " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    Dense resultsSparse2 = outputSparse2.argmax(0);
    cerr << "Accuracy: " << resultsSparse2.percentageDifference(groundTruth) << endl;
    //modelSparse2.~Mnist32x32_4L_KinMSparse();*/

    /*Dense inputSparse3("datasets/mnist_X_test_T.csv", COLUMN_MAJOR);
    Mnist32x32_4L_KinMSparse<4, 16> modelSparse3("weights/weights_", "weights/biases_");
    auto start = chrono::high_resolution_clock::now();
    Dense outputSparse3 = modelSparse3.predictThreads(inputSparse3);
    auto end = chrono::high_resolution_clock::now();
    cerr << "Time of prediction (4 in 16 template threads): " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    Dense resultsSparse3 = outputSparse3.argmax(0);
    cerr << "Accuracy: " << resultsSparse3.percentageDifference(groundTruth) << endl;

    Dense inputSparse4("datasets/mnist_X_test_T.csv", COLUMN_MAJOR);
    Mnist32x32_4L_KinMSparse<4, 16> modelSparse4("weights/weights_", "weights/biases_");
    start = chrono::high_resolution_clock::now();
    Dense outputSparse4 = modelSparse4.predictOptimizedThreads<8>(inputSparse4);
    end = chrono::high_resolution_clock::now();
    cerr << "Time of prediction (4 in 16 template optimized threads): " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    Dense resultsSparse4 = outputSparse4.argmax(0);
    cerr << "Accuracy: " << resultsSparse4.percentageDifference(groundTruth) << endl;*/
 
#endif

    return 0;
}
