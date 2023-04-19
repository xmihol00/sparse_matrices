#include <chrono>

#include "enums.h"
#include "dense.h"
#include "in_data_bitmap_sparse.h"
#include "block_sparse.h"
#include "CSR_sparse.h"
#include "models.h"
#include "block4in16_sparse.h"

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

    Block4in16Sparse rowMat = Block4in16Sparse("weights/weights_l0.csv", ROW_MAJOR);
    rowMat.printMatrix();

    //Mnist32x32_4L model("weights/weights_", "weights/biases_");
    //Dense input("datasets/mnist_X_test_T.csv", COLUMN_MAJOR);
    //Dense output = model.predict(input);
    //Dense results = output.argmax(0);
    //results.printMatrix(1);

    //InDataBitmapSparse rowMat = InDataBitmapSparse(fileName, ROW_MAJOR);
    //BlockSparse rowMat = BlockSparse(fileName, 16, ROW_MAJOR);
    //CSRSparse rowMat = CSRSparse(fileName, ROW_MAJOR);
    //rowMat.printMatrix();
    //Dense rowMat = Dense(fileName, ROW_MAJOR);
    //Dense colMat = Dense(fileName, COLUMN_MAJOR);
    //
    //auto start = chrono::high_resolution_clock::now();
    //Dense denseMat = rowMat.dot(colMat);
    //auto end = chrono::high_resolution_clock::now();
//
    //chrono::duration<double, milli> elapsed_milliseconds = end - start;
    //cout << elapsed_milliseconds.count() << " ms" << std::endl;
    //denseMat.printMatrix(1);

    return 0;
}
