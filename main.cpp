#include <chrono>

#include "enums.h"
#include "dense.h"
#include "in_data_bitmap_sparse.h"
#include "block_sparse.h"
#include "CSR_sparse.h"

using namespace std;
using namespace Matrix;

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
    
    //InDataBitmapSparse rowMat = InDataBitmapSparse(fileName, ROW_MAJOR);
    //BlockSparse rowMat = BlockSparse(fileName, 16, ROW_MAJOR);
    //CSRSparse rowMat = CSRSparse(fileName, ROW_MAJOR);
    //rowMat.printMatrix();
    Dense rowMat = Dense(fileName, ROW_MAJOR);
    Dense colMat = Dense(fileName, COLUMN_MAJOR);
    
    auto start = chrono::high_resolution_clock::now();
    Dense denseMat = rowMat.dot(colMat);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> elapsed_milliseconds = end - start;
    cout << elapsed_milliseconds.count() << " ms" << std::endl;
    denseMat.printMatrix(1);

    return 0;
}
