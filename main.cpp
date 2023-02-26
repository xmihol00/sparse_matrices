#include "enums.h"
#include "dense.h"
#include "in_data_bitmap_sparse.h"
#include "block_sparse.h"
#include "CSR_sparse.h"

using namespace std;
using namespace Matrix;

const char DEFAULT_FILENAME[] = "0.5_saparse.csv";

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
    
    //InDataBitmapSparse rowMat = InDataBitmapSparse("0.5_saparse.csv", ROW_MAJOR);
    //BlockSparse rowMat = BlockSparse("0.5_saparse.csv", 32, ROW_MAJOR);
    CSRSparse rowMat = CSRSparse(fileName, ROW_MAJOR);
    rowMat.printMatrix();
    //Dense rowMat = Dense("0.5_saparse.csv", ROW_MAJOR);
    //Dense colMat = Dense("0.5_saparse.csv", COLUMN_MAJOR);
    //Dense denseMat = rowMat.dotGPU(colMat);
    //denseMat.printMatrix(1);

    //rowMat.printSize();
    //colMat.printSize();

    return 0;
}
