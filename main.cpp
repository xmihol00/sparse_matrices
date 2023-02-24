#include "enums.h"
#include "dense.h"
#include "in_data_bitmap_sparse.h"
#include "block_sparse.h"

using namespace std;
using namespace Matrix;

int main()
{   
    //InDataBitmapSparse rowMat = InDataBitmapSparse("square_0.5_saparse.csv", ROW_MAJOR);
    //Dense colMat = Dense("square_0.5_saparse.csv", COLUMN_MAJOR);
    //Dense denseMat = rowMat.dot(colMat);
    //denseMat.printMatrix(2);
    //BlockSparse rowMat = BlockSparse("square_0.5_saparse.csv", 4, ROW_MAJOR);
    Dense rowMat = Dense("square_0.5_saparse.csv", ROW_MAJOR);
    Dense colMat = Dense("square_0.5_saparse.csv", COLUMN_MAJOR);
    Dense denseMat = rowMat.dotGPU(colMat);
    denseMat.printMatrix(2);

    return 0;
}
