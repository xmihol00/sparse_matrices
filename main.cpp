#include "in_data_bitmap_sparse.h"
#include "dense.h"
#include "enums.h"

using namespace std;
using namespace Matrix;

int main(int argc, char *argv[])
{   
    InDataBitmapSparse rowMat = InDataBitmapSparse("square_0.5_saparse.csv", ROW_MAJOR);
    Dense colMat = Dense("square_0.5_saparse.csv", COLUMN_MAJOR);

    Dense denseMat = rowMat.dot(colMat);
    denseMat.printMatrix(2);

    return 0;
}
