#include "sparse.h"

using namespace std;
using namespace Matrix;

Sparse::Sparse(DimenstionMajorityEnum dimMajority, uint16_t blocksPerDimension) 
    : _blocksPerDimension{blocksPerDimension}, Base(dimMajority) {}

void Sparse::loadCSV(string fileName)
{
    ifstream file(fileName);
    if (file.is_open()) 
    {

        if (_dimMajority == ROW_MAJOR)
        {
            allocateSpaceRowMajorCSV(file);
            
            file.clear();
            file.seekg(0);

            loadDataRowMajorCSV(file);
        }
        else if (_dimMajority == COLUMN_MAJOR)
        {
            allocateSpaceColumnMajorCSV(file);
            
            file.clear();
            file.seekg(0);

            loadDataColumnMajorCSV(file);
        }
        else
        {
            throw invalid_argument(_UNSUPPORTED_MAJORITY);
        }

        file.close();
    }
}
