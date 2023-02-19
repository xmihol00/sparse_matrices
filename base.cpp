#include "base.h"
#include "dense.h"

using namespace std;
using namespace Matrix;

Base::Base(DimenstionMajorityEnum dimMajority) : _dimMajority{dimMajority} { }

Base::Base(uint32_t rows, uint32_t columns, DimenstionMajorityEnum dimMajority) : 
    _rows{rows}, _columns{columns}, _dimMajority{dimMajority} 
{
    _size = rows * columns * sizeof(float);
}

void Base::printRow(uint32_t rowIndex, uint8_t precision)
{
    (void)precision;

    if (rowIndex >= _rows)
    {
        throw invalid_argument((stringstream() << "Row index out of range [0, " << _rows - 1 << "].").str());
    }
}

void Base::printColumn(uint32_t columnIndex, uint8_t precision)
{
    (void)precision;

    if (columnIndex >= _columns)
    {
        throw invalid_argument((stringstream() << "Column index out of range [0, " << _columns - 1 << "].").str());
    }
}

void Base::printMatrix(uint8_t precision)
{
    for (uint32_t i = 0; i < _rows; i++)
    {
        printRow(i, precision);
    }
}
