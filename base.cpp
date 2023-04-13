#include "base.h"
#include "dense.h"

using namespace std;
using namespace Matrix;

Base::Base(DimensionMajorityEnum dimMajority) : _dimMajority{dimMajority} { }

Base::Base(uint16_t rows, uint16_t columns, DimensionMajorityEnum dimMajority) : 
    _rows{rows}, _columns{columns}, _dimMajority{dimMajority}
{
    _size = rows * columns * sizeof(float);
}

Base::~Base()
{
    delete[] _byteMatrix;
    _byteMatrix = nullptr;
}

void Base::printSize()
{
    cout << "size: " << _size << " B" << endl;
}

void Base::printRow(uint16_t rowIndex, uint8_t precision)
{
    (void)precision;

    if (rowIndex >= _rows)
    {
        throw invalid_argument((stringstream() << "Row index out of range [0, " << _rows - 1 << "].").str());
    }
}

void Base::printColumn(uint16_t columnIndex, uint8_t precision)
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
