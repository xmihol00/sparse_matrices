#include "CSR_sparse.h"
#include "typedefs.h"

using namespace Matrix;
using namespace std;

CSRSparse::CSRSparse(string fileName, DimenstionMajorityEnum dimMajority)
    : Sparse(dimMajority)
{
    if (fileName.ends_with(".csv"))
    {
        if (_dimMajority == FILE_DETERMINED)
        {
            throw invalid_argument("Dimension majority must be specified when loading from CSV file.");
        }
        
        loadCSV(fileName);
    }
    else if (fileName.ends_with(".bms"))
    {
        loadBinary(fileName);
    }
    else
    {
        throw invalid_argument("Unsupported file extension.");
    }
}

void CSRSparse::allocateSpaceRowMajorCSV(std::ifstream &file)
{
    uint32_t totalEntries = 0;
    string row;
    string cell;

    while (getline(file, row)) 
    {
        _rows++;
        stringstream rowStream(row);
        while (getline(rowStream, cell, ','))
        {
            float cellValue = stof(cell);
            if (cellValue != 0.0f)
            {
                _nonZeroEntries++;
            }
            totalEntries++;
        }
    }

    _columns = totalEntries / _rows;
    _size = (_rows + 1) * sizeof(uint32_t) + 
            _nonZeroEntries * sizeof(uint16_t) + // -1 becauase the index of the first entry is determined by the block 
            _nonZeroEntries * sizeof(float);
    _byteMatrix = new byte[_size]();
}

void CSRSparse::allocateSpaceColumnMajorCSV(std::ifstream &file)
{
    (void)file;
}

void CSRSparse::loadDataRowMajorCSV(std::ifstream &file)
{
    // creating pointers for the entries offsets and the data, use of unions to avoid casting
    matrix_ptrs_t tmp;
    tmp.uint32s = &_uint32Matrix[_rows + 1];
    _columnIndices = tmp.uint16s;
    tmp.uint16s = &_columnIndices[_nonZeroEntries];
    _dataMatrix = tmp.floats;

    string row;
    string cell;
    uint16_t rowIndex = 0;
    uint32_t entryIndex = 0;
    while (getline(file, row))
    {
        uint16_t columnIndex = 0;
        float cellValue;
        stringstream rowStream(row);
        while (getline(rowStream, cell, ','))
        {
            cellValue = stof(cell);
            if (cellValue != 0.0f)
            {
                _columnIndices[entryIndex] = columnIndex;
                _dataMatrix[entryIndex++] = cellValue;
            }
            columnIndex++;
        }
        _uint32Matrix[++rowIndex] = entryIndex;
    }
}

void CSRSparse::loadDataColumnMajorCSV(std::ifstream &file)
{
    (void)file;
}


void CSRSparse::loadBinary(std::string fileName)
{
    (void)fileName;
}

void CSRSparse::printColumn(uint16_t columnIndex, uint8_t precision)
{

}

void CSRSparse::printRow(uint16_t rowIndex, uint8_t precision)
{
    Base::printRow(rowIndex);

    cout << setprecision(precision) << fixed;
    if (_dimMajority == ROW_MAJOR)
    {
        float *dataOffset = &_dataMatrix[_uint32Matrix[rowIndex]];
        uint16_t *columnIndicesOffset = &_columnIndices[_uint32Matrix[rowIndex]];
        uint16_t entries = _uint32Matrix[rowIndex + 1] - _uint32Matrix[rowIndex];
        uint16_t columnIndex = 0;
        for (uint16_t i = 0; i < entries; i++)
        {
            for ( ; columnIndex < columnIndicesOffset[i]; columnIndex++)
            {
                cout << 0.0f << ',';    
            }

            if (columnIndex++ != _columns - 1)
            {
                cout << dataOffset[i] << ',';
            }
            else
            {
                cout << dataOffset[i] << endl;
                return;
            }
        }

        for ( ; columnIndex < _columns - 1; columnIndex++)
        {
            cout << 0.0f << ','; 
        }
        cout << 0.0f << endl;
    }
}

void CSRSparse::saveAsBinary(std::string fileName)
{

}

void CSRSparse::saveAsCSV(std::string fileName)
{

}

void CSRSparse::dot(Dense &operandMatrix, Dense &targetMatrix)
{

}

Dense CSRSparse::dot(Dense &operandMatrix)
{

}

