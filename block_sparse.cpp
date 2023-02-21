#include "block_sparse.h"
#include "typedefs.h"

using namespace std;
using namespace Matrix;

BlockSparse::BlockSparse(std::string fileName, uint16_t blocksPerDimension, DimenstionMajorityEnum dimMajority)
    : Sparse(dimMajority, blocksPerDimension)
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

void BlockSparse::allocateSpaceRowMajorCSV(std::ifstream &file)
{
    uint32_t totalEntries = 0;
    uint32_t maxEntriesPerLine = 0;
    string row;
    string cell;

    while (getline(file, row)) 
    {
        _rows++;
        uint32_t nonZeroEntriesPerLine = 0;
        stringstream rowStream(row);
        while (getline(rowStream, cell, ','))
        {
            float cellValue = stof(cell);
            if (cellValue != 0.0f)
            {
                nonZeroEntriesPerLine++;
            }
            totalEntries++;
        }

        if (nonZeroEntriesPerLine > maxEntriesPerLine)
        {
            maxEntriesPerLine = nonZeroEntriesPerLine;
        }
    }

    _entriesPerBlock = (maxEntriesPerLine + _blocksPerDimension - 1) / _blocksPerDimension;
    _entriesPerDimension = _entriesPerBlock * _blocksPerDimension;

    _columns = totalEntries / _rows;
    _size = _blocksPerDimension *_rows * sizeof(uint16_t) + 
            (_entriesPerBlock - 1) * _blocksPerDimension * _rows * sizeof(uint8_t) + // -1 becauase the index of the first entry is determined by the block 
            _entriesPerDimension * _rows * sizeof(float);
    _byteMatrix = new byte[_size]();
}

void BlockSparse::allocateSpaceColumnMajorCSV(std::ifstream &file)
{

}

void BlockSparse::loadDataRowMajorCSV(std::ifstream &file)
{
    uint16_t blockIndex = 0;
    uint32_t entriesOffsetsIndex = 0;
    uint32_t dataIndex = 0;
    
    // creating pointers for the entries offsets and the data, use of unions to avoid casting
    matrix_ptrs_t tmp;
    tmp.uint16Matrix = &_uint16Matrix[_blocksPerDimension * _rows];
    _entriesOffsets = tmp.uint8Matrix;
    tmp.uint8Matrix = &_entriesOffsets[(_entriesPerBlock - 1) * _blocksPerDimension * _rows];
    _dataMatrix = tmp.floatMatrix;

    string row;
    string cell;

    while (getline(file, row))
    {
        float cellValue;
        stringstream rowStream(row);
        uint16_t blockOffset = 0;

        for (uint16_t i = 0; i < _blocksPerDimension; i++)
        {
            for (; getline(rowStream, cell, ',') && (cellValue = stof(cell)) == 0; blockOffset++); // loop without body
            _uint16Matrix[blockIndex++] = blockOffset;
            _dataMatrix[dataIndex++] = cellValue;

            uint16_t k = 1;
            for (uint8_t j = 1; j < _entriesPerBlock && k <= UINT8_MAX; k++)
            {
                if (!getline(rowStream, cell, ','))
                {
                    // first, padd the current block
                    for (; j < _entriesPerBlock; j++)
                    {
                        _dataMatrix[dataIndex++] = 0.0f;
                        _entriesOffsets[entriesOffsetsIndex++] = k; // the offset does not metter, as the 0 value indicates padding
                    }
                    i++;

                    // second, padd all other yet unfilled blocks
                    blockOffset += k;
                    k = 0;
                    for (; i < _blocksPerDimension; i++)
                    {
                        _uint16Matrix[blockIndex++] = blockOffset;
                        _dataMatrix[dataIndex++] = 0.0f;
                        for (j = 1; j < _entriesPerBlock; j++)
                        {
                            _dataMatrix[dataIndex++] = 0.0f;
                            _entriesOffsets[entriesOffsetsIndex++] = k; // the offset does not metter, as the 0 value indicates padding
                        }
                    }
                    break;
                }

                cellValue = stof(cell);
                if (cellValue != 0.0f)
                {
                    _dataMatrix[dataIndex++] = cellValue;
                    _entriesOffsets[entriesOffsetsIndex++] = k;
                    j++;
                }
            }

            if (k > UINT8_MAX)
            {
                throw out_of_range((stringstream() << "Not enough values in a range of block " << blockIndex % _blocksPerDimension << " at row " << blockIndex / _blocksPerDimension << '.').str());
            }
            blockOffset += k;
        }
    }
}

void BlockSparse::loadDataColumnMajorCSV(std::ifstream &file)
{

}

void BlockSparse::loadBinary(std::string fileName)
{

}

void BlockSparse::saveAsBinary(std::string fileName)
{

}

void BlockSparse::saveAsCSV(std::string fileName)
{

}

void BlockSparse::printColumn(uint32_t columnIndex, uint8_t precision)
{

}

void BlockSparse::printRow(uint32_t rowIndex, uint8_t precision)
{
    Base::printRow(rowIndex);

    cout << setprecision(precision) << fixed;
    if (_dimMajority == ROW_MAJOR)
    {
        uint32_t rowOffset = rowIndex * _entriesPerBlock;
        int32_t previousBlockIndex = -1;
        float *offsetDataMatrix = &_dataMatrix[rowIndex * _entriesPerDimension];
        uint16_t dataIndex = 0;

        uint32_t columnIndex = 0;
        for (uint32_t i = _blocksPerDimension * rowIndex; i < _blocksPerDimension + _blocksPerDimension * rowIndex; i++)
        {
            int32_t blockIndex = _uint16Matrix[i];
            int32_t blockOffset = blockIndex;
            for (uint32_t j = i * (_entriesPerBlock - 1); j < i * (_entriesPerBlock - 1) + _entriesPerBlock; j++)
            {
                for (int16_t k = 1; k < blockIndex - previousBlockIndex; k++)
                {
                    cout << 0.0f;
                    if (++columnIndex < _columns)
                    {
                        cout << ',';
                    }
                    else
                    {
                        break;
                    }
                }

                if (offsetDataMatrix[dataIndex])
                {
                    cout << offsetDataMatrix[dataIndex];
                    if (++columnIndex < _columns)
                    {
                        cout << ',';
                    }
                }
                dataIndex++;

                previousBlockIndex = blockIndex;
                blockIndex = _entriesOffsets[j] + blockOffset;
            }
        }

        if (columnIndex < _columns)
        {
            for (; columnIndex < _columns - 1; columnIndex++)
            {
                cout << 0.0f << ',';
            }
            cout << 0.0f;
        }

        cout << endl;
    }
    else if (_dimMajority == COLUMN_MAJOR)
    {
        
    }
    else
    {
        throw invalid_argument(_UNSUPPORTED_MAJORITY);
    }
}
