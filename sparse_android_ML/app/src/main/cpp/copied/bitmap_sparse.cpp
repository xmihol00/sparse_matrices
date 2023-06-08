#include "bitmap_sparse.h"

using namespace std;
using namespace Matrix;

BitmapSparse::BitmapSparse(string fileName, DimensionMajorityEnum dimMajority) : Sparse(dimMajority) 
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

void BitmapSparse::allocateSpaceRowMajorCSV(ifstream &file)
{
    uint32_t totalEntries = 0;
    uint32_t nonZeroEntries = 0;
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
                nonZeroEntries++;
            }
            totalEntries++;
        }
    }

    _columns = totalEntries / _rows;
    _blocksPerDimension = (_columns + _BLOCK_SIZE - 1) / _BLOCK_SIZE; // ceil the number of blocks
    _size = _rows * sizeof(uint32_t) +                          // row indices
            _blocksPerDimension * sizeof(uint32_t) * _rows +    // bitmap blocks
            nonZeroEntries * sizeof(float);                     // non-zero values (data)
    _byteMatrix = new byte[_size]();
}

void BitmapSparse::allocateSpaceColumnMajorCSV(ifstream &file)
{
    uint32_t nonZeroEntries = 0;
    string row;
    string cell;
    
    getline(file, row);
    _columns = count(row.begin(), row.end(), ',') + 1;
    file.clear();
    file.seekg(0);

    _buffer = new uint32_t[_columns]();

    while (getline(file, row)) 
    {
        _rows++;
        stringstream rowStream(row);
        for (uint32_t i = 0; getline(rowStream, cell, ','); i++)
        {
            float cellValue = stof(cell);
            if (cellValue != 0.0f)
            {
                nonZeroEntries++;
                _buffer[i]++;
            }
        }
    }

    _blocksPerDimension = (_rows + _BLOCK_SIZE - 1) / _BLOCK_SIZE;
    _size = _columns * sizeof(uint32_t) + 
            _blocksPerDimension * sizeof(uint32_t) * _columns + 
            nonZeroEntries * sizeof(float);
    _byteMatrix = new byte[_size]();
}

void BitmapSparse::loadDataRowMajorCSV(ifstream &file)
{
    uint16_t rowIndex = 0;
    uint32_t dataIndex = _rows;
    uint32_t blockIndex = dataIndex;
    uint8_t blockPosition = 0;
    bool indexMoved = false;
    string row;
    string cell;

    while (getline(file, row))
    {
        blockPosition = 0;
        if (!indexMoved)
        {
            blockIndex = dataIndex++;
        }
        _uint32Matrix[rowIndex++] = dataIndex - 1; // starting index of each row in the continuous memory block

        stringstream rowStream(row);
        while (getline(rowStream, cell, ','))
        {
            indexMoved = false;
            float cellValue = stof(cell);
            if (cellValue != 0.0f)
            {
                _floatMatrix[dataIndex++] = cellValue;
                _uint32Matrix[blockIndex] |= 1 << blockPosition; // set the bit in the bitmap
            }

            blockPosition++;
            if (blockPosition == _BLOCK_SIZE) // move to the next bitmap block
            {
                blockPosition = 0;
                blockIndex = dataIndex;
                dataIndex++;
                indexMoved = true;
            }
        }
    }
}

void BitmapSparse::loadDataColumnMajorCSV(ifstream &file)
{
    uint32_t dataIndex = _columns;
    for (uint32_t i = 0; i < _columns; i++)
    {
        _uint32Matrix[i] = dataIndex;
        dataIndex += _buffer[i] + _blocksPerDimension;
        _buffer[i] = _uint32Matrix[i];
    }

    uint16_t columnIndex = 0;
    uint8_t blockPosition = 0;
    string row;
    string cell;

    while (getline(file, row))
    {
        columnIndex = 0;

        stringstream rowStream(row);
        while (getline(rowStream, cell, ','))
        {
            float cellValue = stof(cell);
            if (cellValue != 0.0f)
            {
                _uint32Matrix[_buffer[columnIndex]] |= 1 << blockPosition;
                _floatMatrix[_buffer[columnIndex] + popcount(_uint32Matrix[_buffer[columnIndex]])] = cellValue;
            }
            columnIndex++;
        }

        blockPosition++;
        if (blockPosition == _BLOCK_SIZE)
        {
            blockPosition = 0;
            
            for (uint32_t i = 0; i < _columns; i++)
            {
                _buffer[i] += popcount(_uint32Matrix[_buffer[i]]) + 1;
            }
        }
    }

    delete[] _buffer;
    _buffer = nullptr;
}

void BitmapSparse::printRow(uint16_t rowIndex, uint8_t precision)
{
    Base::printRow(rowIndex);

    cout << setprecision(precision) << fixed;
    if (_dimMajority == ROW_MAJOR)
    {
        uint32_t blockIndex = _uint32Matrix[rowIndex];
        uint32_t dataIndex = blockIndex;
        uint16_t columnCounter = 0;

        for (uint32_t i = 0; i < _blocksPerDimension; i++)
        {
            dataIndex++;
            for (uint8_t j = 0; j < _BLOCK_SIZE; j++)
            {
                if (_uint32Matrix[blockIndex] & (1 << j))
                {
                    cout << _floatMatrix[dataIndex];
                    dataIndex++;
                }
                else
                {
                    cout << 0.0f;
                }

                if (++columnCounter < _columns)
                {
                    cout << ',';
                }
                else
                {
                    cout << endl;
                    break;
                }
            }
            blockIndex = dataIndex;
        }
    }
    else if (_dimMajority == COLUMN_MAJOR)
    {
        uint16_t blockNumber = rowIndex / _BLOCK_SIZE;
        uint16_t maskedIndex = rowIndex & (_BLOCK_SIZE - 1);
        uint32_t targetBlockMask = ~(_UINT32_MASK_MAX << maskedIndex);

        for (uint16_t i = 0; ; )
        {
            uint32_t columnIndex = _uint32Matrix[i];
            for (uint16_t j = 0; j < blockNumber; j++)
            {
                columnIndex += popcount(_uint32Matrix[columnIndex]) + 1;
            }

            if (_uint32Matrix[columnIndex] & (1 << maskedIndex))
            {
                columnIndex += popcount(_uint32Matrix[columnIndex] & targetBlockMask);
                cout << _floatMatrix[columnIndex];
            }
            else
            {
                cout << 0.0f;
            }

            if (++i == _columns)
            {
                cout << endl;
                break;
            }
            else
            {
                cout << ',';
            }
        }
    }
    else
    {
        throw invalid_argument(_UNSUPPORTED_MAJORITY);
    }
}

void BitmapSparse::printColumn(uint16_t columnIndex, uint8_t precision)
{
    Base::printColumn(columnIndex);

    if (_dimMajority == COLUMN_MAJOR)
    {
        uint32_t blockIndex = _uint32Matrix[columnIndex];
        uint32_t dataIndex = blockIndex;
        uint32_t rowCounter = 0;

        cout << setprecision(precision) << fixed;
        for (uint32_t i = 0; i < _blocksPerDimension; i++)
        {
            dataIndex++;
            for (uint8_t j = 0; j < _BLOCK_SIZE; j++)
            {
                if (_uint32Matrix[blockIndex] & (1 << j))
                {
                    cout << _floatMatrix[dataIndex] << endl;
                    dataIndex++;
                }
                else
                {
                    cout << 0.0f << endl;
                }

                if (++rowCounter == _rows)
                {
                    break;
                }
            }
            blockIndex = dataIndex;
        }
    }
    else if (_dimMajority == ROW_MAJOR)
    {
        uint32_t blockNumber = columnIndex / _BLOCK_SIZE;
        uint32_t maskedIndex = columnIndex & (_BLOCK_SIZE - 1);
        uint32_t targetBlockMask = ~(_UINT32_MASK_MAX << maskedIndex);

        for (uint32_t i = 0; i < _rows; i++)
        {
            uint16_t rowIndex = _uint32Matrix[i];
            for (uint32_t j = 0; j < blockNumber; j++)
            {
                rowIndex += popcount(_uint32Matrix[rowIndex]) + 1;
            }

            if (_uint32Matrix[rowIndex] & (1 << maskedIndex))
            {
                rowIndex += popcount(_uint32Matrix[rowIndex] & targetBlockMask);
                cout << _floatMatrix[rowIndex] << endl;
            }
            else
            {
                cout << 0.0f << endl;
            }
        }
    }
    else
    {
        throw invalid_argument(_UNSUPPORTED_MAJORITY);
    }
}

void BitmapSparse::loadBinary(std::string fileName)
{
    ifstream file(fileName, ios_base::binary);
    if (file.is_open())
    {
        file.read(reinterpret_cast<char *>(&_size), sizeof(_size));
        file.read(reinterpret_cast<char *>(&_rows), sizeof(_rows));
        file.read(reinterpret_cast<char *>(&_columns), sizeof(_columns));
        file.read(reinterpret_cast<char *>(&_blocksPerDimension), sizeof(_blocksPerDimension));

        DimensionMajorityEnum dimMajority;
        file.read(reinterpret_cast<char *>(&dimMajority), sizeof(dimMajority));
        if (_dimMajority != FILE_DETERMINED && _dimMajority != dimMajority)
        {
            throw invalid_argument("Specified dimension majority and stored dimension majority do not match.");
        }
        _dimMajority = dimMajority;

        _byteMatrix = new byte[_size]();
        file.read(_charMatrix, _size);
    }

    file.close();
}

void BitmapSparse::saveAsCSV(std::string fileName)
{
    (void)fileName;
}

void BitmapSparse::saveAsBinary(string fileName)
{
    ofstream file(fileName, ios_base::binary);
    if (file.is_open())
    {
        file.write(reinterpret_cast<char *>(&_size), sizeof(_size));
        file.write(reinterpret_cast<char *>(&_rows), sizeof(_rows));
        file.write(reinterpret_cast<char *>(&_columns), sizeof(_columns));
        file.write(reinterpret_cast<char *>(&_blocksPerDimension), sizeof(_blocksPerDimension));
        file.write(reinterpret_cast<char *>(&_dimMajority), sizeof(_dimMajority));
        file.write(_charMatrix, _size);
    }

    file.close();
}

void BitmapSparse::moveToRow(uint16_t rowIndex)
{
    if (_dimMajority == ROW_MAJOR)
    {
        _blockIndex = _uint32Matrix[rowIndex];
    }
}

void BitmapSparse::moveToColumn(uint16_t columnIndex)
{
    if (_dimMajority == COLUMN_MAJOR)
    {
        _blockIndex = _uint32Matrix[columnIndex];
    }
}

std::tuple<uint32_t, float *> BitmapSparse::nextRowBlock()
{
    if (_dimMajority == ROW_MAJOR)
    {
        uint32_t currentBlock = _uint32Matrix[_blockIndex];
        float *currentBlockData = &_floatMatrix[_blockIndex];
        _blockIndex += popcount(currentBlock) + 1;

        return {currentBlock, currentBlockData};
    }

    return {0, nullptr};
}

std::tuple<uint32_t, float *> BitmapSparse::nextColumnBlock()
{
    if (_dimMajority == COLUMN_MAJOR)
    {
        uint32_t currentBlock = _uint32Matrix[_blockIndex];
        float *currentBlockData = &_floatMatrix[_blockIndex];
        _blockIndex += popcount(currentBlock) + 1;

        return {currentBlock, currentBlockData};
    }

    return {0, nullptr};
}

void BitmapSparse::dot(BitmapSparse &operandMatrix, Dense &targetMatrix)
{
    if (_columns != operandMatrix._rows)
    {
        throw invalid_argument((stringstream() << "Dimension missmatch. The number of columns (" << _columns << ") does not match the number of rows (" << operandMatrix._rows << ").").str());
    }

    for (uint32_t i = 0; i < _rows; i++)
    {
        operandMatrix.moveToColumn(0);
        for (uint32_t j = 0; j < operandMatrix._columns; j++)
        {
            moveToRow(i);
            float accumulator = 0;
            for (uint32_t k = 0; k < _blocksPerDimension; k++)
            {
                auto [rowBlock, rowData] = nextRowBlock();
                auto [columnBlock, columnData] = operandMatrix.nextColumnBlock();

                uint32_t matchedBlock = rowBlock & columnBlock;
                // with HW support countr_zero() and popcount() should be executed with a single instruction, i.e one CPU cycle
                for (uint8_t l = countr_zero(matchedBlock); l < _BLOCK_SIZE; l = countr_zero(matchedBlock & (_UINT32_MASK_MAX << l)))
                {
                    uint32_t popcountMask = ~(_UINT32_MASK_MAX << l);
                    uint8_t rowDataOffset = popcount(rowBlock & popcountMask);
                    uint8_t columnDataOffset = popcount(columnBlock & popcountMask);

                    accumulator += rowData[rowDataOffset] * columnData[columnDataOffset];
                }
            }
            targetMatrix._floatMatrix[i * operandMatrix._columns +  j] = accumulator;
        }
    }

    targetMatrix._dimMajority = ROW_MAJOR;
}

Dense BitmapSparse::dot(BitmapSparse &operandMatrix)
{
    Dense targetMatrix(_rows, operandMatrix._columns, ROW_MAJOR);
    dot(operandMatrix, targetMatrix);

    return targetMatrix;
}

void BitmapSparse::dot(Dense &operandMatrix, Dense &targetMatrix)
{
    if (_dimMajority == ROW_MAJOR)
    {
        if (operandMatrix._dimMajority == ROW_MAJOR)
        {
            for (uint32_t i = 0; i < operandMatrix._columns; i++)
            {
                moveToRow(0);
                uint32_t targetOffset = i * _rows;
                for (uint32_t j = 0; j < _rows; j++)
                {
                    float accumulator = 0;
                    uint16_t rowIndex = 0;    
                    for (uint32_t k = 0; k < _blocksPerDimension; k++)
                    {
                        auto [rowBlock, rowData] = nextRowBlock();    
                        for (uint8_t l = countr_zero(rowBlock), m = 1; l < _BLOCK_SIZE; 
                                l = countr_zero(rowBlock & (_UINT32_MASK_MAX << l)), m++)
                        {
                            accumulator += rowData[m] * operandMatrix._floatMatrix[(rowIndex + l) * operandMatrix._columns + i];
                        }
                        rowIndex += _BLOCK_SIZE;
                    }
                    targetMatrix._floatMatrix[targetOffset +  j] = accumulator;
                }
            }
        }
        else if (operandMatrix._dimMajority == COLUMN_MAJOR)
        {
            for (uint32_t i = 0; i < operandMatrix._columns; i++)
            {
                moveToRow(0);
                uint32_t targetOffset = i * _rows;
                float *operandOffset = &operandMatrix._floatMatrix[i * operandMatrix._rows];
                for (uint32_t j = 0; j < _rows; j++)
                {
                    float accumulator = 0;
                    uint16_t columnIndex = 0;
                    for (uint16_t k = 0; k < _blocksPerDimension; k++)
                    {
                        auto [rowBlock, rowData] = nextRowBlock();    
                        for (uint8_t l = countr_zero(rowBlock), m = 1; l < _BLOCK_SIZE; 
                                l = countr_zero(rowBlock & (_UINT32_MASK_MAX << l)), m++)
                        {
                            accumulator += rowData[m] * operandOffset[columnIndex + l];
                        }
                        columnIndex += _BLOCK_SIZE;
                    }
                    targetMatrix._floatMatrix[targetOffset +  j] = accumulator;
                }
            }
        }
    }
    else if (_dimMajority == COLUMN_MAJOR)
    {

    }
    else
    {

    }
}

Dense BitmapSparse::dot(Dense &operandMatrix)
{
    Dense targetMatrix(_rows, operandMatrix._columns, COLUMN_MAJOR);
    dot(operandMatrix, targetMatrix);

    return targetMatrix;
}
