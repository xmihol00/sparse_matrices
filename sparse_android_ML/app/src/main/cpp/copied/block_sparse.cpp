#include "block_sparse.h"
#include "typedefs.h"

using namespace std;
using namespace Matrix;

BlockSparse::BlockSparse(std::string fileName, uint16_t blocksPerDimension, DimensionMajorityEnum dimMajority)
    : Sparse(dimMajority), _blocksPerDimension(blocksPerDimension)
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
    (void)file;
}

void BlockSparse::loadDataRowMajorCSV(std::ifstream &file)
{
    uint32_t blockIndex = 0;
    uint32_t entriesOffsetsIndex = 0;
    uint32_t dataIndex = 0;
    
    // creating pointers for the entries offsets and the data, use of unions to avoid casting
    matrix_ptrs_t tmp;
    tmp.uint16s = &_uint16Matrix[_blocksPerDimension * _rows];
    _entryIndices = tmp.uint8s;
    tmp.uint8s = &_entryIndices[(_entriesPerBlock - 1) * _blocksPerDimension * _rows];
    _dataMatrix = tmp.floats;

    string row;
    string cell;
    uint32_t line = 0;
    while (getline(file, row))
    {
        line++;
        float cellValue;
        stringstream rowStream(row);
        uint16_t blockOffset = 0;

        for (uint16_t i = 0; i < _blocksPerDimension; i++)
        {
            bool read;
            for (; (read = static_cast<bool>(getline(rowStream, cell, ','))) && (cellValue = stof(cell)) == 0; blockOffset++); // loop without body
            if (read)
            {
                _uint16Matrix[blockIndex++] = blockOffset;
                _dataMatrix[dataIndex++] = cellValue;
            }
            else
            {
                _uint16Matrix[blockIndex++] = --blockOffset;
                _dataMatrix[dataIndex++] = 0.0f;
            }

            uint16_t k = 0;
            for (uint8_t j = 1; j < _entriesPerBlock && k < UINT8_MAX; )
            {
                if (!getline(rowStream, cell, ','))
                {
                    // first, padd the current block
                    for (; j < _entriesPerBlock; j++)
                    {
                        _dataMatrix[dataIndex++] = 0.0f;
                        _entryIndices[entriesOffsetsIndex++] = k; // the offset does not matter, as the 0 value indicates padding
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
                            _entryIndices[entriesOffsetsIndex++] = k; // the offset does not matter, as the 0 value indicates padding
                        }
                    }
                    break;
                }

                k++;
                cellValue = stof(cell);
                if (cellValue != 0.0f)
                {
                    _dataMatrix[dataIndex++] = cellValue;
                    _entryIndices[entriesOffsetsIndex++] = k;
                    j++;
                }
            }

            if (k >= UINT8_MAX)
            {
                throw out_of_range((stringstream() << "Not enough values in a range of block " << blockIndex % _blocksPerDimension << " at row " << blockIndex / _blocksPerDimension << '.').str());
            }
            blockOffset += k + 1;
        }
    }
}

void BlockSparse::loadDataColumnMajorCSV(std::ifstream &file)
{
    (void)file;
}

void BlockSparse::loadBinary(std::string fileName)
{
    (void)fileName;
}

void BlockSparse::saveAsBinary(std::string fileName)
{
    (void)fileName;
}

void BlockSparse::saveAsCSV(std::string fileName)
{
    (void)fileName;
}

void BlockSparse::printColumn(uint16_t columnIndex, uint8_t precision)
{
    (void)columnIndex;
    (void)precision;
}

void BlockSparse::printRow(uint16_t rowIndex, uint8_t precision)
{
    Base::printRow(rowIndex);

    cout << setprecision(precision) << fixed;
    if (_dimMajority == ROW_MAJOR)
    {
        int32_t previousBlockIndex = -1;
        float *offsetDataMatrix = &_dataMatrix[rowIndex * _entriesPerDimension];
        uint16_t dataIndex = 0;

        uint16_t columnIndex = 0;
        uint32_t upperBound = _blocksPerDimension + _blocksPerDimension * rowIndex;
        for (uint32_t i = _blocksPerDimension * rowIndex; i < upperBound; i++)
        {
            int32_t blockIndex = _uint16Matrix[i];
            int32_t blockOffset = blockIndex;
            uint32_t upperBound = static_cast<uint32_t>(i * (_entriesPerBlock - 1) + _entriesPerBlock);
            for (uint32_t j = i * (_entriesPerBlock - 1); j < upperBound; j++)
            {
                for (int16_t k = 1; k < blockIndex - previousBlockIndex && columnIndex < _columns; k++)
                {
                    cout << setw(precision + 3) << 0.0f;
                    if (++columnIndex < _columns)
                    {
                        cout << ',';
                    }
                    else
                    {
                        break;
                    }
                }

                if (offsetDataMatrix[dataIndex] && columnIndex < _columns)
                {
                    cout << setw(precision + 3) << offsetDataMatrix[dataIndex];
                    if (++columnIndex < _columns)
                    {
                        cout << ',';
                    }
                    else
                    {
                        break;
                    }
                }
                dataIndex++;

                previousBlockIndex = blockIndex;
                blockIndex = _entryIndices[j] + blockOffset;
                //cout << previousBlockIndex << " " << blockIndex << endl;
            }
        }

        if (columnIndex < _columns)
        {
            for (columnIndex++; columnIndex < _columns; columnIndex++)
            {
                cout << setw(precision + 3) << 0.0f << ',';
            }
            cout << setw(precision + 3) << 0.0f;
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

void BlockSparse::dotRowColumn(Dense &operandMatrix, Dense &targetMatrix)
{
    for (uint16_t i = 0; i < _rows; i++)
    {
        uint32_t blocks = i * _blocksPerDimension;
        uint32_t entries = blocks * _entriesPerBlock;
        uint32_t entriesSub = entries - blocks;
        for (uint16_t j = 0; j < operandMatrix._columns; j++)
        {
            uint16_t *offsetBlockIndices = &_uint16Matrix[blocks];
            uint8_t *offsetEntryIndices = &_entryIndices[entriesSub];
            float *row = &_dataMatrix[entries];
            float accumulator = 0;
            for (uint16_t k = 0; k < _blocksPerDimension; k++)
            {
                float *column = &operandMatrix._floatMatrix[j * operandMatrix._rows + offsetBlockIndices[k]];
                for (uint8_t m = 0, n = 0; m < _entriesPerBlock; n = offsetEntryIndices[m++])
                {
                    accumulator += row[m] * column[n];
                }
                offsetEntryIndices = &offsetEntryIndices[_entriesPerBlock - 1];
                row = &row[_entriesPerBlock];
            }
            targetMatrix._floatMatrix[j * _rows + i] = accumulator;
        }
    }
}

void BlockSparse::dot(Dense &operandMatrix, Dense &targetMatrix)
{
    if (_dimMajority == ROW_MAJOR)
    {
        if (operandMatrix._dimMajority == COLUMN_MAJOR)
        {
            dotRowColumn(operandMatrix, targetMatrix);
        }
    }
}

Dense BlockSparse::dot(Dense &operandMatrix)
{
    Dense targetMatrix(_rows, operandMatrix._columns, COLUMN_MAJOR);
    dot(operandMatrix, targetMatrix);

    return targetMatrix;
}

