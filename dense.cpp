#include "dense.h"
#include "dense_cuda.h"

using namespace std;
using namespace Matrix;

Dense::Dense(string fileName, DimenstionMajorityEnum dimMajority) : Base(dimMajority) 
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

Dense::Dense(uint32_t rows, uint32_t columns, DimenstionMajorityEnum dimMajority, std::byte *data) : Base(rows, columns, dimMajority)
{
    _byteMatrix = new byte[_size]();

    if (data != nullptr)
    {
        memcpy(_byteMatrix, data, _size);
    }
}

void Dense::loadCSV(string fileName)
{
    ifstream file(fileName);
    if (file.is_open()) 
    {

        string row;
        getline(file, row);
        _columns = count(row.begin(), row.end(), ',') + 1;
        _rows = count(istreambuf_iterator<char>(file), istreambuf_iterator<char>(), '\n') + 1;
        _size = _columns * _rows * sizeof(float);
        _byteMatrix = new byte[_size]();
        
        file.clear();
        file.seekg(0);

        string cell;
        if (_dimMajority == ROW_MAJOR)
        {
            for (uint32_t i = 0; getline(file, row); i++)
            {
                stringstream rowStream(row);
                for (uint32_t j = 0; getline(rowStream, cell, ','); j++)
                {
                    _floatMatrix[i * _columns + j] = stof(cell);
                }
            }
        }
        else if (_dimMajority == COLUMN_MAJOR)
        {
            for (uint32_t i = 0; getline(file, row); i++)
            {
                stringstream rowStream(row);
                for (uint32_t j = 0; getline(rowStream, cell, ','); j++)
                {
                    _floatMatrix[j * _rows + i] = stof(cell);
                }
            }
        }
        else
        {
            throw invalid_argument(_UNSUPPORTED_MAJORITY);
        }

        file.close();
    }
}

void Dense::loadBinary(std::string fileName)
{
    (void)fileName;
}

void Dense::printRow(uint16_t rowIndex, uint8_t precision)
{
    Base::printRow(rowIndex);

    cout << setprecision(precision) << fixed;
    if (_dimMajority == ROW_MAJOR)
    {
        float *offsetMatrix = &_floatMatrix[_columns * rowIndex];
        for (int32_t i = 0; i < _columns - 1; i++)
        {
            cout << offsetMatrix[i] << ',';
        }
        cout << offsetMatrix[_columns - 1] << endl;
    }
    else if (_dimMajority == COLUMN_MAJOR)
    {
        int32_t i = rowIndex;
        int32_t upperBound = (_rows - 1) * _columns;
        for (; i < upperBound; i += _rows)
        {
            cout << _floatMatrix[i] << ',';
        }
        cout << _floatMatrix[i] << endl;
    }
}

void Dense::printColumn(uint16_t columnIndex, uint8_t precision)
{
    Base::printColumn(columnIndex);

    cout << setprecision(precision) << fixed;
    if (_dimMajority == ROW_MAJOR)
    {
        for (uint32_t i = columnIndex; i < _rows * _columns; i += _columns)
        {
            cout << _floatMatrix[i] << endl;
        }
    }
    else if (_dimMajority == COLUMN_MAJOR)
    {
        float *offsetMatrix = &_floatMatrix[_rows * columnIndex];
        for (uint32_t i = 0; i < _rows; i++)
        {
            cout << offsetMatrix[i] << endl;
        }
    }
}

void Dense::saveAsBinary(std::string fileName)
{
    (void)fileName;
}

void Dense::saveAsCSV(std::string fileName)
{
    (void)fileName;
}

void Dense::dot(Dense &operandMatrix, Dense &targetMatrix)
{
    if (_dimMajority == ROW_MAJOR)
    {
        if (operandMatrix._dimMajority == COLUMN_MAJOR)
        {
            for (uint16_t i = 0; i < operandMatrix._columns; i++)
            {
                float *column = &operandMatrix._floatMatrix[i * operandMatrix._rows];
                for (uint16_t j = 0; j < _rows; j++)
                {
                    float *row = &_floatMatrix[j * _columns];
                    float accumulator = 0;
                    for (uint16_t k = 0; k < _columns; k++)
                    {
                        accumulator += row[k] * column[k];
                    }
                    targetMatrix._floatMatrix[i * _rows + j] = accumulator;
                }
            }
        }
    }
}

Dense Dense::dot(Dense &operandMatrix)
{
    Dense targetMatrix(_rows, operandMatrix._columns, COLUMN_MAJOR);
    dot(operandMatrix, targetMatrix);

    return targetMatrix;
}


void Dense::dotGPU(Dense &operandMatrix, Dense &targetMatrix)
{
    if (_dimMajority == ROW_MAJOR)
    {
        if (operandMatrix._dimMajority == COLUMN_MAJOR)
        {
            dotRowsColumns(_floatMatrix, operandMatrix._floatMatrix, targetMatrix._floatMatrix, _rows, operandMatrix._columns, 
                           _size, operandMatrix._size);
        }
    }
}

Dense Dense::dotGPU(Dense &operandMatrix)
{
    Dense targetMatrix(_rows, operandMatrix._columns, COLUMN_MAJOR);
    dotGPU(operandMatrix, targetMatrix);

    return targetMatrix;
}
