#include "dense.h"
#include "dense.cuh"

using namespace std;
using namespace Matrix;

Dense::Dense(string fileName, DimensionMajorityEnum dimMajority, uint16_t bytePadding) : Base(dimMajority), _bytePadding{bytePadding}
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

Dense::Dense(uint16_t rows, uint16_t columns, DimensionMajorityEnum dimMajority, uint16_t bytePadding, std::byte *data) : 
    Base(rows, columns, dimMajority), _bytePadding{bytePadding}
{
    _byteMatrix = new byte[_size + _bytePadding]();

    if (data != nullptr)
    {
        memcpy(_byteMatrix, data, _size);
    }
}

Dense::Dense(uint16_t rows, uint16_t columns, DimensionMajorityEnum dimMajority) : 
    Dense(rows, columns, dimMajority, 0, nullptr) {}

Dense::Dense(uint16_t rows, uint16_t columns, DimensionMajorityEnum dimMajority, uint16_t bytePadding) : 
    Dense(rows, columns, dimMajority, bytePadding, nullptr) {}

Dense::Dense(uint16_t rows, uint16_t columns, DimensionMajorityEnum dimMajority, std::byte *data) :
    Dense(rows, columns, dimMajority, 0, data) {}

Dense &Dense::operator=(Dense &&other)
{
    if (this != &other)
    {
        _floatMatrix = move(other._floatMatrix);
        _columns = move(other._columns);
        _rows = move(other._rows);
        _size = move(other._size);
        _dimMajority = move(other._dimMajority);
        _bytePadding = move(other._bytePadding);

        other._floatMatrix = nullptr;
    }

    return *this;
}

float &Dense::operator()(uint16_t rowIndex, uint16_t columnIndex)
{
    if (_dimMajority == ROW_MAJOR)
    {
        return _floatMatrix[rowIndex * _columns + columnIndex];
    }
    else
    {
        return _floatMatrix[columnIndex * _rows + rowIndex];
    }
    
    throw invalid_argument(_UNSUPPORTED_MAJORITY);
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
        _byteMatrix = new byte[_size + _bytePadding]();
        
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
            cout << setw(precision + 3) << offsetMatrix[i] << ',';
        }
        cout << setw(precision + 3) << offsetMatrix[_columns - 1] << endl;
    }
    else if (_dimMajority == COLUMN_MAJOR)
    {
        int32_t i = rowIndex;
        int32_t upperBound = (_rows - 1) * _columns;
        for (; i < upperBound && _columns > 1; i += _rows)
        {
            cout << setw(precision + 3) << _floatMatrix[i] << ',';
        }
        cout << setw(precision + 3) << _floatMatrix[i] << endl;
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
            cout << setw(precision + 3) << _floatMatrix[i] << endl;
        }
    }
    else if (_dimMajority == COLUMN_MAJOR)
    {
        float *offsetMatrix = &_floatMatrix[_rows * columnIndex];
        for (uint32_t i = 0; i < _rows; i++)
        {
            cout << setw(precision + 3) << offsetMatrix[i] << endl;
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

void Dense::dotGPUCuBLAS(Dense &operandMatrix, Dense &targetMatrix)
{
    dotCuBLAS(_floatMatrix, operandMatrix._floatMatrix, targetMatrix._floatMatrix, _rows, operandMatrix._columns, _columns, 
              _rows, operandMatrix._columns, _columns);
}

Dense Dense::dotGPUCuBLAS(Dense &operandMatrix)
{
    Dense targetMatrix(_rows, operandMatrix._columns, COLUMN_MAJOR);
    dotGPUCuBLAS(operandMatrix, targetMatrix);

    return targetMatrix;
}

void Dense::add(Dense &operandMatrix)
{
    if (_dimMajority == operandMatrix._dimMajority)
    {
        if (_size == operandMatrix._size)
        {
            for (uint32_t i = 0; i < _columns * _rows; i++)
            {
                _floatMatrix[i] += operandMatrix._floatMatrix[i];
            }
        }
        else if (_dimMajority == COLUMN_MAJOR && _rows == operandMatrix._rows && operandMatrix._columns == 1)
        {
            for (uint32_t i = 0; i < _columns; i++)
            {
                float *column = &_floatMatrix[i * _rows];
                for (uint32_t j = 0; j < _rows; j++)
                {
                    column[j] += operandMatrix._floatMatrix[j];
                }
            }
        }
        else if (_dimMajority == ROW_MAJOR && _columns == operandMatrix._columns)
        {
            for (uint32_t i = 0; i < _rows; i++)
            {
                float *row = &_floatMatrix[i * _columns];
                for (uint32_t j = 0; j < _columns; j++)
                {
                    row[j] += operandMatrix._floatMatrix[j];
                }
            }
        }
    }
}

void Dense::ReLU()
{
    for (uint32_t i = 0; i < _columns * _rows; i++)
    {
        _floatMatrix[i] = max(0.0f, _floatMatrix[i]);
    }
}

void Dense::argmax(uint8_t axis, Dense &targetMatrix)
{
    if (axis == 0) // argmax over columns
    {
        if (_dimMajority == COLUMN_MAJOR)
        {
            for (uint32_t i = 0; i < _columns; i++)
            {
                float *column = &_floatMatrix[i * _rows];
                float max = column[0];
                uint32_t maxIndex = 0;
                for (uint32_t j = 1; j < _rows; j++)
                {
                    if (column[j] > max)
                    {
                        max = column[j];
                        maxIndex = j;
                    }
                }
                targetMatrix._floatMatrix[i] = maxIndex;
            }
        }
    }
    else if (axis == 1) // argmax over rows
    {
        if (_dimMajority == ROW_MAJOR)
        {
            for (uint32_t i = 0; i < _rows; i++)
            {
                float *row = &_floatMatrix[i * _columns];
                float max = row[0];
                uint32_t maxIndex = 0;
                for (uint32_t j = 1; j < _columns; j++)
                {
                    if (row[j] > max)
                    {
                        max = row[j];
                        maxIndex = j;
                    }
                }
                targetMatrix._floatMatrix[i] = maxIndex;
            }
        }
    }
}

Dense Dense::argmax(uint8_t axis)
{
    Dense targetMatrix(axis == 0 ? _columns : _rows, 1, COLUMN_MAJOR);
    argmax(axis, targetMatrix);

    return targetMatrix;
}

float Dense::percentageDifference(Dense &operandMatrix, float threshold)
{
    if (_dimMajority == operandMatrix._dimMajority)
    {
        uint32_t numberOfElements = _columns * _rows;
        uint32_t sameElements = 0;
        for (uint32_t i = 0; i < numberOfElements; i++)
        {
            if (abs(_floatMatrix[i] - operandMatrix._floatMatrix[i]) <= threshold)
            {
                sameElements++;
            }
        }

        return static_cast<float>(sameElements) / static_cast<float>(numberOfElements);
    }

    return 1.0f;
}