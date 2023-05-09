#include "dense.h"
#include "dense.cuh"

using namespace std;
using namespace Matrix;

Dense::Dense(string fileName, DimensionMajorityEnum dimMajority, uint16_t bytePadding, uint8_t threadPadding) : 
    Base(dimMajority), _bytePadding{bytePadding}, _threadPadding{threadPadding}
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

Dense::Dense(uint16_t rows, uint16_t columns, DimensionMajorityEnum dimMajority, uint8_t threadPadding) :
    Base(rows, columns, dimMajority), _bytePadding{0}, _threadPadding{threadPadding}
{
    uint32_t padding = (_threadPadding - _columns % _threadPadding) * (_columns % _threadPadding) * _columns * sizeof(float); // FIMXE: might not work for all majorities
    _byteMatrix = new byte[_size + padding]();
}

Dense::Dense(uint16_t rows, uint16_t columns, DimensionMajorityEnum dimMajority, uint16_t bytePadding) : 
    Dense(rows, columns, dimMajority, bytePadding, nullptr) {}

Dense::Dense(uint16_t rows, uint16_t columns, DimensionMajorityEnum dimMajority, std::byte *data) :
    Dense(rows, columns, dimMajority, 0, data) {}

Dense::Dense(const Matrix::Dense& other) : 
    Base(other._rows, other._columns, other._dimMajority), _bytePadding{other._bytePadding}
{
    cout << "Dense copy constructor called." << endl;
    cout << _size << endl;
    _byteMatrix = new byte[_size]();
    memcpy(_byteMatrix, other._byteMatrix, _size);
}

Dense::Dense(Matrix::Dense&& other) : 
    Base(other._rows, other._columns, other._dimMajority), _bytePadding{other._bytePadding}
{
    _floatMatrix = move(other._floatMatrix);
    other._floatMatrix = nullptr;
}

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

void Dense::setFloatMatrix(float *data, uint16_t rows, uint16_t columns, DimensionMajorityEnum dimMajority)
{
    _floatMatrix = data;
    _rows = rows;
    _columns = columns;
    _dimMajority = dimMajority;
}

void Dense::clear()
{
    _floatMatrix = nullptr;
    _rows = 0;
    _columns = 0;
    _size = 0;
    _dimMajority = FILE_DETERMINED;
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
        uint32_t threadPadding = (_threadPadding - _columns % _threadPadding) * (_columns % _threadPadding) * _columns * sizeof(float); // FIMXE: might not work for all majorities
        _byteMatrix = new byte[_size + _bytePadding + threadPadding]();
        
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

void Dense::saveAsBinary(string fileName)
{
    (void)fileName;
}

void Dense::saveAsCSV(string fileName)
{
    (void)fileName;
}

void Dense::dumpData(string fileName)
{
    ofstream file(fileName, ios_base::binary);
    if (file.is_open())
    {
        file.write(reinterpret_cast<char *>(_floatMatrix), _size);
        file.close();
    }
}

void Dense::loadDumpedData(string fileName, uint16_t rows, uint16_t columns, DimensionMajorityEnum dimMajority)
{
    ifstream file(fileName, ios_base::binary);
    if (file.is_open())
    {
        _rows = rows;
        _columns = columns;
        _dimMajority = dimMajority;
        _size = _rows * _columns * sizeof(float);
        _byteMatrix = new byte[_size + _bytePadding]();
        file.read(reinterpret_cast<char *>(_floatMatrix), _size);
        file.close();
    }
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

void Dense::dotNEON(Dense &operandMatrix, Dense &targetMatrix)
{
    if (_dimMajority == ROW_MAJOR)
    {
        if (operandMatrix._dimMajority == COLUMN_MAJOR)
        {
            for (uint16_t i = 0; i < operandMatrix._columns; i++)
            {
                float *column = &operandMatrix._floatMatrix[i * operandMatrix._rows];
                float *row = _floatMatrix;
                for (uint16_t j = 0; j < _rows; j++)
                {
                    float32x4_t accumulator = vdupq_n_f32(0);
                    for (uint16_t k = 0; k < _columns >> 5; k++)
                    {
                        float32x4_t a = vld1q_f32(row);
                        float32x4_t b = vld1q_f32(column);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 4);
                        b = vld1q_f32(column + 4);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 8);
                        b = vld1q_f32(column + 8);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 12);
                        b = vld1q_f32(column + 12);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 16);
                        b = vld1q_f32(column + 16);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 20);
                        b = vld1q_f32(column + 20);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 24);
                        b = vld1q_f32(column + 24);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 28);
                        b = vld1q_f32(column + 28);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        row += 32;
                        column += 32;
                    }
                    
                    targetMatrix._floatMatrix[i * _rows + j] = accumulator[0] + accumulator[1] + accumulator[2] + accumulator[3];
                    column -= _columns;
                }
            }
        }
    }
}

Dense Dense::dotNEON(Dense &operandMatrix)
{
    Dense targetMatrix(_rows, operandMatrix._columns, COLUMN_MAJOR);
    dotNEON(operandMatrix, targetMatrix);

    return targetMatrix;
}

void Dense::dotNEONThread(Dense &operandMatrix, Dense &targetMatrix, uint8_t numberOfThreads, uint8_t threadId)
{
    if (_dimMajority == ROW_MAJOR)
    {
        if (operandMatrix._dimMajority == COLUMN_MAJOR)
        {
            float *column = operandMatrix._floatMatrix;
            float *target = targetMatrix._floatMatrix + threadId * targetMatrix._rows / numberOfThreads;
            for (uint16_t i = 0; i < operandMatrix._columns; i++)
            {
                float *row = _floatMatrix + _columns * threadId * _rows / numberOfThreads;
                for (uint16_t j = 0; j < _rows / numberOfThreads; j++)
                {
                    float32x4_t accumulator = vdupq_n_f32(0);
                    for (uint16_t k = 0; k < _columns >> 5; k++)
                    {
                        float32x4_t a = vld1q_f32(row);
                        float32x4_t b = vld1q_f32(column);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 4);
                        b = vld1q_f32(column + 4);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 8);
                        b = vld1q_f32(column + 8);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 12);
                        b = vld1q_f32(column + 12);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 16);
                        b = vld1q_f32(column + 16);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 20);
                        b = vld1q_f32(column + 20);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 24);
                        b = vld1q_f32(column + 24);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 28);
                        b = vld1q_f32(column + 28);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        row += 32;
                        column += 32;
                    }
                    
                    target[j] = accumulator[0] + accumulator[1] + accumulator[2] + accumulator[3];
                    column -= _columns;
                }

                target += targetMatrix._rows;
                column += operandMatrix._rows;
            }
        }
    }
}

void Dense::dotNEONThreads(Dense &operandMatrix, Dense &targetMatrix, uint8_t numberOfThreads)
{
    vector<thread> threads(numberOfThreads);
    for (uint8_t i = 0; i < numberOfThreads; i++)
    {
        threads[i] = thread(&Dense::dotNEONThread, this, ref(operandMatrix), ref(targetMatrix), numberOfThreads, i);
    }

    for (uint8_t i = 0; i < numberOfThreads; i++)
    {
        threads[i].join();
    }
}

Dense Dense::dotNEONThreads(Dense &operandMatrix, uint8_t numberOfThreads)
{
    Dense targetMatrix(_rows, operandMatrix._columns, COLUMN_MAJOR);
    dotNEONThreads(operandMatrix, targetMatrix, numberOfThreads);

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

void Dense::dotAddActivate(Dense &dotMatrix, Dense &addMatrix, Dense &targetMatrix, float (*activationFunction)(float))
{
    if (_dimMajority == ROW_MAJOR)
    {
        if (dotMatrix._dimMajority == COLUMN_MAJOR && addMatrix._dimMajority == COLUMN_MAJOR)
        {
            for (uint16_t i = 0; i < dotMatrix._columns; i++)
            {
                float *column = &dotMatrix._floatMatrix[i * dotMatrix._rows];
                float *row = _floatMatrix;
                for (uint16_t j = 0; j < _rows; j++)
                {
                    float32x4_t accumulator = vdupq_n_f32(0);
                    for (uint16_t k = 0; k < _columns >> 5; k++)
                    {
                        float32x4_t a = vld1q_f32(row);
                        float32x4_t b = vld1q_f32(column);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 4);
                        b = vld1q_f32(column + 4);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 8);
                        b = vld1q_f32(column + 8);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 12);
                        b = vld1q_f32(column + 12);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 16);
                        b = vld1q_f32(column + 16);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 20);
                        b = vld1q_f32(column + 20);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 24);
                        b = vld1q_f32(column + 24);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 28);
                        b = vld1q_f32(column + 28);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        row += 32;
                        column += 32;
                    }
                    
                    float result = accumulator[0] + accumulator[1] + accumulator[2] + accumulator[3] + addMatrix._floatMatrix[j];
                    result = activationFunction(result);
                    targetMatrix._floatMatrix[i * _rows + j] = result;
                    column -= _columns;
                }
            }
        }
    }
}

Dense Dense::dotAddActivate(Dense &dotMatrix, Dense &addMatrix, float (*activationFunction)(float))
{
    Dense targetMatrix(_rows, dotMatrix._columns, COLUMN_MAJOR);
    dotAddActivate(dotMatrix, addMatrix, targetMatrix, activationFunction);

    return targetMatrix;
}

void Dense::dotAddActivateThread(Dense &dotMatrix, Dense &addMatrix, Dense &targetMatrix, float (*activationFunction)(float),
                                 uint8_t numberOfThreads, uint8_t threadId)
{
    if (_dimMajority == ROW_MAJOR)
    {
        if (dotMatrix._dimMajority == COLUMN_MAJOR && addMatrix._dimMajority == COLUMN_MAJOR)
        {
            float *dotColumn = dotMatrix._floatMatrix;
            float *target = targetMatrix._floatMatrix + threadId * _rows / numberOfThreads;
            float *addColumn = addMatrix._floatMatrix + threadId * _rows / numberOfThreads;
            for (uint16_t i = 0; i < dotMatrix._columns; i++)
            {
                float *row = _floatMatrix + _columns * threadId * _rows / numberOfThreads;
                for (uint16_t j = 0; j < _rows / numberOfThreads; j++)
                {
                    float32x4_t accumulator = vdupq_n_f32(0);
                    for (uint16_t k = 0; k < _columns >> 5; k++)
                    {
                        float32x4_t a = vld1q_f32(row);
                        float32x4_t b = vld1q_f32(dotColumn);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 4);
                        b = vld1q_f32(dotColumn + 4);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 8);
                        b = vld1q_f32(dotColumn + 8);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 12);
                        b = vld1q_f32(dotColumn + 12);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 16);
                        b = vld1q_f32(dotColumn + 16);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 20);
                        b = vld1q_f32(dotColumn + 20);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 24);
                        b = vld1q_f32(dotColumn + 24);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        a = vld1q_f32(row + 28);
                        b = vld1q_f32(dotColumn + 28);
                        accumulator = vmlaq_f32(accumulator, a, b);

                        row += 32;
                        dotColumn += 32;
                    }
                    
                    float result = accumulator[0] + accumulator[1] + accumulator[2] + accumulator[3] + addColumn[j];
                    result = activationFunction(result);
                    target[j] = result;
                    dotColumn -= _columns;
                }

                target += targetMatrix._rows;
                dotColumn += dotMatrix._rows;
            }
        }
    }
}

void Dense::dotAddActivateThreads(Dense &dotMatrix, Dense &addMatrix, Dense &targetMatrix, float (*activationFunction)(float),
                                  uint8_t numberOfThreads)
{
    vector<thread> threads(numberOfThreads);
    for (uint8_t i = 0; i < numberOfThreads; i++)
    {
        threads[i] = thread(&Dense::dotAddActivateThread, this, ref(dotMatrix), ref(addMatrix), ref(targetMatrix),
                             activationFunction, numberOfThreads, i);
    }

    for (auto &thread : threads)
    {
        thread.join();
    }
}

Dense Dense::dotAddActivateThreads(Dense &dotMatrix, Dense &addMatrix, float (*activationFunction)(float),
                                   uint8_t numberOfThreads)
{
    Dense targetMatrix(_rows, dotMatrix._columns, COLUMN_MAJOR);
    dotAddActivateThreads(dotMatrix, addMatrix, targetMatrix, activationFunction, numberOfThreads);

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

uint32_t Dense::argmax()
{
    float max = _floatMatrix[0];
    uint32_t maxIndex = 0;

    for (uint32_t i = 1; i < _columns * _rows; i++)
    {
        if (_floatMatrix[i] > max)
        {
            max = _floatMatrix[i];
            maxIndex = i;
        }
    }

    return maxIndex;
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
