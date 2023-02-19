#include "dense.h"

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
        loadBMS(fileName);
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

Dense::~Dense()
{
    delete[] _byteMatrix;
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

void Dense::loadBMS(std::string fileName)
{

}

void Dense::printRow(uint32_t rowIndex, uint8_t precision)
{
    Base::printRow(rowIndex);

    cout << setprecision(precision) << fixed;
    if (_dimMajority == ROW_MAJOR)
    {
        float *offsetMatrix = &_floatMatrix[_columns * rowIndex];
        for (uint32_t i = 0; i < _columns - 1; i++)
        {
            cout << offsetMatrix[i] << ',';
        }
        cout << offsetMatrix[_columns - 1] << endl;
    }
    else if (_dimMajority == COLUMN_MAJOR)
    {
        uint32_t i = rowIndex;
        for (; i < (_rows - 1) * _columns; i += _rows)
        {
            cout << _floatMatrix[i] << ',';
        }
        cout << _floatMatrix[i] << endl;
    }
}

void Dense::printColumn(uint32_t columnIndex, uint8_t precision)
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

void Dense::saveAsBMS(std::string fileName)
{

}

void Dense::saveAsCSV(std::string fileName)
{

}

