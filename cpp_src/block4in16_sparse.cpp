#include "block4in16_sparse.h"

using namespace std;
using namespace Matrix;

Block4in16Sparse::Block4in16Sparse(string fileName, DimensionMajorityEnum dimMajority) : Sparse(dimMajority)
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

void Block4in16Sparse::allocateSpaceRowMajorCSV(ifstream &file)
{
    string row;
    getline(file, row);
    _columns = count(row.begin(), row.end(), ',') + 1;
    _rows = count(istreambuf_iterator<char>(file), istreambuf_iterator<char>(), '\n') + 1;
    _columns += (16 - (_columns & 15)) * ((_columns & 15) != 0);  // pad columns to 16
    _rows += (16 - (_rows & 15)) * ((_rows & 15) != 0);           // pad rows to 16
    _size = (_columns * _rows * sizeof(float) >> 2) + _columns * _rows * sizeof(uint8_t);
    _byteMatrix = new(align_val_t{16}) byte[_size](); // allocate aligned memory to 16 bytes to allow AVX/NEON instructions
}

void Block4in16Sparse::allocateSpaceColumnMajorCSV(ifstream &file)
{
    allocateSpaceRowMajorCSV(file);
}

void Block4in16Sparse::loadDataRowMajorCSV(ifstream &file)
{
    // variables for loading 16 rows at a time
    string rows[16];
    stringstream rowStreams[16];
    float *floatMatrices[4] = { 
                                _floatMatrix, 
                                _floatMatrix +  _columns + _columns / sizeof(float),
                                _floatMatrix + (_columns + _columns / sizeof(float)) * 2,
                                _floatMatrix + (_columns + _columns / sizeof(float)) * 3
                              };
    uint8_t *byteMatrices[4] = { 
                                 _uint8Matrix + 16 * sizeof(float), 
                                 _uint8Matrix +  _columns * sizeof(float) + _columns      + 16 * sizeof(float),
                                 _uint8Matrix + (_columns * sizeof(float) + _columns) * 2 + 16 * sizeof(float),
                                 _uint8Matrix + (_columns * sizeof(float) + _columns) * 3 + 16 * sizeof(float)
                               };
    
    string cell;
    uint8_t indices[4];
    for (uint16_t i = 0; i < _rows >> 4; i++)
    {
        uint8_t row = 0;
        while (row < 16 && getline(file, rows[row]))
        {
            row++;
        }

        for ( ; row < 16; row++) // pad with empty rows
        {
            rows[row] = "";
        }

        for (uint8_t i = 0; i < 16; i++)
        {
            rowStreams[i] = stringstream(rows[i]);
        }

        for (uint16_t j = 0; j < _columns >> 2; j++)
        {
            for (uint8_t k = 0; k < 4; k++)
            {
                indices[k] = 0;
            }

            for (uint8_t k = 0; k < 16; k++)
            {
                for (uint8_t l = 0; l < 4; l++)
                {
                    if (getline(rowStreams[k], cell, ','))
                    {
                        float value = stof(cell);
                        if (value != 0.0f)
                        {
                            if (indices[l] >= 4)
                            {
                                throw invalid_argument("Sparse matrix cannot have more than 4 non-zero values per block 16.");
                            }

                            floatMatrices[indices[l]][l] = value;
                            byteMatrices[indices[l]++][l] = k;
                        }
                    }
                    else if (indices[l] < 4)
                    {
                        floatMatrices[indices[l]][l] = 0.0f;
                        byteMatrices[indices[l]++][l] = k;
                    }
                }
            }

            for (uint8_t k = 0; k < 4; k++)
            {
                floatMatrices[k] += 4;
                byteMatrices[k] += 4;
            }

            if ((j & 3) == 3)
            {
                for (uint8_t k = 0; k < 4; k++)
                {
                    floatMatrices[k] += 16 / sizeof(float);
                    byteMatrices[k] += 16 * sizeof(float);
                }
            }
        }
        
        for (uint8_t i = 0; i < 4; i++)
        {
            floatMatrices[i] = floatMatrices[i] + (_columns + _columns / sizeof(float)) * 3;
            byteMatrices[i] = byteMatrices[i] + (_columns * sizeof(float) + _columns) * 3;
        } 
    }

}

void Block4in16Sparse::loadDataColumnMajorCSV(ifstream &file)
{
    (void)file;
}

void Block4in16Sparse::loadBinary(std::string fileName)
{
    (void)fileName;
}

void Block4in16Sparse::printColumn(uint16_t columnIndex, uint8_t precision)
{
    (void)columnIndex;
    (void)precision;
}

void Block4in16Sparse::printRow(uint16_t rowIndex, uint8_t precision)
{
    uint8_t blockIndex = rowIndex & 15;
    rowIndex >>= 4;
    rowIndex <<= 2;

    float *row[4] = {
                       _floatMatrix + rowIndex       * (_columns + _columns / sizeof(float)),
                       _floatMatrix + (rowIndex + 1) * (_columns + _columns / sizeof(float)),
                       _floatMatrix + (rowIndex + 2) * (_columns + _columns / sizeof(float)),
                       _floatMatrix + (rowIndex + 3) * (_columns + _columns / sizeof(float))
                    };
    uint8_t *indices[4] = {
                            _uint8Matrix + rowIndex       * (_columns * sizeof(float) + _columns) + 16 * sizeof(float),
                            _uint8Matrix + (rowIndex + 1) * (_columns * sizeof(float) + _columns) + 16 * sizeof(float),
                            _uint8Matrix + (rowIndex + 2) * (_columns * sizeof(float) + _columns) + 16 * sizeof(float),
                            _uint8Matrix + (rowIndex + 3) * (_columns * sizeof(float) + _columns) + 16 * sizeof(float)
                          };

    cout << setprecision(precision) << fixed;
    for (uint16_t i = 0; i < (_columns >> 4) - 1; i++)
    {
        for (uint8_t j = 0; j < 16; j++)
        {
            for (uint8_t k = 0; k < 4; k++)
            {
                if (indices[k][j] == blockIndex)
                {
                    cout << setw(precision + 3) << row[k][j] << ",";
                    goto printed1;
                }
            }
            cout << setw(precision + 3) << 0.0f << ",";
        printed1:
            continue;
        }

        for (uint8_t j = 0; j < 4; j++)
        {
            row[j] += 16 + 16 / sizeof(float);
            indices[j] += 16 * sizeof(float) + 16;
        }
    }

    for (uint8_t j = 0; j < 15; j++)
    {
        for (uint8_t k = 0; k < 4; k++)
        {
            if (indices[k][j] == blockIndex)
            {
                cout << setw(precision + 3) << row[k][j] << ",";
                goto printed2;
            }
        }
        cout << setw(precision + 3) << 0.0f << ",";
    printed2:
        continue;
    }

    for (uint8_t k = 0; k < 4; k++)
    {
        if (indices[k][15] == blockIndex)
        {
            cout << setw(precision + 3) << row[k][15] << endl;
            return;
        }
    }
    cout << setw(precision + 3) << 0.0f << endl;
}

void Block4in16Sparse::printCompressed(uint8_t precision)
{
    for (uint16_t i = 0; i < _rows >> 2; i++)
    {
        float *row = _floatMatrix + i * (_columns + _columns / sizeof(float));

        cout << setprecision(precision) << fixed;
        for (uint16_t i = 0; i < (_columns >> 4) - 1; i++)
        {
            for (uint8_t j = 0; j < 16; j++)
            {
                cout << setw(precision + 3) << row[j] << ",";
            }
            row += 16 + 16 / sizeof(float);
        }

        for (uint8_t j = 0; j < 15; j++)
        {
            cout << setw(precision + 3) << row[j] << ",";
        }
        cout << setw(precision + 3) << row[_columns - 1] << endl;
    }
}

void Block4in16Sparse::saveAsBinary(string fileName)
{
    (void)fileName;
}

void Block4in16Sparse::saveAsCSV(string fileName)
{
    (void)fileName;
}
