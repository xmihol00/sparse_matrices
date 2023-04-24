
/*template <uint8_t K, uint8_t N, uint16_t denseRows, uint16_t denseColumns, DimensionMajorityEnum dimMajority>
BlockKinMSparse<K, N, denseRows, denseColumns, dimMajority>::BlockKinMSparse(std::string fileName) : Sparse(dimMajority)
{
    static_assert(dimMajority != FILE_DETERMINED, "dimension majority must be specified");
    static_assert(dimMajority != COLUMN_MAJOR || !(denseRows & 3), "rows must be a multiple of 4");
    static_assert(dimMajority != ROW_MAJOR || !(denseColumns & 3), "columns must be a multiple of 4");

    if (dimMajority == COLUMN_MAJOR)
    {
        compressedDimension = (denseColumns + N - 1) / N * K;
    }
    else if (dimMajority == ROW_MAJOR)
    {
        compressedDimension = (denseRows + N - 1) / N * K;
    }

    loadCSV(fileName);
}

template <uint8_t K, uint8_t N, uint16_t denseRows, uint16_t denseColumns, DimensionMajorityEnum dimMajority>
void BlockKinMSparse<K, N, denseRows, denseColumns, dimMajority>::allocateSpaceRowMajorCSV(ifstream &file)
{
    _size = (_columns + (16 - (_columns & 15)) * ((_columns & 15) != 0)) * compressedDimension * (sizeof(float) + sizeof(uint8_t));
    _byteMatrix = new(align_val_t{16}) byte[_size](); // allocate aligned memory to 16 bytes to allow AVX/NEON instructions
}

template <uint8_t K, uint8_t N, uint16_t denseRows, uint16_t denseColumns, DimensionMajorityEnum dimMajority>
void BlockKinMSparse<K, N, denseRows, denseColumns, dimMajority>::allocateSpaceColumnMajorCSV(ifstream &file)
{
    _size = (_rows + (16 - (_rows & 15)) * ((_rows & 15) != 0)) * compressedDimension * (sizeof(float) + sizeof(uint8_t));
    _byteMatrix = new(align_val_t{16}) byte[_size](); // allocate aligned memory to 16 bytes to allow AVX/NEON instructions
}

template <uint8_t K, uint8_t N, uint16_t denseRows, uint16_t denseColumns, DimensionMajorityEnum dimMajority>
void BlockKinMSparse<K, N, denseRows, denseColumns, dimMajority>::loadDataRowMajorCSV(ifstream &file)
{
    // variables for loading 16 rows at a time
    string rows[N];
    stringstream rowStreams[N];

    float *floatMatrices[K];
    floatMatrices[0] = _floatMatrix;
    for (uint8_t i = 1; i < K; i++)
    {
        floatMatrices[i] = floatMatrices[i] + (_columns + _columns / sizeof(float));
    }

    uint8_t *byteMatrices[K];
    byteMatrices[0] = _uint8Matrix + 16 * sizeof(float);
    for (uint8_t i = 1; i < K; i++)
    {
        byteMatrices[i] = byteMatrices[i] + _columns * sizeof(float) + _columns;
    }
    
    string cell;
    uint8_t indices[K];
    for (uint16_t i = 0; i < _rows >> 4; i++)
    {
        uint8_t row = 0;
        while (row < N && getline(file, rows[row]))
        {
            row++;
        }

        for ( ; row < N; row++) // pad with empty rows
        {
            rows[row] = "";
        }

        for (uint8_t i = 0; i < N; i++)
        {
            rowStreams[i] = stringstream(rows[i]);
        }

        for (uint16_t j = 0; j < _columns >> 2; j++)
        {
            for (uint8_t k = 0; k < K; k++)
            {
                indices[k] = 0;
            }

            for (uint8_t k = 0; k < N; k++)
            {
                for (uint8_t l = 0; l < K; l++)
                {
                    if (getline(rowStreams[k], cell, ','))
                    {
                        float value = stof(cell);
                        if (value != 0.0f)
                        {
                            if (indices[l] >= K)
                            {
                                throw invalid_argument("Sparse matrix cannot have more than 4 non-zero values per block 16.");
                            }

                            floatMatrices[indices[l]][l] = value;
                            byteMatrices[indices[l]++][l] = k;
                        }
                    }
                    else if (indices[l] < K)
                    {
                        floatMatrices[indices[l]][l] = 0.0f;
                        byteMatrices[indices[l]++][l] = k;
                    }
                }
            }

            for (uint8_t k = 0; k < K; k++)
            {
                floatMatrices[k] += 4;
                byteMatrices[k] += 4;
            }

            if ((j & 3) == 3)
            {
                for (uint8_t k = 0; k < K; k++)
                {
                    floatMatrices[k] += 16 / sizeof(float);
                    byteMatrices[k] += 16 * sizeof(float);
                }
            }
        }
        
        for (uint8_t i = 0; i < K; i++)
        {
            floatMatrices[i] = floatMatrices[i] + (_columns + _columns / sizeof(float)) * (K - 1);
            byteMatrices[i] = byteMatrices[i] + (_columns * sizeof(float) + _columns) * (K - 1);
        } 
    }
}

void loadDataColumnMajorCSV(ifstream &file)
{
    (void)file;
}*/

