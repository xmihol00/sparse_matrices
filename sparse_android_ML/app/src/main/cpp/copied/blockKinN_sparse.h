#ifndef BLOCKKINN_SPARSE_H
#define BLOCKKINN_SPARSE_H

#include <thread>

#include "sparse.h"
#if PHONE
#include <arm_neon.h>
#else
#include "arm_neon_.h"
#endif

namespace Matrix 
{
    template <uint8_t K, uint8_t N, uint16_t denseRows, uint16_t denseColumns, DimensionMajorityEnum dimMajority>
    class BlockKinNSparse : public Sparse
    {
        private:
            const uint16_t _compressedDimension;
            const uint16_t _alignedColumns;

            void allocateSpaceRowMajorCSV(std::ifstream &file)
            {
                (void)file;
                using namespace std;

                _size = (_columns + (16 - (_columns & 15)) * ((_columns & 15) != 0)) * _compressedDimension * (sizeof(float) + sizeof(uint8_t));
                _byteMatrix = new(align_val_t{16}) byte[_size](); // allocate aligned memory to 16 bytes to allow AVX/NEON instructions
            }

            void allocateSpaceColumnMajorCSV(std::ifstream &file)
            {
                (void)file;
                using namespace std;

                _size = (_rows + (16 - (_rows & 15)) * ((_rows & 15) != 0)) * _compressedDimension * (sizeof(float) + sizeof(uint8_t));
                _byteMatrix = new(align_val_t{16}) byte[_size](); // allocate aligned memory to 16 bytes to allow AVX/NEON instructions
            }

            void loadDataRowMajorCSV(std::ifstream &file)
            {
                using namespace std;

                // variables for loading 16 rows at a time
                string rows[N];
                stringstream rowStreams[N];

                float *floatMatrices[K];
                uint8_t *byteMatrices[K];

                string cell;
                uint8_t indices[4];
                for (uint16_t i = 0; i < (_rows + N - 1) / N; i++)
                {
                    floatMatrices[0] = _floatMatrix + i * K * (_alignedColumns + _alignedColumns / sizeof(float));
                    for (uint8_t j = 1; j < K; j++)
                    {
                        floatMatrices[j] = floatMatrices[j - 1] + _alignedColumns + _alignedColumns / sizeof(float);
                    }

                    byteMatrices[0] = _uint8Matrix + i * K * (_alignedColumns * sizeof(float) + _alignedColumns) + 16 * sizeof(float);
                    for (uint8_t j = 1; j < K; j++)
                    {
                        byteMatrices[j] = byteMatrices[j - 1] + _alignedColumns * sizeof(float) + _alignedColumns;
                    }

                    uint8_t row = 0;
                    while (row < N && getline(file, rows[row]))
                    {
                        row++;
                    }

                    for ( ; row < N; row++) // pad with empty rows
                    {
                        rows[row] = "";
                    }

                    for (uint8_t j = 0; j < N; j++)
                    {
                        rowStreams[j] = stringstream(rows[j]);
                    }

                    for (uint16_t j = 0; j < _columns >> 2; j++)
                    {
                        for (uint8_t k = 0; k < 4; k++)
                        {
                            indices[k] = 0;
                        }

                        for (uint8_t k = 0; k < N; k++)
                        {
                            for (uint8_t l = 0; l < 4; l++)
                            {
                                if (getline(rowStreams[k], cell, ','))
                                {
                                    float value = stof(cell);
                                    if (value != 0.0f)
                                    {
                                        if (indices[l] >= K)
                                        {
                                            throw invalid_argument("Sparse matrix cannot have more than " + to_string(K) + 
                                                                   " non-zero values per block of " + to_string(N) + " rows.");
                                        }

                                        floatMatrices[indices[l]][l] = value;
                                        byteMatrices[indices[l]++][l] = k;
                                    }
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
                }
            }

            void loadDataColumnMajorCSV(std::ifstream &file)
            {
                (void)file;
            }

            void loadBinary(std::string fileName)
            {
                (void)fileName;
            }

            void dotPart(Dense &operandMatrix, Dense &targetMatrix, uint8_t threadCount, uint8_t threadId)
            {
                float accumulators[N] = { 0.0f, };

                if (_dimMajority == ROW_MAJOR)
                {
                    const uint16_t rowBlocks = _compressedDimension / K / threadCount;
                    const uint32_t operandOffset = _compressedDimension / threadCount * (_columns + _columns / sizeof(float)) * threadId;
                    const uint16_t targetOffset = targetMatrix._rows / threadCount * threadId;
                    const int16_t targetPadding = denseRows - _compressedDimension / K * N + targetMatrix._rows - targetMatrix._rows / threadCount;
                    const uint16_t columnBlocks = denseColumns >> 4;

                    float *target = targetMatrix._floatMatrix + targetOffset;
                    float *column = operandMatrix._floatMatrix;
                    float *offsetedRow = _floatMatrix + operandOffset;
                    uint8_t *offsetedIndices = _uint8Matrix + (operandOffset + 16) * sizeof(float);

                    if (operandMatrix._dimMajority == COLUMN_MAJOR)
                    {
                        for (uint16_t i = 0; i < operandMatrix._columns; i++)
                        {
                            float *row = offsetedRow;
                            uint8_t *indices = offsetedIndices;
                            
                            for (uint16_t j = 0; j < rowBlocks; j++)
                            {
                                for (uint8_t k = 0; k < K; k++)
                                {
                                    #pragma GCC unroll 16
                                    for (uint16_t l = 0; l < columnBlocks; l++)
                                    {
                                        #pragma GCC unroll 4
                                        for (uint8_t m = 0; m < 4; m++)
                                        {
                                            float32x4_t a = vld1q_f32(row);
                                            float32x4_t b = vld1q_f32(column);
                                            a = vmulq_f32(a, b);

                                            #pragma GCC unroll 4
                                            for (uint8_t n = 0; n < 4; n++)
                                            {
                                                accumulators[indices[n]] += a[n];
                                            }

                                            row += 4;
                                            indices += 4;
                                            column += 4;
                                        }

                                        row += 16 / sizeof(float);
                                        indices += 16 * sizeof(float);
                                    }
                                    column -= operandMatrix._rows;
                                }

                                #pragma GCC ivdep
                                for (uint8_t k = 0; k < N; k++)
                                {
                                    target[k] = accumulators[k];
                                    accumulators[k] = 0.0f;
                                }
                                target += N;
                            }
                            target += targetPadding;
                            column += operandMatrix._rows;
                        }
                    }
                }
            }
        
        public:
            BlockKinNSparse() : Sparse(), _compressedDimension{0}, _alignedColumns{0} {}

            BlockKinNSparse(std::string fileName) : Sparse(denseRows, denseColumns, dimMajority), 
                                                    _compressedDimension{dimMajority == COLUMN_MAJOR ? 
                                                                         ((denseColumns + N - 1) / N) * K : 
                                                                         ((denseRows + N - 1) / N) * K},
                                                    _alignedColumns{static_cast<uint16_t>(denseColumns + (16 - (denseColumns & 15)) * ((_columns & 15) != 0))}
            {
                using namespace std;

                static_assert(dimMajority != FILE_DETERMINED, "dimension majority must be specified");
                static_assert(dimMajority != COLUMN_MAJOR || !(denseRows & 3), "rows must be a multiple of 4");
                static_assert(dimMajority != ROW_MAJOR || !(denseColumns & 3), "columns must be a multiple of 4");
                
                loadCSV(fileName);
            }
            ~BlockKinNSparse() = default;

            BlockKinNSparse<K, N, denseRows, denseColumns, dimMajority> &operator=(BlockKinNSparse<K, N, denseRows, denseColumns, dimMajority>&& other)
            {
                if (this != &other)
                {
                    _floatMatrix = std::move(other._floatMatrix);
                    const_cast<uint16_t&>(_compressedDimension) = std::move(other._compressedDimension);
                    const_cast<uint16_t&>(_alignedColumns) = std::move(other._alignedColumns);
                    
                    _size = std::move(other._size);
                    _rows = std::move(other._rows);
                    _columns = std::move(other._columns);
                    _dimMajority = std::move(other._dimMajority);

                    other._floatMatrix = nullptr;
                }
                return *this;
            }

            void printCompressed(uint8_t precision = 7)
            {
                using namespace std;

                for (uint16_t i = 0; i < _compressedDimension; i++)
                {
                    float *row = _floatMatrix + i * (_alignedColumns + _alignedColumns / sizeof(float));

                    cout << setprecision(precision) << fixed;
                    uint16_t j = 0;
                    for ( ; j < (_alignedColumns >> 4) - 1; j++)
                    {
                        for (uint8_t k = 0; k < 16; k++)
                        {
                            cout << setw(precision + 3) << row[k] << ",";
                        }
                        row += 16 + 16 / sizeof(float);
                    }

                    uint8_t l = 0;
                    for (uint16_t k = j << 4; k < _columns - 1; k++, l++)
                    {
                        cout << setw(precision + 3) << row[l] << ",";
                    }
                    cout << setw(precision + 3) << row[l] << endl;
                }
            }

            void printColumn(uint16_t columnIndex, uint8_t precision = 7)
            {
                (void)columnIndex;
                (void)precision;
            }

            void printRow(uint16_t rowIndex, uint8_t precision = 7)
            {
                using namespace std;

                uint8_t blockIndex = rowIndex % N;
                rowIndex /= N;
                rowIndex *= K;

                float *row[K];
                for (uint8_t i = 0; i < K; i++)
                {
                    row[i] = _floatMatrix + (rowIndex + i) * (_alignedColumns + _alignedColumns / sizeof(float));
                }

                uint8_t *indices[K];
                for (uint8_t i = 0; i < K; i++)
                {
                    indices[i] = _uint8Matrix + (rowIndex + i) * (_alignedColumns * sizeof(float) + _alignedColumns) + 16 * sizeof(float);
                }

                cout << setprecision(precision) << fixed;
                for (uint16_t i = 0; i < (_alignedColumns >> 4) - 1; i++)
                {
                    for (uint8_t j = 0; j < 16; j++)
                    {
                        for (uint8_t k = 0; k < K; k++)
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

                    for (uint8_t j = 0; j < K; j++)
                    {
                        row[j] += 16 + 16 / sizeof(float);
                        indices[j] += 16 * sizeof(float) + 16;
                    }
                }

                uint8_t lastIndex = (_columns & 15) + 16 * ((_columns & 15) == 0) - 1;
                for (uint8_t j = 0; j < lastIndex; j++)
                {
                    for (uint8_t k = 0; k < K; k++)
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

                for (uint8_t k = 0; k < K; k++)
                {
                    if (indices[k][lastIndex] == blockIndex)
                    {
                        cout << setw(precision + 3) << row[k][lastIndex] << endl;
                        return;
                    }
                }
                cout << setw(precision + 3) << 0.0f << endl;
            }

            virtual void saveAsBinary(std::string fileName)
            {
                (void)fileName;
            }

            virtual void saveAsCSV(std::string fileName)
            {
                (void)fileName;
            }

            void dot(Dense &operandMatrix, Dense &targetMatrix)
            {
                dotPart(operandMatrix, targetMatrix, 1, 0);
            }
            
            Dense dot(Dense &operandMatrix)
            {
                Dense targetMatrix(_rows, operandMatrix._columns, COLUMN_MAJOR);
                dot(operandMatrix, targetMatrix);

                return targetMatrix;
            }

            void dotThreads(Dense &operandMatrix, Dense &targetMatrix)
            {
                using namespace std;
                
                const uint8_t numThreads = 8;
                thread threads[numThreads];
                for (uint8_t i = 0; i < numThreads; i++)
                {
                    threads[i] = thread(&BlockKinNSparse::dotPart, this, ref(operandMatrix), ref(targetMatrix), numThreads, i);
                }

                for (uint8_t i = 0; i < numThreads; i++)
                {
                    threads[i].join();
                }
            }

            Dense dotThreads(Dense &operandMatrix)
            {
                Dense targetMatrix(_rows, operandMatrix._columns, COLUMN_MAJOR);
                dotThreads(operandMatrix, targetMatrix);

                return targetMatrix;
            }

            template <float (*activationFunction)(float)>
            void dotAddActivate(Dense &dotMatrix, Dense &addVector, Dense &targetMatrix, uint8_t threadCount = 1, uint8_t threadId = 0)
            {
                float accumulators[N] = { 0.0f, };

                if (_dimMajority == ROW_MAJOR)
                {
                    const uint16_t rowBlocks = _compressedDimension / K / threadCount;
                    const uint32_t operandOffset = _compressedDimension / threadCount * (_columns + _columns / sizeof(float)) * threadId;
                    const uint16_t targetOffset = targetMatrix._rows / threadCount * threadId;
                    const int16_t targetPadding = denseRows - _compressedDimension / K * N + targetMatrix._rows - targetMatrix._rows / threadCount;
                    const uint16_t columnBlocks = denseColumns >> 4;

                    float *target = targetMatrix._floatMatrix + targetOffset;
                    float *dotColumn = dotMatrix._floatMatrix;
                    float *addColumnOffseted = addVector._floatMatrix + N * threadId;
                    float *addColumn = addColumnOffseted;
                    float *offsetedRow = _floatMatrix + operandOffset;
                    uint8_t *offsetedIndices = _uint8Matrix + (operandOffset + 16) * sizeof(float);

                    if (dotMatrix._dimMajority == COLUMN_MAJOR && addVector._dimMajority == COLUMN_MAJOR)
                    {
                        for (uint16_t i = 0; i < dotMatrix._columns; i++)
                        {
                            float *row = offsetedRow;
                            uint8_t *indices = offsetedIndices;
                            
                            for (uint16_t j = 0; j < rowBlocks; j++)
                            {
                                for (uint8_t k = 0; k < K; k++)
                                {
                                    #pragma GCC unroll 16
                                    for (uint16_t l = 0; l < columnBlocks; l++)
                                    {
                                        #pragma GCC unroll 4
                                        for (uint8_t m = 0; m < 4; m++)
                                        {
                                            float32x4_t a = vld1q_f32(row);
                                            float32x4_t b = vld1q_f32(dotColumn);
                                            a = vmulq_f32(a, b);

                                            #pragma GCC unroll 4
                                            for (uint8_t n = 0; n < 4; n++)
                                            {
                                                accumulators[indices[n]] += a[n];
                                            }

                                            row += 4;
                                            indices += 4;
                                            dotColumn += 4;
                                        }

                                        row += 16 / sizeof(float);
                                        indices += 16 * sizeof(float);
                                    }
                                    dotColumn -= dotMatrix._rows;
                                }

                                #pragma GCC ivdep
                                for (uint8_t k = 0; k < N; k++)
                                {
                                    target[k] = activationFunction(accumulators[k] + addColumn[k]);
                                    accumulators[k] = 0.0f;
                                }
                                target += N;
                                addColumn += N;
                            }

                            addColumn = addColumnOffseted;
                            target += targetPadding;
                            dotColumn += dotMatrix._rows;
                        }
                    }
                }
            }
                        
            template <float (*activationFunction)(float)>
            Dense dotAddActivate(Dense &dotMatrix, Dense &addVector)
            {
                Dense targetMatrix(_rows, dotMatrix._columns, COLUMN_MAJOR);
                dotAddActivate<activationFunction>(dotMatrix, addVector, targetMatrix);

                return targetMatrix;
            }
    };
}

#endif