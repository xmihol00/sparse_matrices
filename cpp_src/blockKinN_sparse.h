#ifndef BLOCKKINN_SPARSE_H
#define BLOCKKINN_SPARSE_H

#include <thread>

#include "sparse.h"
#include "arm_neon_.h"

namespace Matrix 
{
    template <uint8_t K, uint8_t N, uint16_t denseRows, uint16_t denseColumns, DimensionMajorityEnum dimMajority>
    class BlockKinNSparse : public Sparse
    {
        private:
            const uint16_t _compressedDimension; // number of rows/columns in the compressed matrix
            const uint16_t _alignedColumns;      // columns padded to multiple of 16
            bool _metadataFirst = false;
            uint16_t _floatOffset; // zero when _metadataFirst is false
            uint16_t _byteOffset;  // zero when _metadataFirst is true

            void allocateSpaceRowMajorCSV(std::ifstream &file)
            {
                (void)file; // suppress unused parameter warning, no need to read the file
                using namespace std;

                _size = (_columns + (16 - (_columns & 15)) * ((_columns & 15) != 0)) * // pad columns to multiple of 16
                        _compressedDimension * (sizeof(float) + sizeof(uint8_t));
                _byteMatrix = new(align_val_t{16}) byte[_size](); // allocate aligned memory to 16 bytes to allow AVX/NEON instructions
            }

            void allocateSpaceColumnMajorCSV(std::ifstream &file)
            {
                (void)file; // suppress unused parameter warning, no need to read the file
                using namespace std;

                _size = (_rows + (16 - (_rows & 15)) * ((_rows & 15) != 0)) * // pad rows to multiple of 16
                        _compressedDimension * (sizeof(float) + sizeof(uint8_t));
                _byteMatrix = new(align_val_t{16}) byte[_size](); // allocate aligned memory to 16 bytes to allow AVX/NEON instructions
            }

            void loadDataRowMajorCSV(std::ifstream &file)
            {
                using namespace std;

                // variables for loading N rows at a time
                string rows[N];
                stringstream rowStreams[N];

                // pointers to the start of each block in contiguous memory
                float *floatMatrices[K];
                uint8_t *byteMatrices[K];

                string cell;
                uint8_t indices[4];
                for (uint16_t i = 0; i < (_rows + N - 1) / N; i++)
                {
                    // initialize pointers to the start of each block
                    floatMatrices[0] = _floatMatrix + i * K * (_alignedColumns + _alignedColumns / sizeof(float)) + _floatOffset;
                    for (uint8_t j = 1; j < K; j++) // add offsets from the starting address
                    {
                        floatMatrices[j] = floatMatrices[j - 1] + _alignedColumns + _alignedColumns / sizeof(float);
                    }
                    byteMatrices[0] = _uint8Matrix + i * K * (_alignedColumns + _alignedColumns * sizeof(float)) + _byteOffset;
                    for (uint8_t j = 1; j < K; j++) // add offsets from the starting address
                    {
                        byteMatrices[j] = byteMatrices[j - 1] + _alignedColumns + _alignedColumns * sizeof(float);
                    }

                    uint8_t row = 0;
                    while (row < N && getline(file, rows[row])) // read N rows
                    {
                        row++;
                    }

                    for ( ; row < N; row++) // pad with empty rows at the end of the matrix if necessary
                    {
                        rows[row] = "";
                    }

                    for (uint8_t j = 0; j < N; j++)
                    {
                        rowStreams[j] = stringstream(rows[j]);
                    }

                    for (uint16_t j = 0; j < _columns >> 2; j++) // process columns in batches of 4
                    {
                        for (uint8_t k = 0; k < 4; k++) // reset indices
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
                                    if (value != 0.0f) // load non-zero values
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

                        for (uint8_t k = 0; k < K; k++) // increment pointers
                        {
                            floatMatrices[k] += 4;
                            byteMatrices[k] += 4;
                        }
                        
                        if ((j & 3) == 3) // modulo 4
                        {
                            for (uint8_t k = 0; k < K; k++) // skip indices in case of float pointers and skip non-zero values in case of byte pointers
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

            void dotPart(Dense &operandMatrix, Dense &targetMatrix, uint8_t numberOfThreads, uint8_t threadId)
            {
                float accumulators[N] = { 0.0f, };

                if (_dimMajority == ROW_MAJOR)
                {
                    const uint16_t rowBlocks = _compressedDimension / K / numberOfThreads; // number of rows processed by each thread
                    const uint32_t operandOffset = _compressedDimension / numberOfThreads * // offset of each thread in the first operand matrix
                                                   (_columns + _columns / sizeof(float)) * threadId;
                    const uint16_t targetOffset = targetMatrix._rows / numberOfThreads * threadId; // offset of each thread in the target matrix
                    const int16_t targetShift = denseRows - _compressedDimension / K * N + // target matrix pointer shift after each iteration
                                                targetMatrix._rows - targetMatrix._rows / numberOfThreads;
                    const uint16_t columnBlocks = denseColumns >> 4; // number of column blocks in the dense matrix

                    float *target = targetMatrix._floatMatrix + targetOffset;
                    float *column = operandMatrix._floatMatrix;
                    float *offsetedRow = _floatMatrix + operandOffset + _floatOffset;
                    uint8_t *offsetedIndices = _uint8Matrix + operandOffset * sizeof(float) + _byteOffset;

                    if (operandMatrix._dimMajority == COLUMN_MAJOR)
                    {
                        for (uint16_t i = 0; i < operandMatrix._columns; i++) // all columns of 2nd operand matrix
                        {
                            float *row = offsetedRow;
                            uint8_t *indices = offsetedIndices;
                            
                            for (uint16_t j = 0; j < rowBlocks; j++) // all rowBlock of a given thread
                            {
                                for (uint8_t k = 0; k < K; k++) // row block has K rows
                                {
                                    #pragma GCC unroll 16
                                    for (uint16_t l = 0; l < columnBlocks; l++) // column block are batches of 16 non-zero values
                                    {
                                        #pragma GCC unroll 4
                                        for (uint8_t m = 0; m < 4; m++) // use of 4 floats SIMD instructions
                                        {
                                            float32x4_t a = vld1q_f32(row);
                                            float32x4_t b = vld1q_f32(column);
                                            a = vmulq_f32(a, b);

                                            #pragma GCC unroll 4
                                            for (uint8_t n = 0; n < 4; n++)
                                            {
                                                accumulators[indices[n]] += a[n]; // store values to 1 of N accumulators
                                            }

                                            // move pointers
                                            row += 4;
                                            indices += 4;
                                            column += 4;
                                        }

                                        row += 16 / sizeof(float);     // skip indices
                                        indices += 16 * sizeof(float); // skip non-zero values
                                    }
                                    column -= operandMatrix._rows; // reset column pointer, so the column is processed by all rows in the block
                                }

                                #pragma GCC ivdep
                                for (uint8_t k = 0; k < N; k++) // store N results to the target matrix
                                {
                                    target[k] = accumulators[k];
                                    accumulators[k] = 0.0f;
                                }
                                target += N; // move pointer
                            }
                            
                            target += targetShift;         // move target pointer to the next area computed by the same thread
                            column += operandMatrix._rows; // move column pointer to the next column
                        }
                    }
                }
            }
        
        public:
            BlockKinNSparse() : Sparse(), _compressedDimension{0}, _alignedColumns{0} {}

            BlockKinNSparse(std::string fileName, bool metadataFirst = false)
                : Sparse(denseRows, denseColumns, dimMajority), 
                  _compressedDimension{dimMajority == COLUMN_MAJOR ? 
                                       ((denseColumns + N - 1) / N) * K : // ceil(denseColumns / N) * K
                                       ((denseRows + N - 1) / N) * K},    // ceil(denseRows / N) * K
                  _alignedColumns{static_cast<uint16_t>(denseColumns + (16 - (denseColumns & 15)) * ((_columns & 15) != 0))},
                  _metadataFirst{metadataFirst},
                  _floatOffset{metadataFirst ? static_cast<uint16_t>(16 / sizeof(float)) : static_cast<uint16_t>(0U)},
                  _byteOffset{metadataFirst ? static_cast<uint16_t>(0U) : static_cast<uint16_t>(16 * sizeof(float))}
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
                    _floatOffset = std::move(other._floatOffset);
                    _byteOffset = std::move(other._byteOffset);

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
                    for ( ; j < (_alignedColumns >> 4) - 1; j++) // all block apart from the last one
                    {
                        for (uint8_t k = 0; k < 16; k++)
                        {
                            cout << setw(precision + 3) << row[k] << ",";
                        }
                        row += 16 + 16 / sizeof(float);
                    }

                    uint8_t l = 0;
                    for (uint16_t k = j << 4; k < _columns - 1; k++, l++) // all values in the last block apart from the last one
                    {
                        cout << setw(precision + 3) << row[l] << ",";
                    }
                    cout << setw(precision + 3) << row[l] << endl; // last value in the last block
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
                // adjust the index to the compressed matrix
                rowIndex /= N;
                rowIndex *= K;
                
                // the requested row can have values in a whole block, load pointer to rows of the block
                float *row[K];
                for (uint8_t i = 0; i < K; i++)
                {
                    row[i] = _floatMatrix + (rowIndex + i) * (_alignedColumns + _alignedColumns / sizeof(float)) + _floatOffset;
                }
                uint8_t *indices[K];
                for (uint8_t i = 0; i < K; i++)
                {
                    indices[i] = _uint8Matrix + (rowIndex + i) * (_alignedColumns * sizeof(float) + _alignedColumns) + _byteOffset;
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
                        cout << setw(precision + 3) << 0.0f << ","; // value was not found, therefore it is 0
                    printed1:
                        continue;
                    }

                    for (uint8_t j = 0; j < K; j++) // move pointers to the next block of column of the block of rows
                    {
                        row[j] += 16 + 16 / sizeof(float);     // skip indices
                        indices[j] += 16 + 16 * sizeof(float); // skip non-zero values
                    }
                }

                uint8_t lastIndex = (_columns & 15) + 16 * ((_columns & 15) == 0) - 1; // last index of a column block
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
                    cout << setw(precision + 3) << 0.0f << ","; // value was not found, therefore it is 0
                printed2:
                    continue;
                }

                for (uint8_t k = 0; k < K; k++) // last value in the column block
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
                for (uint8_t i = 0; i < numThreads; i++) // spawn 8 threads
                {
                    threads[i] = thread(&BlockKinNSparse::dotPart, this, ref(operandMatrix), ref(targetMatrix), numThreads, i);
                }

                for (uint8_t i = 0; i < numThreads; i++) // wait for all threads to finish
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
            void dotAddActivateRowThread(Dense &dotMatrix, Dense &addVector, Dense &targetMatrix, uint8_t numberOfThreads = 1, uint8_t threadId = 0)
            {
                float accumulators[N] = { 0.0f, };

                if (_dimMajority == ROW_MAJOR)
                {
                    const uint16_t rowBlocks = _compressedDimension / K / numberOfThreads;
                    const uint32_t operandOffset = _compressedDimension / numberOfThreads * (_columns + _columns / sizeof(float)) * threadId;
                    const uint16_t targetOffset = targetMatrix._rows / numberOfThreads * threadId;
                    const int16_t targetShift = denseRows - _compressedDimension / K * N + targetMatrix._rows - targetMatrix._rows / numberOfThreads;
                    const uint16_t columnBlocks = denseColumns >> 4;

                    float *target = targetMatrix._floatMatrix + targetOffset;
                    float *dotColumn = dotMatrix._floatMatrix;
                    float *addColumnOffseted = addVector._floatMatrix + N * threadId;
                    float *addColumn = addColumnOffseted;
                    float *offsetedRow = _floatMatrix + operandOffset + _floatOffset;
                    uint8_t *offsetedIndices = _uint8Matrix + operandOffset * sizeof(float) + _byteOffset;

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
                            target += targetShift;
                            dotColumn += dotMatrix._rows;
                        }
                    }
                }
            }

            template <float (*activationFunction)(float)>
            void dotAddActivateColumnThread(Dense &dotMatrix, Dense &addVector, Dense &targetMatrix, uint8_t numberOfThreads = 1, uint8_t threadId = 0)
            {
                float accumulators[N] = { 0.0f, };

                if (_dimMajority == ROW_MAJOR)
                {
                    const uint16_t rowBlocks = _compressedDimension / K; // each thread computes all rows
                    const uint16_t columnBlocks = denseColumns >> 4; // number of block of 16 columns     

                    float *dotColumn = dotMatrix._floatMatrix + dotMatrix._columns * // each thread computes part of the columns
                                       threadId * dotMatrix._rows / numberOfThreads;
                    float *target = targetMatrix._floatMatrix + targetMatrix._columns * // same offset as with columns
                                    threadId * targetMatrix._rows / numberOfThreads;
                    float *addColumn = addVector._floatMatrix; // biases

                    if (dotMatrix._dimMajority == COLUMN_MAJOR && addVector._dimMajority == COLUMN_MAJOR)
                    {
                        for (uint16_t i = 0; i < dotMatrix._columns / numberOfThreads; i++) // compute columns assigned to a thread
                        {
                            // correct offsets based of the location on metadata
                            float *row = _floatMatrix + _floatOffset;
                            uint8_t *indices = _uint8Matrix + _byteOffset;
                            
                            for (uint16_t j = 0; j < rowBlocks; j++) // all rows of the 1st operand matrix
                            {
                                for (uint8_t k = 0; k < K; k++) // all rows of a block of rows
                                {
                                    #pragma GCC unroll 16
                                    for (uint16_t l = 0; l < columnBlocks; l++) // all column blocks
                                    {
                                        #pragma GCC unroll 4
                                        for (uint8_t m = 0; m < 4; m++) // batches in one column block
                                        {
                                            float32x4_t a = vld1q_f32(row);
                                            float32x4_t b = vld1q_f32(dotColumn);
                                            a = vmulq_f32(a, b);

                                            #pragma GCC unroll 4
                                            for (uint8_t n = 0; n < 4; n++)
                                            {
                                                accumulators[indices[n]] += a[n]; // store value to 1 of N accumulators
                                            }                                        

                                            // move pointers
                                            row += 4;
                                            indices += 4;
                                            dotColumn += 4;
                                        }

                                        row += 16 / sizeof(float);      // skip indices
                                        indices += 16 * sizeof(float);  // skip non-zero values
                                    }
                                    dotColumn -= dotMatrix._rows; // reset column pointer, so the column is processed by all rows in the block
                                }

                                #pragma GCC ivdep
                                for (uint8_t k = 0; k < N; k++)
                                {
                                    target[k] = activationFunction(accumulators[k] + addColumn[k]); // add bias and apply activation function
                                    accumulators[k] = 0.0f; // reset accumulators
                                }

                                // move pointers
                                target += N;
                                addColumn += N;
                            }

                            // got to next column
                            addColumn = addVector._floatMatrix;
                            dotColumn += dotMatrix._rows;
                        }
                    }
                }
            }
                        
            template <float (*activationFunction)(float)>
            Dense dotAddActivateRowThread(Dense &dotMatrix, Dense &addVector)
            {
                Dense targetMatrix(_rows, dotMatrix._columns, COLUMN_MAJOR);
                dotAddActivateRowThread<activationFunction>(dotMatrix, addVector, targetMatrix);

                return targetMatrix;
            }
    };
}

#endif