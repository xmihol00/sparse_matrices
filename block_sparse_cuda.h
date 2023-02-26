#ifndef BLOCK_SPARSE_CUDA_H
#define BLOCK_SPARSE_CUDA_H

#include <string>
#include "typedefs.h"

void dotRowsColumns(Matrix::matrix_ptrs_t operandA, float *operandB, float *target, uint32_t offsetOfEntryIndices, uint64_t offsetOfData,
                    uint16_t blocksPerRow, uint8_t entriesPerBlock, uint16_t rowsA, uint16_t columnsB, uint64_t sizeA, uint64_t sizeB);

#endif
