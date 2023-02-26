#ifndef BLOCK_SPARSE_CUDA_PRIV_H
#define BLOCK_SPARSE_CUDA_PRIV_H

#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include "stdio.h"

__global__ void dotRowsColumns(uint16_t *blockIndicesOperandA, uint8_t *entryIndicesOperandA, float *dataOperandA, float *operandB, float *target, 
                               uint16_t rowsA, uint16_t columnsB, uint16_t blocksPerRow, uint8_t entriesPerBlock, uint16_t elements);

#endif
