#ifndef DENS_CUDA_H
#define DENS_CUDA_H

#include <string>

void dotRowsColumns(float *operandA, float *operandB, float *target, uint16_t rowsA, uint16_t columnsB, uint64_t sizeA, uint64_t sizeB);


void dotCuBLAS(float *operandA, float *operandB, float *target, uint16_t rowsA, uint16_t columnsB, uint16_t columnsARowsB, 
               uint16_t lda, uint16_t ldb, uint16_t ldc);

#endif