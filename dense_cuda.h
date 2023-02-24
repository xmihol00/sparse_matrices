#ifndef DENS_CUDA_H
#define DENS_CUDA_H

#include <string>

void dotRowsColumns(float *operandA, float *operandB, float *target, uint16_t rowsA, uint16_t columnsB, uint64_t sizeA, uint64_t sizeB);

#endif