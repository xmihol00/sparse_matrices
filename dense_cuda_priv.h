#ifndef DENS_CUDA_PRIV_H
#define DENS_CUDA_PRIV_H

#include <string>

__global__ void dotRowsColumns(float *operandA, float *operandB, float *target, uint16_t rowsA, uint16_t columnsB, uint16_t elements);

#endif