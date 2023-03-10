#ifndef DENS_CUDA_PRIV_H
#define DENS_CUDA_PRIV_H

#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

__global__ void dotRowsColumns(float *operandA, float *operandB, float *target, uint16_t rowsA, uint16_t columnsB, uint16_t elements);

#endif