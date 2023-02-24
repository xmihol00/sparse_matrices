#include "dense_cuda.h"
#include "dense_cuda_priv.h"

__global__ void dotRowsColumns(float *operandA, float *operandB, float *target, uint16_t rowsA, uint16_t columnsB, uint16_t elements)
{
    uint16_t rowIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint16_t columnIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (rowIndex >= rowsA || columnIndex >= columnsB)
    {
        return;
    }

    float *offsetOperandA = &operandA[rowIndex * elements];
    float *offsetOperandB = &operandB[columnIndex * elements];
    
    float accumulator = 0;
    for (uint16_t i = 0; i < elements; i++)
    {
        accumulator += offsetOperandA[i] * offsetOperandB[i];
    }

    target[columnIndex * columnsB + rowIndex] = accumulator;
}

void dotRowsColumns(float *operandA, float *operandB, float *target, uint16_t rowsA, uint16_t columnsB, uint64_t sizeA, uint64_t sizeB)
{
    const uint8_t BLOCK_DIMENSION = 16;
    float *dOperandA, *dOperandB, *dTarget;
    uint32_t sizeTraget = rowsA * columnsB * sizeof(float);
    
    cudaMalloc((void **)&dOperandA, sizeA);
    cudaMalloc((void **)&dOperandB, sizeB);
    cudaMalloc((void **)&dTarget, sizeTraget);
    
    cudaMemcpy(dOperandA, operandA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dOperandB, operandB, sizeB, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_DIMENSION, BLOCK_DIMENSION, 1);
    dim3 gridSize((rowsA + BLOCK_DIMENSION - 1) / BLOCK_DIMENSION, (columnsB + BLOCK_DIMENSION - 1) / BLOCK_DIMENSION, 1);
    dotRowsColumns<<<blockSize, gridSize>>>(dOperandA, dOperandB, dTarget, rowsA, columnsB, sizeA / (rowsA * sizeof(float)));
    cudaDeviceSynchronize();
    
    cudaMemcpy(target, dTarget, sizeTraget, cudaMemcpyDeviceToHost);
    
    cudaFree(dOperandA);
    cudaFree(dOperandB);
    cudaFree(dTarget);
}
