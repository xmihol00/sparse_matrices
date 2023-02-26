#include "block_sparse_cuda.h"
#include "block_sparse_cuda_priv.h"

using namespace Matrix;
using namespace std;

__global__ void dotRowsColumns(uint16_t *blockIndicesOperandA, uint8_t *entryIndicesOperandA, float *dataOperandA, float *operandB, float *target, 
                               uint16_t rowsA, uint16_t columnsB, uint16_t blocksPerRow, uint8_t entriesPerBlock, uint16_t elements)
{
    uint16_t rowIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint16_t columnIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (rowIndex >= rowsA || columnIndex >= columnsB)
    {
        return;
    }

    uint16_t *offsetBlockIndicesOperandA = &blockIndicesOperandA[rowIndex * blocksPerRow];
    uint8_t *offsetEntryIndicesOperandA = &entryIndicesOperandA[rowIndex * blocksPerRow * (entriesPerBlock - 1)];
    float *offsetOperandA = &dataOperandA[rowIndex * blocksPerRow * entriesPerBlock];
    
    register float accumulator = 0;
    for (uint16_t i = 0; i < blocksPerRow; i++)
    {
        float *offsetOperandB = &operandB[columnIndex * elements + offsetBlockIndicesOperandA[i]];
        for (uint8_t j = 0, k = 0; j < entriesPerBlock; k = offsetEntryIndicesOperandA[j++])
        {
             accumulator += offsetOperandA[j] * offsetOperandB[k];
        }
        offsetEntryIndicesOperandA = &offsetEntryIndicesOperandA[entriesPerBlock - 1];
        offsetOperandA = &offsetOperandA[entriesPerBlock];
    }

    target[columnIndex * columnsB + rowIndex] = accumulator;
}

void dotRowsColumns(matrix_ptrs_t operandA, float *operandB, float *target, uint32_t offsetOfEntryIndices, uint64_t offsetOfData,
                    uint16_t blocksPerRow, uint8_t entriesPerBlock, uint16_t rowsA, uint16_t columnsB, uint64_t sizeA, uint64_t sizeB)
{
    uint8_t blockDimension = rowsA * columnsB / (UINT8_MAX + 1) > UINT16_MAX ? 32 : 16;
    float *dOperandB, *dTarget, *dDataOperandA;
    matrix_ptrs_t dOperandA, tmp;
    uint8_t *dEntryIndicesOperandA;
    uint64_t sizeTraget = rowsA * columnsB * sizeof(float);
    
    cudaMalloc(reinterpret_cast<void **>(&dOperandA.uint8s), sizeA);
    cudaMalloc(reinterpret_cast<void **>(&dOperandB), sizeB);
    cudaMalloc(reinterpret_cast<void **>(&dTarget), sizeTraget);
    
    cudaMemcpy(dOperandA.uint8s, operandA.uint8s, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dOperandB, operandB, sizeB, cudaMemcpyHostToDevice);
    tmp.uint16s = &dOperandA.uint16s[offsetOfEntryIndices];
    dEntryIndicesOperandA = tmp.uint8s;
    tmp.uint8s = &dOperandA.uint8s[offsetOfData];
    dDataOperandA = tmp.floats;

    dim3 blockSize(blockDimension, blockDimension, 1);
    dim3 gridSize((rowsA + blockDimension - 1) / blockDimension, (columnsB + blockDimension - 1) / blockDimension, 1);
    dotRowsColumns<<<gridSize, blockSize>>>(dOperandA.uint16s, dEntryIndicesOperandA, dDataOperandA, dOperandB, dTarget, 
                                            rowsA, columnsB, blocksPerRow, entriesPerBlock, sizeB / (columnsB * sizeof(float)));
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        cerr << cudaGetErrorString(error) << endl;
        exit(1);
    }
    
    cudaMemcpy(target, dTarget, sizeTraget, cudaMemcpyDeviceToHost);
    
    cudaFree(dOperandA.uint8s);
    cudaFree(dOperandB);
    cudaFree(dTarget);
}
