#include "dense.cuh"
#include "dense_priv.cuh"

using namespace std;

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
    
    register float accumulator = 0;
    for (uint16_t i = 0; i < elements; i++)
    {
        accumulator += offsetOperandA[i] * offsetOperandB[i];
    }

    target[columnIndex * columnsB + rowIndex] = accumulator;
}

void dotRowsColumns(float *operandA, float *operandB, float *target, uint16_t rowsA, uint16_t columnsB, uint64_t sizeA, uint64_t sizeB)
{
    uint8_t blockDimension = rowsA * columnsB / (UINT8_MAX + 1) > UINT16_MAX ? 32 : 16;
    float *dOperandA, *dOperandB, *dTarget;
    uint64_t sizeTraget = rowsA * columnsB * sizeof(float);
    
    cudaMalloc(reinterpret_cast<void **>(&dOperandA), sizeA);
    cudaMalloc(reinterpret_cast<void **>(&dOperandB), sizeB);
    cudaMalloc(reinterpret_cast<void **>(&dTarget), sizeTraget);
    
    cudaMemcpy(dOperandA, operandA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dOperandB, operandB, sizeB, cudaMemcpyHostToDevice);

    dim3 blockSize(blockDimension, blockDimension, 1);
    dim3 gridSize((rowsA + blockDimension - 1) / blockDimension, (columnsB + blockDimension - 1) / blockDimension, 1);
    dotRowsColumns<<<gridSize, blockSize>>>(dOperandA, dOperandB, dTarget, rowsA, columnsB, sizeA / (rowsA * sizeof(float)));
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        cerr << cudaGetErrorString(error) << endl;
        exit(1);
    }
    
    cudaMemcpy(target, dTarget, sizeTraget, cudaMemcpyDeviceToHost);
    
    cudaFree(dOperandA);
    cudaFree(dOperandB);
    cudaFree(dTarget);
}


void dotCuBLAS(float *operandA, float *operandB, float *target, uint16_t rowsA, uint16_t columnsB, uint16_t columnsARowsB, 
                uint16_t lda, uint16_t ldb, uint16_t ldc)
{
    float *dOperandA, *dOperandB, *dTarget;
    uint32_t sizeA = rowsA * columnsARowsB * sizeof(float);
    uint32_t sizeB = columnsB * columnsARowsB * sizeof(float);
    uint32_t sizeTarget = rowsA * columnsB * sizeof(float);
    cudaMalloc(reinterpret_cast<void **>(&dOperandA), sizeA);
    cudaMalloc(reinterpret_cast<void **>(&dOperandB), sizeB);
    cudaMalloc(reinterpret_cast<void **>(&dTarget), sizeTarget);
    
    cudaMemcpy(dOperandA, operandA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dOperandB, operandB, sizeB, cudaMemcpyHostToDevice);

    const float alpha = 1;
    const float beta = 0;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, rowsA, columnsB, columnsARowsB, 
                                        &alpha, dOperandA, lda, dOperandB, ldb, &beta, dTarget, ldc);
    cudaError_t error = cudaDeviceSynchronize();
    cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS || error != cudaSuccess)
    {
        cerr << "cuBLAS matrix multiply failed: " << cudaGetErrorString(error) << endl;
        exit(1);
    }

    cudaMemcpy(target, dTarget, sizeTarget, cudaMemcpyDeviceToHost);
    
    cudaFree(dOperandA);
    cudaFree(dOperandB);
    cudaFree(dTarget);
}
