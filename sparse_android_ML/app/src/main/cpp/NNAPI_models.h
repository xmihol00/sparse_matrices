
#ifndef NNAPI_MODELS_H
#define NNAPI_MODELS_H

#include <jni.h>
#include <android/log.h>
#include <android/NeuralNetworks.h>
#include <android/NeuralNetworksTypes.h>
#include <android/sharedmem.h>
#include <sys/mman.h>
#include <string>
#include <unistd.h>

#include "copied/dense.h"

namespace Models
{
    class NNAPI_Mnist32x32_4L
    {
        private:
            Matrix::Dense _W0;
            Matrix::Dense _W1;
            Matrix::Dense _W2;
            Matrix::Dense _W3;
            Matrix::Dense _W4;
            
            Matrix::Dense _B0;
            Matrix::Dense _B1;
            Matrix::Dense _B2;
            Matrix::Dense _B3;
            Matrix::Dense _B4;

            int _W0fd;
            int _W1fd;
            int _W2fd;
            int _W3fd;
            int _W4fd;
            int _B0fd;
            int _B1fd;
            int _B2fd;
            int _B3fd;
            int _B4fd;

            ANeuralNetworksMemory *_W0SharedMemory;
            ANeuralNetworksMemory *_W1SharedMemory;
            ANeuralNetworksMemory *_W2SharedMemory;
            ANeuralNetworksMemory *_W3SharedMemory;
            ANeuralNetworksMemory *_W4SharedMemory;
            ANeuralNetworksMemory *_B0SharedMemory;
            ANeuralNetworksMemory *_B1SharedMemory;
            ANeuralNetworksMemory *_B2SharedMemory;
            ANeuralNetworksMemory *_B3SharedMemory;
            ANeuralNetworksMemory *_B4SharedMemory;

            ANeuralNetworksModel *_model = nullptr;
            ANeuralNetworksCompilation *_compilation = nullptr;
            ANeuralNetworksExecution *_execution = nullptr;

            int32_t _activationReLU = ANEURALNETWORKS_FUSED_RELU;
            int32_t _activationNONE = ANEURALNETWORKS_FUSED_NONE;

            const uint16_t INPUT_SIZE = 1024 * sizeof(float);
            Matrix::Dense _output;

            void loadToSharedMemory(std::string name, ANeuralNetworksMemory *&memory, Matrix::Dense &matrix);
        
        public:
            NNAPI_Mnist32x32_4L(std::string path);
            NNAPI_Mnist32x32_4L();
            ~NNAPI_Mnist32x32_4L();

            void load(std::string path);
            uint8_t predict(float *input);
    };
}

#endif
