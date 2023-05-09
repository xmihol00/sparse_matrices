#ifndef DENS_H
#define DENS_H

#include <thread>
#include <vector>

#if PHONE
    #include <arm_neon.h>
#else
    #include "arm_neon_.h"
#endif
#include "base.h"

namespace Matrix
{
    class Dense : public Base
    {
        private:
            friend class InDataBitmapSparse;
            friend class BlockSparse;
            friend class Block4in16Sparse;
            template <uint8_t K, uint8_t N, uint16_t denseRows, uint16_t denseColumns, DimensionMajorityEnum dimMajority>
            friend class BlockKinNSparse;

            uint16_t _bytePadding;
            uint8_t _threadPadding;

            void loadCSV(std::string fileName) override;
            void loadBinary(std::string fileName) override;
        
        public:
            Dense() = default;
            Dense(std::string fileName, DimensionMajorityEnum dimMajority = FILE_DETERMINED, uint16_t bytePadding = 0, uint8_t threadPadding = 1);
            Dense(uint16_t rows, uint16_t columns, DimensionMajorityEnum dimMajority);
            Dense(uint16_t rows, uint16_t columns, DimensionMajorityEnum dimMajority, uint8_t threadPadding);
            Dense(uint16_t rows, uint16_t columns, DimensionMajorityEnum dimMajority, uint16_t bytePadding);
            Dense(uint16_t rows, uint16_t columns, DimensionMajorityEnum dimMajority, std::byte *data);
            Dense(uint16_t rows, uint16_t columns, DimensionMajorityEnum dimMajority, uint16_t bytePadding, std::byte *data);
            Dense(const Matrix::Dense& other);
            Dense(Matrix::Dense&& other);

            ~Dense() = default;

            Dense &operator=(Dense &&other);
            float &operator()(uint16_t rowIndex, uint16_t columnIndex);

            float *getData();
            void setFloatMatrix(float *data, uint16_t rows, uint16_t columns, DimensionMajorityEnum dimMajority);
            void clear();

            void printColumn(uint16_t columnIndex, uint8_t precision = 7) override;
            void printRow(uint16_t rowIndex, uint8_t precision = 7) override;

            virtual void saveAsBinary(std::string fileName) override;
            virtual void saveAsCSV(std::string fileName) override;

            void add(Dense &operandMatrix);

            void ReLU();

            void argmax(uint8_t axis, Dense &targetMatrix);
            Dense argmax(uint8_t axis);
            uint32_t argmax();

            float percentageDifference(Dense &operandMatrix, float threshold = 0.001);

            void dot(Dense &operandMatrix, Dense &targetMatrix);
            Dense dot(Dense &operandMatrix);

            void dotNEON(Dense &operandMatrix, Dense &targetMatrix);
            Dense dotNEON(Dense &operandMatrix);

            void dotNEONThread(Dense &operandMatrix, Dense &targetMatrix, uint8_t numberOfThreads, uint8_t threadId);
            void dotNEONThreads(Dense &operandMatrix, Dense &targetMatrix, uint8_t numberOfThreads);
            Dense dotNEONThreads(Dense &operandMatrix, uint8_t numberOfThreads);

            void dotGPU(Dense &operandMatrix, Dense &targetMatrix);
            Dense dotGPU(Dense &operandMatrix);

            void dotGPUCuBLAS(Dense &operandMatrix, Dense &targetMatrix);
            Dense dotGPUCuBLAS(Dense &operandMatrix);

            void dotAddActivate(Dense &dotMatrix, Dense &addMatrix, Dense &targetMatrix, float (*activationFunction)(float));
            Dense dotAddActivate(Dense &dotMatrix, Dense &addMatrix, float (*activationFunction)(float));

            void dotAddActivateThread(Dense &dotMatrix, Dense &addMatrix, Dense &targetMatrix, float (*activationFunction)(float),
                                      uint8_t numberOfThreads, uint8_t threadId);
            void dotAddActivateThreads(Dense &dotMatrix, Dense &addMatrix, Dense &targetMatrix, float (*activationFunction)(float),
                                       uint8_t numberOfThreads);
            Dense dotAddActivateThreads(Dense &dotMatrix, Dense &addMatrix, float (*activationFunction)(float),
                                        uint8_t numberOfThreads);
    };
}

#endif