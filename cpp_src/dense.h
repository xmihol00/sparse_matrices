#ifndef DENS_H
#define DENS_H

#include "base.h"

namespace Matrix
{
    class Dense : public Base
    {
        private:
            friend class InDataBitmapSparse;
            friend class BlockSparse;
            friend class Block4in16Sparse;

            void loadCSV(std::string fileName) override;
            void loadBinary(std::string fileName) override;
        
        public:
            Dense() = default;
            Dense(std::string fileName, DimensionMajorityEnum dimMajority = FILE_DETERMINED);
            Dense(uint16_t rows, uint16_t columns, DimensionMajorityEnum dimMajority, std::byte *data = nullptr);
            ~Dense() = default;

            void printColumn(uint16_t columnIndex, uint8_t precision = 7) override;
            void printRow(uint16_t rowIndex, uint8_t precision = 7) override;

            virtual void saveAsBinary(std::string fileName) override;
            virtual void saveAsCSV(std::string fileName) override;

            void add(Dense &operandMatrix);

            void ReLU();

            void argmax(uint8_t axis, Dense &targetMatrix);
            Dense argmax(uint8_t axis);

            void dot(Dense &operandMatrix, Dense &targetMatrix);
            Dense dot(Dense &operandMatrix);

            void dotGPU(Dense &operandMatrix, Dense &targetMatrix);
            Dense dotGPU(Dense &operandMatrix);

            void dotGPUCuBLAS(Dense &operandMatrix, Dense &targetMatrix);
            Dense dotGPUCuBLAS(Dense &operandMatrix);
    };
}

#endif