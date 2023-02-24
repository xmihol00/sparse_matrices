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

            void loadCSV(std::string fileName) override;
            void loadBinary(std::string fileName) override;
        
        public:
            Dense() = default;
            Dense(std::string fileName, DimenstionMajorityEnum dimMajority = FILE_DETERMINED);
            Dense(uint32_t rows, uint32_t columns, DimenstionMajorityEnum dimMajority, std::byte *data = nullptr);
            ~Dense() = default;

            void printColumn(uint16_t columnIndex, uint8_t precision = 7) override;
            void printRow(uint16_t rowIndex, uint8_t precision = 7) override;

            virtual void saveAsBinary(std::string fileName) override;
            virtual void saveAsCSV(std::string fileName) override;

            void dotGPU(Dense &matrix, Dense &targetMatrix);
            Dense dotGPU(Dense &matrix);
    };
}

#endif