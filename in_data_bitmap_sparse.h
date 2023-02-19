#ifndef BITMAP_SPARSE_H
#define BITMAP_SPARSE_H

#include "base.h"
#include "dense.h"

namespace Matrix
{
    class InDataBitmapSparse : public Base
    {
        private:
            friend class Dense;

            uint32_t *_buffer;
            uint32_t _blocksPerDimension;

            void loadCSV(std::string fileName) override;
            void allocateSpaceRowMajorCSV(std::ifstream &file);
            void allocateSpaceColumnMajorCSV(std::ifstream &file);
            void loadDataRowMajorCSV(std::ifstream &file);
            void loadDataColumnMajorCSV(std::ifstream &file);
            
            void loadBMS(std::string fileName) override;

            void moveToRow(uint32_t rowIndex);
            void moveToColumn(uint32_t columnIndex);
            std::tuple<uint32_t, float *> nextRowBlock();
            std::tuple<uint32_t, float *> nextColumnBlock();

        public:
            InDataBitmapSparse(std::string fileName, DimenstionMajorityEnum dimMajority = FILE_DETERMINED);
            ~InDataBitmapSparse();

            void printRow(uint32_t rowIndex, uint8_t precision = 7) override;
            void printColumn(uint32_t columnIndex, uint8_t precision = 7) override;

            void saveAsBMS(std::string fileName) override;
            void saveAsCSV(std::string fileName) override;

            void dot(InDataBitmapSparse &matrix, Dense &targetMatrix);
            Dense dot(InDataBitmapSparse &matrix);
            void dot(Dense &matrix, Dense &targetMatrix);
            Dense dot(Dense &matrix);
    };
}

#endif
