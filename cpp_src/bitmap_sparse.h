#ifndef BITMAP_SPARSE_H
#define BITMAP_SPARSE_H

#include "sparse.h"
#include "dense.h"

namespace Matrix
{
    class BitmapSparse : public Sparse
    {
        private:
            friend class Dense;
            
            uint32_t *_buffer;
            uint32_t _blockIndex = 0;
            uint16_t _blocksPerDimension = 0;

            void allocateSpaceRowMajorCSV(std::ifstream &file) override;
            void allocateSpaceColumnMajorCSV(std::ifstream &file) override;
            void loadDataRowMajorCSV(std::ifstream &file) override;
            void loadDataColumnMajorCSV(std::ifstream &file) override;
            
            void loadBinary(std::string fileName) override;

            void moveToRow(uint16_t rowIndex);
            void moveToColumn(uint16_t columnIndex);
            std::tuple<uint32_t, float *> nextRowBlock();
            std::tuple<uint32_t, float *> nextColumnBlock();

        public:
            BitmapSparse(std::string fileName, DimensionMajorityEnum dimMajority = FILE_DETERMINED);
            ~BitmapSparse() = default;

            void printRow(uint16_t rowIndex, uint8_t precision = 7) override;
            void printColumn(uint16_t columnIndex, uint8_t precision = 7) override;

            void saveAsBinary(std::string fileName) override;
            void saveAsCSV(std::string fileName) override;

            void dot(BitmapSparse &operandMatrix, Dense &targetMatrix);
            Dense dot(BitmapSparse &operandMatrix);
            void dot(Dense &operandMatrix, Dense &targetMatrix);
            Dense dot(Dense &operandMatrix);
    };
}

#endif
