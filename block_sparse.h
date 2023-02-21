#ifndef BLOCK_SPARSE_H
#define BLOCK_SPARSE_H

#include "sparse.h"

namespace Matrix
{
    class BlockSparse : public Sparse
    {
        private:
            uint8_t _entriesPerBlock;
            uint16_t _entriesPerDimension;

            uint8_t *_entriesOffsets;
            float *_dataMatrix;

            void allocateSpaceRowMajorCSV(std::ifstream &file) override;
            void allocateSpaceColumnMajorCSV(std::ifstream &file) override;
            void loadDataRowMajorCSV(std::ifstream &file) override;
            void loadDataColumnMajorCSV(std::ifstream &file) override;

            void loadBinary(std::string fileName) override;
        
        public:
            BlockSparse() = default;
            BlockSparse(std::string fileName, uint16_t blocksPerDimension, DimenstionMajorityEnum dimMajority = FILE_DETERMINED);
            ~BlockSparse() = default;

            void printColumn(uint32_t columnIndex, uint8_t precision = 7) override;
            void printRow(uint32_t rowIndex, uint8_t precision = 7) override;

            virtual void saveAsBinary(std::string fileName) override;
            virtual void saveAsCSV(std::string fileName) override;
    };
}

#endif
