#ifndef CSR_SPARSE_H
#define CSR_SPARSE_H

#include "sparse.h"
#include "dense.h"

namespace Matrix
{
    class CSRSparse : public Sparse
    {
        private:
            uint32_t _nonZeroEntries = 0;
            uint16_t *_columnIndices;
            float *_dataMatrix;

            void allocateSpaceRowMajorCSV(std::ifstream &file) override;
            void allocateSpaceColumnMajorCSV(std::ifstream &file) override;
            void loadDataRowMajorCSV(std::ifstream &file) override;
            void loadDataColumnMajorCSV(std::ifstream &file) override;

            void loadBinary(std::string fileName) override;
        
        public:
            CSRSparse() = default;
            CSRSparse(std::string fileName, DimensionMajorityEnum dimMajority = FILE_DETERMINED);
            ~CSRSparse() = default;

            void printColumn(uint16_t columnIndex, uint8_t precision = 7) override;
            void printRow(uint16_t rowIndex, uint8_t precision = 7) override;

            virtual void saveAsBinary(std::string fileName) override;
            virtual void saveAsCSV(std::string fileName) override;

            void dot(Dense &operandMatrix, Dense &targetMatrix);
            Dense dot(Dense &operandMatrix);
    };
}

#endif
