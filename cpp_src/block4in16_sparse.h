#ifndef BLOCK4IN16_SPARSE_H
#define BLOCK4IN16_SPARSE_H

#include "base.h"
#include "sparse.h"
#include "dense.h"

namespace Matrix 
{
    class Block4in16Sparse : public Sparse
    {
        private:
            void allocateSpaceRowMajorCSV(std::ifstream &file) override;
            void allocateSpaceColumnMajorCSV(std::ifstream &file) override;
            void loadDataRowMajorCSV(std::ifstream &file) override;
            void loadDataColumnMajorCSV(std::ifstream &file) override;

            void loadBinary(std::string fileName) override;
        
        public:
            Block4in16Sparse() = default;
            Block4in16Sparse(std::string fileName, DimensionMajorityEnum dimMajority = FILE_DETERMINED);
            ~Block4in16Sparse() = default;

            void printCompressed(uint8_t precision = 7);

            void printColumn(uint16_t columnIndex, uint8_t precision = 7) override;
            void printRow(uint16_t rowIndex, uint8_t precision = 7) override;

            virtual void saveAsBinary(std::string fileName) override;
            virtual void saveAsCSV(std::string fileName) override;

            void dot(Dense &operandMatrix, Dense &targetMatrix);
            Dense dot(Dense &operandMatrix);

            void dot1(Dense &operandMatrix, Dense &targetMatrix);
            Dense dot1(Dense &operandMatrix);

            void dot2(Dense &operandMatrix, Dense &targetMatrix);
            Dense dot2(Dense &operandMatrix);
    };
}

#endif