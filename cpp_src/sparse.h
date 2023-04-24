#ifndef SPARSE_H
#define SPARSE_H

#include "base.h"

namespace Matrix
{
    class Sparse : public Base
    {
        protected:
            void loadCSV(std::string fileName) override;
            virtual void allocateSpaceRowMajorCSV(std::ifstream &file) = 0;
            virtual void allocateSpaceColumnMajorCSV(std::ifstream &file) = 0;
            virtual void loadDataRowMajorCSV(std::ifstream &file) = 0;
            virtual void loadDataColumnMajorCSV(std::ifstream &file) = 0;

        public:
            Sparse() = default;
            Sparse(uint16_t rows, uint16_t columns, DimensionMajorityEnum dimMajority);
            Sparse(DimensionMajorityEnum dimMajority);
            ~Sparse() = default;
    };
}

#endif
