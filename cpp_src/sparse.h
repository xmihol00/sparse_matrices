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
            Sparse(DimensionMajorityEnum dimMajority);
            ~Sparse() = default;
    };
}

#endif
