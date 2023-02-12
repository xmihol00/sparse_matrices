#ifndef BITMAP_SPARSE_H
#define BITMAP_SPARSE_H

#include <cstddef>
#include <string>
#include <string_view>
#include <stdexcept>
#include <iostream>

namespace SparseMatrix
{
    class BitmapSparse
    {
        private:
            std::byte *_matrix;
            void parseCSV(std::string fileName);
            void parseBMS(std::string fileName);

        public:
            BitmapSparse(std::string fileName);
            ~BitmapSparse();
    };
}

#endif
