#ifndef BASE_H
#define BASE_H

#include <cstddef>
#include <cstring>
#include <string>
#include <string_view>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <bit>
#include <climits>
#include <algorithm>

#include "enums.h"

namespace Matrix
{
    class Base
    {
        protected:
            const uint8_t _BLOCK_SIZE = sizeof(float) * 8;
            const uint32_t _UINT32_MASK_MAX = UINT32_MAX - 1;
            const char *_UNSUPPORTED_MAJORITY = "Unsupported dimension majority.";
            
            union
            {
                float *_floatMatrix = nullptr;
                uint32_t *_uint32Matrix;
                uint8_t *_uint8Matrix;
                char *_charMatrix;
                std::byte *_byteMatrix;
            };
            
            uint32_t _size = 0;
            uint32_t _rows = 0;
            uint32_t _columns = 0;
            DimenstionMajorityEnum _dimMajority = FILE_DETERMINED;

            uint32_t _blockIndex = 0;

            virtual void loadCSV(std::string fileName) = 0;
            virtual void loadBMS(std::string fileName) = 0;

        public:
            Base() = default;
            Base(DimenstionMajorityEnum dimMajority);
            Base(uint32_t rows, uint32_t columns, DimenstionMajorityEnum dimMajority);
            ~Base() = default;

            virtual void printRow(uint32_t rowIndex, uint8_t precision = 7);
            virtual void printColumn(uint32_t columnIndex, uint8_t precision = 7);
            virtual void printMatrix(uint8_t precision = 7);

            virtual void saveAsBMS(std::string fileName) = 0;
            virtual void saveAsCSV(std::string fileName) = 0;
    };    
}

#endif