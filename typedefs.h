#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <cstddef>
#include <string>

namespace Matrix
{
    typedef union
    {
        float *floatMatrix = nullptr;
        uint32_t *uint32Matrix;
        uint16_t *uint16Matrix;
        uint8_t *uint8Matrix;
        char *charMatrix;
        std::byte *byteMatrix;
    } matrix_ptrs_t;
}

#endif