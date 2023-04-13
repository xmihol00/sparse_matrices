#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <string>

namespace Matrix
{
    typedef union
    {
        float *floats = nullptr;
        uint32_t *uint32s;
        uint16_t *uint16s;
        uint8_t *uint8s;
        char *chars;
    } matrix_ptrs_t;
}

#endif