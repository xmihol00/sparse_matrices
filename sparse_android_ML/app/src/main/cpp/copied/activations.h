#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

namespace Matrix
{
    constexpr float ReLU(float x) 
    {
        return (x > 0.0f ? x : 0.0f);
    }
}

#endif