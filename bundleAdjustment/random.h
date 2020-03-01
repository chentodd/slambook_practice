#ifndef RANDOM_H
#define RANDOM_H

#include <math.h>
#include <stdlib.h>

inline double RandDouble()
{
    double r = static_cast<double>(rand());
    return r / RAND_MAX;
}

// Box–Muller transform, generate normal-distributed random number
inline double RandNormal()
{
    double x1, x2, w;
    do
    {
        // Get two independent variable in (-1, 1)
        x1 = 2.0 * RandDouble() - 1.0;
        x2 = 2.0 * RandDouble() - 1.0;
        w = x1 * x1 + x2 * x2;
    }
    while( w >= 1.0 || w == 0.0);

    w = sqrt((-2.0 * log(w))/w);
    return x1 * w;
}

#endif