#pragma once

// Common definitions

#include <opencv2/core.hpp>

#ifdef DOUBLE_PRECISION
typedef double real;
#else
typedef float real;
#endif

namespace yalal {
    
    enum MatStructure {
        ARBITRARY    = 0,
        UPPER_TRI    = (1 << 0),
        LOWER_TRI    = (1 << 1),
        DIAGONAL     = UPPER_TRI | LOWER_TRI,
        HESSENBERG   = (1 << 2),
        LOWER_BIDIAG = HESSENBERG | LOWER_TRI,
        UPPER_BIDIAG = (1 << 3),
    };

};