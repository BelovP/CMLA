#pragma once

// Common definitions

#include <opencv2/core.hpp>

#ifdef DOUBLE_PRECISION
typedef double real;
#else
typedef float real;
#endif

namespace yalal {

    namespace MatStructure {
        enum MatStructure {
            DIAGONAL = (1 << 0),
            LOWER_BIDIAG = (1 << 1) | DIAGONAL,
            UPPER_BIDIAG = (1 << 2) | DIAGONAL,
            TRIDIAGONAL = (1 << 3) | DIAGONAL | UPPER_BIDIAG | LOWER_BIDIAG,
            LOWER_TRI = (1 << 4) | DIAGONAL | LOWER_BIDIAG,
            UPPER_TRI = (1 << 5) | DIAGONAL | UPPER_BIDIAG,
            HESSENBERG = (1 << 6) | DIAGONAL | UPPER_BIDIAG | LOWER_BIDIAG | UPPER_TRI | TRIDIAGONAL,
            ARBITRARY = ~0
        };
    };

};