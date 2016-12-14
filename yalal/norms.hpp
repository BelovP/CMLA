#include "common.hpp"

namespace yalal {

    namespace VectorNorm {
        enum VectorNorm {
            L2_SQUARED = -1,
            INF = 0
        };
    }

    real vectorNorm(cv::Mat_<real> vec, int p = 2);

    namespace MatrixNorm {
        enum MatrixNorm {
            FROBENIUS_SQUARED = -2,
            FROBENIUS = -1,
            INF = 0,
        };
    };

    real matrixNorm(cv::Mat_<real> A, int type = 2);

};