#include "common.hpp"

namespace yalal {

    enum VectorNorm {
        L2_SQUARED = -1,
        INF        =  0
    };

    real vectorNorm(cv::Mat_<real> v, int p = 2);

};