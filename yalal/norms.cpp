#include "norms.hpp"
#include <cmath>

real fastPower(real x, int p) {
    real retval = 1;
    while (p) {
        if (p & 1)
            retval *= x;
        x *= x;
        p >>= 1;
    }
    return retval;
}

namespace yalal {

    real vectorNorm(cv::Mat_<real> vec, int p) {

        switch (p) {
            case VectorNorm::INF: {
                real retval = 0;
                for (auto & x : vec) {
                    retval = std::max(retval, std::abs(x));
                }
                return retval;
            }

            case 1: {
                real retval = 0;
                for (auto & x : vec) {
                    // should be more precise if parallelized
                    retval += std::abs(x);
                }
                return retval;
            }

            case 2:
            case VectorNorm::L2_SQUARED: {
                real retval = 0;
                for (auto & x : vec) {
                    retval += x*x;
                }
                return (p == 2 ? std::sqrt(retval) : retval);
            }

            default: {
                real retval = 0;
                for (auto & x : vec) {
                    retval += fastPower(x, p);
                }
                return std::pow(retval, real(1) / p);
            }
        }
    }

}; // namespace yalal