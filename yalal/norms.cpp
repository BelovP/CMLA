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

    real matrixNorm(cv::Mat_<real> A, int type) {
        
        switch (type) {
            case MatrixNorm::INF: {
                real retval = 0;
                for (int i = 0; i < A.rows; ++i) {
                    retval = std::max(retval, vectorNorm(A.row(i), 1));
                }
                return retval;
            }

            case MatrixNorm::FROBENIUS:
            case MatrixNorm::FROBENIUS_SQUARED: {
                real retval = 0;
                for (auto & x : A) {
                    // should be more precise if parallelized
                    retval += x * x;
                }
                return (type == MatrixNorm::FROBENIUS ? std::sqrt(retval) : retval);
            }

            case 1: {
                real retval = 0;
                for (int j = 0; j < A.cols; ++j) {
                    retval = std::max(retval, vectorNorm(A.col(j), 1));
                }
                return retval;
            }

            case 2: {
                assert(false && "Matrix spectral norm is NYI");
                return -1;
            }

            default: {
                assert(false && "Wrong matrix norm type");
                return -1;
            }
        }
    }

}; // namespace yalal