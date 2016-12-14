#include "common.hpp"

namespace yalal {

    enum CholeskyMethod {
        CHOLESKY_CHOLESKY,
        CHOLESKY_BANACHIEWICZ,
        CHOLESKY_CROUT
    };

    void Cholesky(cv::Mat_<real> & A, cv::Mat_<real> & L,
                  int matStructure = ARBITRARY, int method = CHOLESKY_CHOLESKY);

};