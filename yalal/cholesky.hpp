#include "common.hpp"

namespace yalal {

    namespace CholeskyMethod {
        enum {
            OUTER_PRODUCT,
            CHOLESKY_BANACHIEWICZ,
            CHOLESKY_CROUT
        };
    };

    void Cholesky(cv::Mat_<real> & A, cv::Mat_<real> & L,
                  int matStructure = MatStructure::ARBITRARY,
                  int method = CholeskyMethod::OUTER_PRODUCT);

};