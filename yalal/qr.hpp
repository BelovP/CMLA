#include "common.hpp"

namespace yalal {

    enum QRMethod {
        GS,
        GS_MODIFIED,
        HOUSEHOLDER,
        GIVENS
    };

    void QR(cv::Mat_<real> A, cv::Mat_<real> & Q, cv::Mat_<real> & R,
               int method = HOUSEHOLDER, int matStructure = ARBITRARY);

}