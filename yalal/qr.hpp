#include "common.hpp"

namespace yalal {

    namespace QRMethod {
        enum QRMethod {
            GS,
            GS_MODIFIED,
            HOUSEHOLDER,
            GIVENS
        };
    };

    void QR(cv::Mat_<real> & A, cv::Mat_<real> & Q, cv::Mat_<real> & R,
            int matStructure = MatStructure::ARBITRARY,
            int method = QRMethod::HOUSEHOLDER);

}
