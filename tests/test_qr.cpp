#include <iostream>
#include "qr.hpp"

const real EPS = 1e-4;

using yalal::QR;
using yalal::QRMethod;
using yalal::MatStructure;

int main() {

    std::vector<const char*> qrTypes =
            {"GS", "Modified GS", "Householder", "Givens"};
    std::vector<real> allowedError =
            {3e-2, 1e-3, 1e-4, 1e-4};

    std::vector<const char*> matTypes =
            {"random", "diagonal", "diagonal (treated as arbitrary)",
             "Hilbert", "upper triangular", "lower triangular", "tridiagonal"};
    std::vector<MatStructure> matTypeFlags =
            {MatStructure::ARBITRARY, MatStructure::DIAGONAL, MatStructure::ARBITRARY,
             MatStructure::ARBITRARY, MatStructure::UPPER_TRI, MatStructure::LOWER_TRI,
             MatStructure::TRIDIAGONAL};

    std::vector<cv::Mat_<real>> mats(matTypes.size());

    // random
    mats[0].create(235, 177);
    cv::randu(mats[0], -1., 1.);

    // diagonal
    mats[1] = cv::Mat_<real>::zeros(117, 117);
    for (int i = 0; i < mats[1].rows; ++i) {
        mats[1].at<real>(i, i) = (cv::randu<real>() - 0.5) * 100;
    }

    // diagonal
    mats[2] = mats[1];

    // Hilbert
    mats[3].create(127, 127);
    for (int i = 0; i < mats[3].rows; ++i) {
        for (int j = 0; j < mats[3].cols; ++j) {
            mats[3].at<real>(i, j) = 1. / (i + j + 1);
        }
    }

    // upper triangular
    mats[4] = cv::Mat_<real>::zeros(354, 354);
    for (int i = 0; i < 354; ++i) {
        for (int j = i; j < 354; ++j) {
            mats[4].at<real>(i, j) = (cv::randu<real>() - 0.5) * 2.;
        }
    }

    // lower triangular
    cv::transpose(mats[4], mats[5]);

    // tridiagonal
    mats[6] = cv::Mat_<real>::zeros(345, 345);
    for (int i = 0; i < 345; ++i) {
        for (int j = std::max(0, i-1); j < std::min(345-1, i+1); ++j) {
            mats[6].at<real>(i, j) = (cv::randu<real>() - 0.5) * 10;
        }
    }

    cv::Mat_<real> Q, R;

    for (int qrType = 0; qrType < qrTypes.size(); ++qrType) {
        for (int matType = 0; matType < matTypes.size(); ++matType) {

            QR(mats[matType], Q, R, qrType, matTypeFlags[matType]);
            real error = cv::norm(Q * R - mats[matType], cv::NORM_L2);

            if (error > allowedError[qrType]) {
                std::cout <<
                    "Wrong result for " << qrTypes[qrType] << " QR at " <<
                    mats[matType].rows << " x " << mats[matType].cols <<
                    " " << matTypes[matType] << " matrix: ||QR-A|| = " <<
                    error << std::endl;
                return 1;
            }
        }
    }

    return 0;
}