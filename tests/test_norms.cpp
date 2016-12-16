#include <iostream>
#include "norms.hpp"

const real EPS = 1e-4;

using namespace yalal::VectorNorm;
using yalal::vectorNorm;

using namespace yalal::MatrixNorm;
using yalal::matrixNorm;

using namespace yalal::MatStructure;

bool testVectorNorm(cv::Mat_<real> & vec) {

    for (auto p: {int(VectorNorm::L2_SQUARED), int(VectorNorm::INF), 1, 2}) {

        int openCVNormType = p;
        if (p == VectorNorm::L2_SQUARED) {
            openCVNormType = cv::NORM_L2SQR;
        } else if (p == VectorNorm::INF) {
            openCVNormType = cv::NORM_INF;
        } else if (p == 1) {
            openCVNormType = cv::NORM_L1;
        } else if (p == 2) {
            openCVNormType = cv::NORM_L2;
        }

        real exact = cv::norm(vec, openCVNormType);
        real our   = vectorNorm(vec, p);

        // Relative error
        if (std::abs(exact - our) > EPS * std::abs(exact)) {
            std::cout << "Vector norm test failed." << std::endl;
            std::cout << "Vector:" << std::endl << vec << std::endl;
            std::cout << "Expected norm with p = " << p << ": " << exact << std::endl;
            std::cout << "Got " << our << std::endl;

            return false;
        }
    }

    return true;
}

int main() {

    std::vector<cv::Mat_<real>> testVectors(4);

    testVectors[0] = (cv::Mat_<real>(1, 1) << -17.3);

    testVectors[1].create(1, 65536);
    testVectors[1] = 0;

    testVectors[2].create(1, 65536);
    cv::randu(testVectors[2], -1.1, 1.1);

    testVectors[3].create(1, 65536);
    cv::randu(testVectors[3], -1.1, 1.1);

    for (cv::Mat_<real> & vec : testVectors) {
        if (not testVectorNorm(vec)) {
            return 1;
        }
    }

    std::vector<const char*> matTypes =
            {"random", "diagonal", "diagonal (treated as arbitrary)",
             "Hilbert", "upper triangular", "lower triangular",
             "tridiagonal", "sparse"};

    std::vector<MatStructure> matTypeFlags =
            {ARBITRARY, DIAGONAL, ARBITRARY,
             ARBITRARY, UPPER_TRI, LOWER_TRI,
             TRIDIAGONAL, ARBITRARY};

    std::vector<cv::Mat_<real>> mats(matTypes.size());

    // random
    mats[0].create(233, 233);
    cv::randu(mats[0], -1., 1.);

//    mats[0] = (cv::Mat_<real>(3,3) << 3,2,1,8,1,8,0,3,3);

    // diagonal
    mats[1] = cv::Mat_<real>::zeros(133, 133);
    for (int i = 0; i < mats[1].rows; ++i) {
        mats[1](i, i) = 1;//(cv::randu<real>() - 0.5) * 100;
    }

    // diagonal
    mats[2] = mats[1];

    // Hilbert
    mats[3].create(100, 100);
    for (int i = 0; i < mats[3].rows; ++i) {
        for (int j = 0; j < mats[3].cols; ++j) {
            mats[3](i, j) = 1. / (i + j + 1);
        }
    }

    // upper triangular
    mats[4] = cv::Mat_<real>::zeros(50, 50);
    for (int i = 0; i < mats[4].rows; ++i) {
        for (int j = i; j < mats[4].cols; ++j) {
            mats[4](i, j) = (cv::randu<real>() - 0.5) * 2.;
        }
    }

    // lower triangular
    cv::transpose(mats[4], mats[5]);

    // tridiagonal
    mats[6] = cv::Mat_<real>::zeros(50, 50);
    for (int i = 0; i < mats[6].rows; ++i) {
        for (int j = std::max(0, i-1); j <= std::min(mats[6].cols-1, i+1); ++j) {
            mats[6](i, j) = (cv::randu<real>() - 0.5) * 10;
        }
    }

    rand(); rand();
    const int sparseMatSize = 129;
    mats[7] = cv::Mat_<real>::zeros(sparseMatSize, sparseMatSize);
    std::vector<int> permutation(sparseMatSize);
    for (int i = 0; i < sparseMatSize; ++i) {
        permutation[i] = i;
    }
    std::random_shuffle(permutation.begin(), permutation.end());
    for (int i = 0; i < sparseMatSize; ++i) {
        mats[7](i, permutation[i]) = 5 * real(i) / sparseMatSize;
        mats[7](i, rand() % sparseMatSize) += (cv::randu<real>() - 0.5) * 4;
        mats[7](i, rand() % sparseMatSize) += (cv::randu<real>() - 0.5) * 4;
    }

    // Test spectral norm
    for (int i = 0; i < mats.size(); ++i) {
        cv::SVD svd(mats[i], cv::SVD::NO_UV);
        double exact;
        cv::minMaxLoc(svd.w, 0, &exact);

        real our = matrixNorm(mats[i]);

        if (std::abs(our - exact) > 1e-4) {
            std::cout << "Wrong matrix norm at " << mats[i].rows << " x "
                      << mats[i].cols << " " << matTypes[i] << " matrix: expected "
                      << exact << " but got " << our << std::endl;
            return 1;
        }

        std::cout << i << std::endl;
    }

    return 0;
}