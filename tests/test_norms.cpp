#include <iostream>
#include "norms.hpp"

const real EPS = 1e-4;

using yalal::VectorNorm;
using yalal::vectorNorm;

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

    return 0;
}