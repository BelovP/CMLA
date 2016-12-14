#include <iostream>
#include <chrono>

#include "cholesky.hpp"

using yalal::Cholesky;
using namespace yalal::MatStructure;

int main(int argc, char* argv[]) {

    bool verbose = (argc > 1 && !strcmp(argv[1], "-v"));
    std::cout.precision(7);
    std::cout << std::fixed;

    std::vector<const char*> matTypes =
            {"random", "random", "diagonal", "diagonal (treated as arbitrary)",
             "Hilbert", "tridiagonal"};
    std::vector<MatStructure> matTypeFlags =
            {MatStructure::ARBITRARY, MatStructure::ARBITRARY, MatStructure::DIAGONAL,
             MatStructure::ARBITRARY, MatStructure::ARBITRARY, MatStructure::TRIDIAGONAL};

    std::vector<cv::Mat_<real>> mats(matTypes.size());

    // small random symmetric positive definite
    mats[0].create(7, 7);
    cv::randu(mats[0], -1., 1.);
    mats[0] = mats[0].t() * mats[0];

    // large random symmetric positive definite
    mats[1].create(433, 433);
    cv::randu(mats[1], -1., 1.);
    mats[1] = mats[1].t() * mats[1];

    // diagonal
    mats[2] = cv::Mat_<real>::zeros(477, 477);
    for (int i = 0; i < mats[2].rows; ++i) {
        mats[2](i, i) = cv::randu<real>() + 0.5;
    }
    
    // diagonal
    mats[3] = mats[2];

    // Hilbert
    mats[4].create(127, 127);
    for (int i = 0; i < mats[4].rows; ++i) {
        for (int j = 0; j < mats[4].cols; ++j) {
            mats[4](i, j) = 1. / (i + j + 1);
        }
    }

    // tridiagonal
    mats[5] = cv::Mat_<real>::zeros(457, 457);
    for (int i = 0; i < mats[5].rows; ++i) {
        mats[5](i, i) = cv::randu<real>() * 10 + 21.;
        if (i != mats[5].rows - 1) {
            mats[5](i, i+1) = cv::randu<real>() * 10;
            mats[5](i+1, i) = mats[5](i, i+1);
        }
    }

    typedef std::chrono::microseconds Microseconds;
    typedef std::chrono::steady_clock Clock;
    typedef Clock::time_point Time;

    cv::Mat_<real> L;

    for (int matType = 0; matType < matTypes.size(); ++matType) {

        Time start = Clock::now();

        Cholesky(mats[matType], L, matTypeFlags[matType]);

        Time end = Clock::now();
        int duration = std::chrono::duration_cast<Microseconds>(end - start).count() / 1000;

        real error = cv::norm(L * L.t() - mats[matType], cv::NORM_L2);

        if (verbose) {
            std::cout << "||LL* - A|| = " << error << ",\t"
            << duration << " ms"
            << ",\t" << mats[matType].rows << " x " << mats[matType].cols
            << " " << matTypes[matType]
            << std::endl;
        }

        if (error > 5e-3) {
            std::cout <<
            "Wrong result for Cholesky at " <<
            mats[matType].rows << " x " << mats[matType].cols <<
            " " << matTypes[matType] << " matrix: ||LL*-A|| = " <<
            error << std::endl;
            return 1;
        }

        real upperTriangleMax = 0;
        for (int i = 0; i < L.rows; ++i) {
            for (int j = i + 1; j < L.cols; ++j) {
                upperTriangleMax = std::max(upperTriangleMax, std::abs(L(i, j)));
            }
        }

        if (upperTriangleMax > 4e-5) {
            std::cout <<
            "Wrong result for Cholesky at " <<
            mats[matType].rows << " x " << mats[matType].cols <<
            " " << matTypes[matType] << " matrix: L is not triangular"
                    " enough (" << upperTriangleMax << ")" << std::endl;
            return 1;
        }
    }

    return 0;
}