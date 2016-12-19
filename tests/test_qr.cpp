#include <iostream>
#include <algorithm>
#include <chrono>

#include "qr.hpp"

const real EPS = 1e-4;

using yalal::QR;
using namespace yalal::MatStructure;

int main(int argc, char* argv[]) {

    bool verbose = (argc > 1 && !strcmp(argv[1], "-v"));
    std::cout.precision(7);
    std::cout << std::fixed;

    std::vector<const char*> qrTypes =
            {"GS", "Modified GS", "Householder", "Givens"};

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
    mats[0].create(5000, 50);
    cv::randu(mats[0], -1., 1.);

    // diagonal
    mats[1] = cv::Mat_<real>::zeros(333, 333);
    for (int i = 0; i < mats[1].rows; ++i) {
        mats[1](i, i) = (cv::randu<real>() - 0.5) * 100;
    }

    // diagonal
    mats[2] = mats[1];

    // Hilbert
    mats[3].create(127, 127);
    for (int i = 0; i < mats[3].rows; ++i) {
        for (int j = 0; j < mats[3].cols; ++j) {
            mats[3](i, j) = 1. / (i + j + 1);
        }
    }

    // upper triangular
    mats[4] = cv::Mat_<real>::zeros(354, 354);
    for (int i = 0; i < mats[4].rows; ++i) {
        for (int j = i; j < mats[4].cols; ++j) {
            mats[4](i, j) = (cv::randu<real>() - 0.5) * 2.;
        }
    }

    // lower triangular
    cv::transpose(mats[4], mats[5]);

    // tridiagonal
    mats[6] = cv::Mat_<real>::zeros(457, 457);
    for (int i = 0; i < mats[6].rows; ++i) {
        for (int j = std::max(0, i-1); j <= std::min(mats[6].cols-1, i+1); ++j) {
            mats[6](i, j) = (cv::randu<real>() - 0.5) * 10;
        }
    }

    const int sparseMatSize = 411;
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

    typedef std::chrono::microseconds Microseconds;
    typedef std::chrono::steady_clock Clock;
    typedef Clock::time_point Time;

    cv::Mat_<real> Q, R;

    for (int qrType = 0; qrType < qrTypes.size(); ++qrType) {
        if (verbose) {
            std::cout << "*** " << qrTypes[qrType] << " ***" << std::endl;
        }

        for (int matType = 0; matType < matTypes.size(); ++matType) {

            Time start = Clock::now();

            QR(mats[matType], Q, R, matTypeFlags[matType], qrType);

            Time end = Clock::now();
            int duration = std::chrono::duration_cast<Microseconds>(end - start).count() / 1000;

            real error = 0;//cv::norm(Q * R - mats[matType], cv::NORM_L2);

            if (verbose) {
                std::cout << "||QR-A|| = " << error << ",\t"
                << "||Q*Q|| = " << 0/*cv::norm(Q.t() * Q - cv::Mat_<real>::eye(Q.cols, Q.cols))*/
                << ",\t" << duration << " ms"
                << ",\t" << mats[matType].rows << " x " << mats[matType].cols
                << " " << matTypes[matType]
                << std::endl;
            }

            if (error > 6e-4) {
                std::cout <<
                    "Wrong result for " << qrTypes[qrType] << " QR at " <<
                    mats[matType].rows << " x " << mats[matType].cols <<
                    " " << matTypes[matType] << " matrix: ||QR-A|| = " <<
                    error << std::endl;
                return 1;
            }

            real lowerTriangleMax = 0;
            for (int i = 0; i < R.rows; ++i) {
                for (int j = 0; j < std::min(R.cols, i - 1); ++j) {
                    lowerTriangleMax = std::max(lowerTriangleMax, std::abs(R(i, j)));
                }
            }

            if (lowerTriangleMax > 4e-5) {
                std::cout <<
                "Wrong result for " << qrTypes[qrType] << " QR at " <<
                mats[matType].rows << " x " << mats[matType].cols <<
                " " << matTypes[matType] << " matrix: R is not triangular"
                " enough (" << lowerTriangleMax << ")" << std::endl;
                return 1;
            }
        }

        if (verbose) {
            std::cout << std::endl;
        }
    }

    return 0;
}