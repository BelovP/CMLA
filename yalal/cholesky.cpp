#include "cholesky.hpp"
#include <iostream>
namespace yalal {

    void Cholesky(cv::Mat_<real> & A, cv::Mat_<real> & L,
                  int matStructure, int method) {

        assert(A.rows == A.cols && "Cholesky needs a symmetric matrix");

        L.create(A.rows, A.cols);
        L = 0;

        cv::Mat_<real> A_i = A.clone();

        for (int i = 0; i < A_i.rows; ++i) {

            // L *= L_i (fill L[i:,i])
            real sqrtAii = std::sqrt(A_i(i, i));
            real invSqrtAii = real(1) / sqrtAii;

            L(i, i) = sqrtAii;
            for (int k = i+1; k < L.rows; ++k) {
                L(k, i) = invSqrtAii * A_i(k, i);
            }

            // A_i[i+1:, i+1:] -= (1/a_ii) * b*b^T
            real invAii = real(1) / A_i(i, i);

            for (int i1 = i+1; i1 < A_i.rows; ++i1) {
                for (int j1 = i+1; j1 < A_i.cols; ++j1) {
                    A_i(i1, j1) -= invAii * A_i(i, i1) * A_i(i, j1);
                }
            }
        }
    }

};