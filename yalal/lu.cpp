#include "lu.hpp"
#include "norms.hpp"
#include <vector>

const real close_to_zero_eps = 1e-8;

using std::pair;
using std::vector;

namespace yalal {

    // Returns false if there is no good pivoting (A_{k, k} close to 0).
    bool Pivot(cv::Mat_<real> & A, cv::Mat_<real> & P, vector<int> & row_permutations, int k) {
        int n = A.rows;
        int row_to_swap = k;
        int i;

        for (i = k + 1; i < n; ++i) {
            if (std::abs(A(i, k)) > std::abs(A(row_to_swap, k))) {
                row_to_swap = i;
            }
        }

        // Return false if we divide on a very small number.
        if (std::abs(A(row_to_swap, k)) < close_to_zero_eps) {
            return false;
        }

        P(k, row_permutations[row_to_swap]) = 1;

        if (row_to_swap == k) {
            return true;
        }

        std::swap(row_permutations[row_to_swap], row_permutations[k]);

        for (int i = 0; i < n; ++i) {
            std::swap(A(k, i), A(row_to_swap, i));
        }

        return true;
    }

// Returns false if LU is not possible. PA = LU
    bool _LU_compute(cv::Mat_<real> & A, cv::Mat_<real> & P, vector<pair<int, int> > & borders, bool pivoting = false) {
        int n = A.rows;
        int col, i, j;
        real mul1, mul2;
        // Stores the indices of rows in the initial matrix A before permutations.
        vector<int> row_permutations;

        if (pivoting) {
            row_permutations.resize(n);
            for (i = 0; i < row_permutations.size(); ++i) {
                row_permutations[i] = i;
            }
        }

        if (pivoting) {
            P.create(n, n);
        }

        for (col = 0; col < n; ++col) {
            if (pivoting && !Pivot(A, P, row_permutations, col)) {
                return false;
            }

            for (i = col + 1; i < n; ++i) {
                A(i, col) /= A(col, col);
            }

            for (i = col + 1; i < n; ++i) {
                for (j = col + 1; j <= borders[i].second; ++j) {     // < n
                    A(i, j) = A(i, j) - A(i, col) * A(col, j);
                }
            }
        }

        return true;
    }

    bool LU_internal(cv::Mat_<real> & A,
                     cv::Mat_<real> & P, int matStructure, bool pivoting) {
        // Non-empty square matrices

        assert(A.rows && A.rows == A.cols);

        auto n = A.rows;
        vector<pair<int, int> > borders(n);

        if (matStructure == MatStructure::DIAGONAL) {
            if (pivoting) P.create(n, n);
            // We don't need to do anything here.

            return true;
        } else if (matStructure == MatStructure::LOWER_TRI) {
            if (pivoting) P.create(n, n);
            // We have to adjust numbers on diagonal.

            for (int i = 0; i < n; ++i) {
                borders[i].first = 0;
                borders[i].second = i;
            }

            return _LU_compute(A, P, borders, pivoting);
        } else if (matStructure == MatStructure::UPPER_TRI) {
            if (pivoting) P.create(n, n);
            // The answer is the matrix itself.

            return true;
        } else if (matStructure == MatStructure::HESSENBERG) {
            if (pivoting) P.create(n, n);
            for (int i = 0; i < n; ++i) {
                borders[i].first = i - 1;
                borders[i].second = n - 1;
            }
            borders[0].first = 0;

            return _LU_compute(A, P, borders, pivoting);
        } else if (matStructure == MatStructure::TRIDIAGONAL) {
            if (pivoting) P.create(n, n);
            for (int i = 0; i < n; ++i) {
                borders[i].first = i - 1;
                borders[i].second = i + 1;
            }
            borders[0].first = 0;
            borders[n - 1].second = n - 1;

            return _LU_compute(A, P, borders, pivoting);
        } else {
            for (int i = 0; i < n; ++i) {
                borders[i].first = 0;
                borders[i].second = n - 1;
            }
            return _LU_compute(A, P, borders, pivoting);
        }
    }

    // LU algorithm without pivoting.
    bool LU(cv::Mat_<real> A, int matStructure) {
        cv::Mat_<real> P;
        return LU_internal(A, P, matStructure, false);
    }

    // LU algorithm with partial pivoting. PA = LU
    bool LU(cv::Mat_<real> A, cv::Mat_<real> & P, int matStructure) {
        return LU_internal(A, P, matStructure, true);
    }

    bool LU_mt(cv::Mat_<real> A, int matStructure) {
        assert(A.rows && A.rows == A.cols);
        auto n = A.rows;

        //vector<real> A_lined(n * n);
        real A_lined[n * n];
        int it = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A_lined[it++] = A(i, j);
            }
        }

        for(int col = 0; col < n; ++col) {
            int i;

            // #pragma omp parallel for private(i) shared(A_lined)
            for(i = col + 1; i < n; ++i){
                real diag_mul = A_lined[i * n + col] / A_lined[col * n + col];
                int j;
                for(j = col + 1; j < n; ++j){
                    A_lined[i * n + j] = A_lined[i * n + j] - diag_mul * A_lined[col * n + j];
                }
                A_lined[i * n + col] = diag_mul;
            }
        }
        it = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A(i, j) = A_lined[it++];
            }
        }

        return true;
    }

    void RecoverLU(cv::Mat_<real>& A, cv::Mat_<real>& L, cv::Mat_<real>& U) {
        int n = A.rows;
        L.create(n, n);
        U.create(n, n);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                L(i, j) = 0;
                U(i, j) = 0;
            }
        }

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (j < i) {
                    L(i, j) = A(i, j);
                }
                else {
                    U(i, j) = A(i, j);
                }
                if (j == i) {
                    L(i, j) = 1;
                }
            }
        }
    }
}