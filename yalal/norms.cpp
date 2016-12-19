#include "norms.hpp"
#include <cmath>
#include <array>
#include <iostream>

real fastPower(real x, int p) {
    real retval = 1;
    while (p) {
        if (p & 1)
            retval *= x;
        x *= x;
        p >>= 1;
    }
    return retval;
}

namespace yalal {

    /*
     Вспомогательный метод для хаусхолдера, возвращает вектор v (который в алгоритме
     хаусхолдера), для вектора "a" который хотим привести к виду (какое-то число,0,0..,0).
     Возвращает false, если вектор "а" уже приведен к этому виду
    */

    real householderAux(cv::Mat_<real> & a, cv::Mat_<real> & e) {

        // Assumption: e is a ROW vector, e.total() == a.total()

        real tailNormSqr = 0;
        for (int k = 1; k < a.total(); ++k) {
            tailNormSqr += a(k) * a(k);
            e(k) = a(k);
        }
        e(0) = real(1.);

        real nonzero;

        if (tailNormSqr < 1e-6) {
            nonzero = 0;
            e /= real(M_SQRT2);
        } else {
            real norm = std::sqrt(a(0) * a(0) + tailNormSqr);
            if (a(0) <= 0) {
                e(0) = a(0) - norm;
            } else {
                e(0) = - tailNormSqr / (a(0) + norm);
            }

            nonzero = real(2) * e(0) * e(0) / (tailNormSqr + e(0) * e(0));
            e /= real(M_SQRT2) * e(0);
        }

        return nonzero;
    }

    /*
     двусторонний хаусхолдер для бидиагонолизации, его реализация описана в
     книжке "Handbook of Linear Algera", страница 45-5(738 of 1402) алгоритм 1b,
     там в целом немного другая реализация, у меня комбинация этого алгоритма
     и алгоритма из книги "Golub G.H., Van Loan C.F. - Matrix computations"
     (первая ссылка в гугле), страница 252, алгоритм 5.4.
    */

    void twoSideHouseholder(cv::Mat_<real> & A, cv::Mat_<real> & B) {

        cv::Mat_<real> tmpVec, outer;

        if (A.rows > A.cols) {
            cv::transpose(A, B);
        } else {
            A.copyTo(B);
        }

        // Householder for columns
        for (int i = 0; i < B.cols; ++i) {
            cv::Mat_<real> a = B.rowRange(i, B.rows).col(i);
            cv::Mat_<real> v(1, B.rows - i);
            real nonzero = householderAux(a, v);

            cv::Mat_<real> submatB = B.rowRange(i, B.rows).colRange(i, B.cols);
            cv::gemm(v, submatB, 1.0, cv::noArray(), 0., tmpVec);

            // subtract outer product
            for (int i1 = 0; i1 < submatB.rows; ++i1) {
                for (int j1 = 0; j1 < submatB.cols; ++j1) {
                    submatB(i1,j1) -= 2. * nonzero * v(i1) * tmpVec(j1);
                }
            }

            // right householder for rows (almost the same as for columns)
            if (i <= B.rows - 3) {
                a = B.row(i).colRange(i+1, B.cols);
                nonzero = householderAux(a, v);

                submatB = B.rowRange(i, B.rows).colRange(i+1, B.cols);
                v = v.colRange(0, B.cols - i - 1);
                cv::gemm(submatB, v.t(), 1.0, cv::noArray(), 0., tmpVec);

                // subtract outer product
                for (int i1 = 0; i1 < submatB.rows; ++i1) {
                    for (int j1 = 0; j1 < submatB.cols; ++j1) {
                        submatB(i1,j1) -= 2. * nonzero * tmpVec(i1) * v(j1);
                    }
                }
            }
        }

        if (A.cols > A.rows) {
            // TODO smart transpose B
            cv::transpose(B, B);
        }
    }

    /*
     Это вспомогательная функция для функции mySVD, она возвращает количество
     сингулярных значений меньше чем "edge". Этот метод есть в книжке
     "Handbook of Linear Algera" страница 45-9(742 of 1402) алгоритм 4b,
     там есть опечатка, параметр "мю" нужно возвести в квадрат
    */
    int countSingularValuesLowerThan(cv::Mat_<real> B, real edge) {
        edge = edge * edge;
        real t = -edge;
        int count = 0;
        int n = B.rows - 1;

        for (int k = 0; k < n; ++k) {
            real d = B(k, k) * B(k, k) + t;
            if (d < 0) {
                ++count;
            }
            t = t * (B(k,k+1) * B(k,k+1) / d) - edge;
        }

        if (B(n, n) * B(n, n) + t < 0) {
            ++count;
        }

        return count;
    }

    /*
     Это сам SVD по алгоритму bisection
     Этот алгоритм есть в книжке "Handbook of Linear Algebra" страница 45-9(742 of 1402),
     алгоритм 4a. Ему на вход подается двудиагональная матрица "B", интервал в котором мы
     ищем сингулярные значения "a" и "b" и точность с которой мы хотим получить сингулярные
     значения "tol".

     !!! Матрицу обязательно нужно привести к бидиагональному виду, то есть можно прямо
     внутрь truncSVD первой строчкой вписать B = twoSideHouseholder(B)[1]
    */

    real truncSVDForMaxSingularValue(cv::Mat_<real> & B, real a, real b, real tol) {
        int n = B.rows;
        real n_a = countSingularValuesLowerThan(B, a);
        real n_b = countSingularValuesLowerThan(B, b);

        if (n_a == n_b) {
            return 0;
        }

        cv::Mat_<real> retval(1, n);
        std::vector<real> worklist = {a, n_a, b, n_b};

        while (true) {

            real low = worklist[0];
            int n_low = worklist[1];
            real up = worklist[2];
            int n_up = worklist[3];

            real mid = (low + up) / real(2.);

            if ((up - low) < tol) {
                for (int i = n_low; i < n_up; ++i) {
                    retval(i - n_a) = mid;
                }
                break;
            } else {
                int n_mid = countSingularValuesLowerThan(B, mid);

                if (n_mid > n_low) {
                    worklist[0] = low;
                    worklist[1] = n_low;
                    worklist[2] = mid;
                    worklist[3] = n_mid;
                }

                if (n_up > n_mid) {
                    worklist[0] = mid;
                    worklist[1] = n_mid;
                    worklist[2] = up;
                    worklist[3] = n_up;
                }
            }
        }

        return retval(retval.total() - 1);
    }

    real vectorNorm(cv::Mat_<real> vec, int p) {

        switch (p) {
            case VectorNorm::INF: {
                real retval = 0;
                for (auto & x : vec) {
                    retval = std::max(retval, std::abs(x));
                }
                return retval;
            }

            case 1: {
                real retval = 0;
                for (auto & x : vec) {
                    // should be more precise if parallelized
                    retval += std::abs(x);
                }
                return retval;
            }

            case 2:
            case VectorNorm::L2_SQUARED: {
                real retval = 0;
                for (auto & x : vec) {
                    retval += x*x;
                }
                return (p == 2 ? std::sqrt(retval) : retval);
            }

            default: {
                real retval = 0;
                for (auto & x : vec) {
                    retval += fastPower(x, p);
                }
                return std::pow(retval, real(1) / p);
            }
        }
    }

    real matrixNorm(cv::Mat_<real> A, int type, int matStructure) {

        switch (type) {
            case MatrixNorm::INF: {
                real retval = 0;
                for (int i = 0; i < A.rows; ++i) {
                    retval = std::max(retval, vectorNorm(A.row(i), 1));
                }
                return retval;
            }

            case MatrixNorm::FROBENIUS:
            case MatrixNorm::FROBENIUS_SQUARED: {
                real retval = 0;

                if (matStructure == MatStructure::UPPER_BIDIAG) {
                    int kLimit = std::min(A.rows, A.cols);
                    for (int k = 0; k < std::min(A.rows, A.cols) - 1; ++k) {
                        retval += A(k,k) * A(k,k);
                        retval += A(k,k+1) * A(k,k+1);
                    }
                    retval += A(kLimit, kLimit) * A(kLimit, kLimit);
                } else {
                    for (auto & x : A) {
                        // should be more precise if parallelized
                        retval += x * x;
                    }
                }
                return (type == MatrixNorm::FROBENIUS ? std::sqrt(retval) : retval);
            }

            case 1: {
                real retval = 0;
                for (int j = 0; j < A.cols; ++j) {
                    retval = std::max(retval, vectorNorm(A.col(j), 1));
                }
                return retval;
            }

            case 2: {
                cv::Mat_<real> B; // bidiagonal
                twoSideHouseholder(A, B);
                return truncSVDForMaxSingularValue(
                        B, 0,
                        // Frobenius norm is an upper bound for the max singular value
                        matrixNorm(B, MatrixNorm::FROBENIUS, MatStructure::UPPER_BIDIAG),
                        1e-3);
            }

            default: {
                assert(false && "Wrong matrix norm type");
                return -1;
            }
        }
    }

}; // namespace yalal
