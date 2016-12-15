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

    bool householderAux(cv::Mat_<real> & a, cv::Mat_<real> & e) {

        if (vectorNorm(a, VectorNorm::L2_SQUARED) == a(0) * a(0)) {
            return false;
        } else {
            a.copyTo(e);
            real sign = (a(0) > 0 ? real(1.0) : real(-1.0));
            if (a(0) == 0) {
                sign = 0;
            }
            e(0) += sign * vectorNorm(a);
            e /= vectorNorm(e);
            return true;
        }
    }

    /*
     двусторонний хаусхолдер для бидиагонолизации, его реализация описана в
     книжке "Handbook of Linear Algera", страница 45-5(738 of 1402) алгоритм 1b,
     там в целом немного другая реализация, у меня комбинация этого алгоритма
     и алгоритма из книги "Golub G.H., Van Loan C.F. - Matrix computations"
     (первая ссылка в гугле), страница 252, алгоритм 5.4.
    */

    void twoSideHouseholder(cv::Mat_<real> & A, cv::Mat_<real> & B) {

        if (A.rows > A.cols) {
            cv::transpose(A, B);
        } else {
            A.copyTo(B);
        }

//        // U = eye(B.rows)
//        U.create(B.rows, B.rows);
//        U = 0;
//        for (int k = 0; k < B.rows; ++k) {
//            U(k,k) = real(1.);
//        }
//
//        // V = eye(B.cols)
//        V.create(B.cols, B.cols);
//        V = 0;
//        for (int k = 0; k < B.cols; ++k) {
//            V(k,k) = real(1.);
//        }

        cv::Mat_<real> Q = cv::Mat_<real>::eye(B.rows, B.rows);
        cv::Mat_<real> v(B.rows, 1);

        // Householder for columns
        for (int i = 0; i < B.cols; ++i) {
            cv::Mat_<real> a = B.rowRange(i, B.rows).col(i);
            bool nonzero = householderAux(a, v);

            Q = 0;
            for (int k = 0; k < Q.rows; ++k) {
                Q(k,k) = real(1.);
            }

            cv::Mat_<real> Qsub = Q.rowRange(i, Q.rows).colRange(i, Q.cols);
            if (nonzero) {
                for (int i1 = 0; i1 < a.total(); ++i1) {
                    for (int j1 = 0; j1 < a.total(); ++j1) {
                        Qsub(i1, j1) -= real(2.) * v(i1) * v(j1);
                    }
                }
            }

            cv::gemm(Q, B, 1.0, cv::noArray(), 0., B);
//            cv::gemm(U, Q, 1.0, cv::noArray(), 0., U);

            // right householder for rows (almost the same as for columns)
            if (i <= B.rows - 3) {
                int BWidth = B.cols - i;
                Q = 0;
                for (int k = 0; k < Q.rows; ++k) {
                    Q(k,k) = real(1.);
                }

                a = B.row(i).colRange(i+1, B.cols);
                nonzero = householderAux(a, v);

                // TODO optimize for Q
                Qsub = Q.rowRange(i+1, Q.rows).colRange(i+1, Q.cols);
                if (nonzero) {
                    for (int i1 = 0; i1 < a.total(); ++i1) {
                        for (int j1 = 0; j1 < a.total(); ++j1) {
                            Qsub(i1, j1) -= real(2.) * v(i1) * v(j1);
                        }
                    }
                }

                cv::gemm(B, Q, 1.0, cv::noArray(), 0., B);
//                cv::gemm(Q, V, 1.0, cv::noArray(), 0., V);
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
     Этот алгоритм есть в книжке "Handbook of Linear Algera" страница 45-9(742 of 1402),
     алгоритм 4a. Ему на вход подается двудиагональная матрица "B", интервал в котором мы
     ищем сингулярные значения "a" и "b" и точность с которой мы хотим получить сингулярные
     значения "tol".

     !!! Матрицу обязательно нужно привести к бидиагональному виду, то есть можно прямо
     внутрь truncSVD первой строчкой вписать B = twoSideHouseholder(B)[1]
    */

    real truncSVDForMaxSingularValue(cv::Mat_<real> & B, real a, real b, real tol) {
        int n = B.rows;
        int n_a = countSingularValuesLowerThan(B, a);
        int n_b = countSingularValuesLowerThan(B, b);

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

//            for (auto x : worklist) {
//                std::cout << x << " ";
//            }
//            std::cout << ", " << countSingularValuesLowerThan(B, 0.9375) << std::endl;

            real mid = (low + up) / real(2.);

            if ((up - low) < tol) {
                for (int i = n_low; i < n_up; ++i) {
                    retval(i - n_a) = mid;
                }
                break;
            } else {
                int n_mid = countSingularValuesLowerThan(B, mid);
//                std::cout << n_mid << std::endl;

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
                for (auto & x : A) {
                    // should be more precise if parallelized
                    retval += x * x;
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
//                std::cout << B << std::endl;
                return truncSVDForMaxSingularValue(B, 0., 20., 1e-5);
            }

            default: {
                assert(false && "Wrong matrix norm type");
                return -1;
            }
        }
    }

}; // namespace yalal