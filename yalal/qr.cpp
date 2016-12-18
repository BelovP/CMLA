#include <iostream>

#include <omp.h>

#include "qr.hpp"
#include "norms.hpp"

using namespace yalal::MatStructure;

namespace yalal {

    void QR_diagonal(cv::Mat_<real> & A, cv::Mat_<real> & Q, cv::Mat_<real> & R) {
        R = cv::Mat_<real>::eye(A.cols, A.cols);
        for (int i = 0; i < R.cols; ++i) {
            R(i, i) = A(i, i);
        }
        Q = cv::Mat_<real>::eye(A.rows, A.cols);
    }

    void QR_upperTri(cv::Mat_<real> & A, cv::Mat_<real> & Q, cv::Mat_<real> & R) {
        R = A;
        Q = cv::Mat_<real>::eye(A.rows, A.cols);
    }

    void QR_GS_Plain(cv::Mat_<real> & A, cv::Mat_<real> & Q, cv::Mat_<real> & R) {
        cv::transpose(A, Q); // Q will be transposed back before return
        R = cv::Mat_<real>::eye(A.cols, A.cols);

        cv::Mat_<real> A_i(1, A.rows); // accomodation for a_i

        for (int i = 0; i < Q.rows; ++i) {
            auto Q_i = Q.row(i);

            auto A_i_col = A.col(i);
            std::copy(A_i_col.begin(), A_i_col.end(), A_i.begin());

            auto R_i_row = R.col(i);

            for (int j = 0; j < i; ++j) {
                auto Q_j = Q.row(j);
                real dotProduct = A_i.dot(Q_j);
                cv::addWeighted(Q_j, -dotProduct, Q_i, 1., 0., Q_i);

                R(j, i) = dotProduct;
            }

            real Q_i_norm = vectorNorm(Q_i);
            Q_i /= Q_i_norm;
            R.row(i) *= Q_i_norm;
        }

        cv::transpose(Q, Q);
    }

    void QR_GS_Modified(cv::Mat_<real> & A, cv::Mat_<real> & Q, cv::Mat_<real> & R) {
        cv::transpose(A, Q); // Q will be transposed back before return
        R = cv::Mat_<real>::eye(A.cols, A.cols);

        cv::Mat_<real> A_i(1, A.rows); // accomodation for a_i

        for (int i = 0; i < Q.rows; ++i) {
            auto Q_i = Q.row(i);

            auto A_i_col = A.col(i);
            std::copy(A_i_col.begin(), A_i_col.end(), A_i.begin());

            auto R_i_row = R.col(i);

            for (int j = 0; j < i; ++j) {
                auto Q_j = Q.row(j);
                real dotProduct = Q_i.dot(Q_j);
                cv::addWeighted(Q_j, -dotProduct, Q_i, 1., 0., Q_i);

                R(j, i) = dotProduct;
            }

            real Q_i_norm = vectorNorm(Q_i);
            Q_i /= Q_i_norm;
            R.row(i) *= Q_i_norm;
        }

        cv::transpose(Q, Q);
    }

    void QR_Householder(cv::Mat_<real> & A, cv::Mat_<real> & Q, cv::Mat_<real> & R) {
        // it's actually Q*, will be transposed back at the end of the function
        Q = cv::Mat_<real>::eye(A.rows, A.rows);
        A.copyTo(R);

        cv::Mat_<real> v(R.rows, 1);  // accomodation for all v's
        cv::Mat_<real> vR(R.rows, 1); // accomodation for v*R[j:,j:] and v*Q[j:]
        cv::Mat_<real> & vQ = vR;     // a reference with different name for convenience

        real vNormSqr, vNorm, v0Sign, v0;

        for (int j = 0; j < R.cols; ++j) {
            // Copy current column to v
            vNormSqr = 0;

            for (int k = j; k < R.rows; ++k) {
                real v_i = R(k, j);
                v(k-j) = v_i;
                vNormSqr += v_i * v_i;
            }

            vNorm = std::sqrt(vNormSqr);
            v0Sign = (v(0) > 0 ? -1. : 1.);
            v0 = v(0);

            v(0) -= v0Sign * vNorm;
            v /= v0Sign * std::sqrt(2. * (vNormSqr - v0Sign * vNorm * v0));

            // Compute v * R[j:,j:]
            vR = 0;

            for (int i1 = j; i1 < R.rows; ++i1) {
//                cv::Mat_<real> vRng = v.rowRange(0, R.cols - j);
//                cv::Mat_<real> Rrng = R.row(i1).colRange(j, R.cols);
//                std::cout << vRng.rows << " " << vRng.cols << std::endl;
//                std::cout << Rrng.rows << " " << Rrng.cols << std::endl;
//                vR.rowRange(0, R.cols - j) += Rrng.dot(vRng.t());
                for (int j1 = j; j1 < R.cols; ++j1) {
                    vR(j1-j) += v(i1-j) * R(i1, j1);
                }
            }

            // R[j:,j:] -= 2. * outer(v, v * R[i:,i:])
            for (int i1 = j; i1 < R.rows; ++i1) {
                for (int j1 = j; j1 < R.cols; ++j1) {
                    R(i1, j1) -= 2. * v(i1-j) * vR(j1-j);
                }
            }

            // Compute v * Q[j:]
            vQ = 0; // Remember: vQ refers to vR

            for (int i1 = j; i1 < Q.rows; ++i1) {
                for (int j1 = 0; j1 < Q.cols; ++j1) {
                    vQ(j1) += v(i1-j) * Q(i1, j1);
                }
            }

            // Q[j:] -= 2. * outer(v, v * Q[j:])
            for (int i1 = j; i1 < Q.rows; ++i1) {
                for (int j1 = 0; j1 < Q.cols; ++j1) {
                    Q(i1, j1) -= 2. * v(i1-j) * vQ(j1);
                }
            }
        }

        cv::transpose(Q, Q);
    }

    void QR_Givens(cv::Mat_<real> & A, cv::Mat_<real> & Q, cv::Mat_<real> & R) {
        // it's actually Q*, will be transposed back at the end of the function
        Q = cv::Mat_<real>::eye(A.rows, A.rows);
        A.copyTo(R);

        for (int j = 0; j < A.cols; ++j) {
            for (int i = A.rows - 1; i > j; --i) {

                // If the element is zero, then there's no need to rotate
                // Otherwise:
                real x_i, x_j, coeff, cos_a, sin_a, x_i_new, x_j_new;

                if (std::abs(R(i,j)) > 5e-7) {
                    // Prepare the rotation
                    x_i = R(i-1,j);
                    x_j = R(i,j);

                    coeff = real(1.) / std::sqrt(x_i*x_i + x_j*x_j);
                    cos_a =  x_i * coeff;
                    sin_a = -x_j * coeff;

                    // Rotate rows
                    for (int k = j; k < R.cols; ++k) {
                        x_i_new = R(i-1,k) * cos_a - R(i,k) * sin_a;
                        x_j_new = R(i-1,k) * sin_a + R(i,k) * cos_a;
                        R(i-1,k) = x_i_new;
                        R(i  ,k) = x_j_new;
                    }

                    // Rotate rows of Q
                    for (int k = 0; k < Q.cols; ++k) {
                        x_i_new = Q(i-1,k) * cos_a - Q(i,k) * sin_a;
                        x_j_new = Q(i-1,k) * sin_a + Q(i,k) * cos_a;
                        Q(i-1,k) = x_i_new;
                        Q(i  ,k) = x_j_new;
                    }
                }
            }
        }

        cv::transpose(Q, Q);
    }

    void QR_Givens_parallel(cv::Mat_<real> & A, cv::Mat_<real> & Q, cv::Mat_<real> & R) {
        // it's actually Q*, will be transposed back at the end of the function
        Q = cv::Mat_<real>::eye(A.rows, A.rows);
        A.copyTo(R);

        for (int j = 0; j < A.cols; ++j) {

            if (A.rows - j > 9) {

//                if (j <= 100) std::cout << R.col(j) << std::endl << std::endl;

                std::vector<int> borders(omp_get_max_threads() + 1);
                for (int rank = 0; rank < borders.size(); ++rank) {
                    int start = rank * (R.rows - j - 1) / omp_get_max_threads();
                    int end = (rank + 1) * (R.rows - j - 1) / omp_get_max_threads();

                    borders[rank] = start;
                    borders[rank + 1] = end;
                }

//            if (j == 0) {
//                std::cout << omp_get_max_threads() << std::endl;
//                for (auto x : borders) {
//                    std::cout << x << " ";
//                }
//                std::cout << std::endl;
//            }

                #pragma omp parallel
                {
                    int nth = omp_get_num_threads();
                    int ith = omp_get_thread_num();
                    
                    for (int i = A.rows - 1 - borders[ith]; i > A.rows - 1 - borders[ith+1]; --i) {
                        real x_i, x_j, coeff, cos_a, sin_a, x_i_new, x_j_new;

                        if (std::abs(R(i, j)) > 5e-7) {
                            // If the element is zero, then there's no need to rotate
                            // Otherwise:

                            // Prepare the rotation
                            x_i = R(i - 1, j);
                            x_j = R(i, j);

                            coeff = real(1.) / std::sqrt(x_i * x_i + x_j * x_j);
                            cos_a = x_i * coeff;
                            sin_a = -x_j * coeff;

                            // Rotate rows
                            for (int k = j; k < R.cols; ++k) {
                                x_i_new = R(i - 1, k) * cos_a - R(i, k) * sin_a;
                                x_j_new = R(i - 1, k) * sin_a + R(i, k) * cos_a;
                                R(i - 1, k) = x_i_new;
                                R(i, k) = x_j_new;
                            }

                            // Rotate rows of Q
                            for (int k = 0; k < Q.cols; ++k) {
                                x_i_new = Q(i - 1, k) * cos_a - Q(i, k) * sin_a;
                                x_j_new = Q(i - 1, k) * sin_a + Q(i, k) * cos_a;
                                Q(i - 1, k) = x_i_new;
                                Q(i, k) = x_j_new;
                            }
                        }
                    }
                }

//                if (j <= 100) std::cout << R.col(j) << std::endl << std::endl;

                for (int q = 1; q < borders.size() - 1; ++q) {
                    
                    int i = A.rows - 1 - borders[q];
                    int iNext = (q == borders.size() - 1 ? i-1 : A.rows - 1 - borders[q+1]);
//                    printf("%d <-> %d\n", i, iNext);
                    
                    real x_i, x_j, coeff, cos_a, sin_a, x_i_new, x_j_new;

                    if (std::abs(R(i, j)) > 5e-7) {
                        // If the element is zero, then there's no need to rotate
                        // Otherwise:

                        // Prepare the rotation
                        x_i = R(iNext, j);
                        x_j = R(i, j);

                        coeff = real(1.) / std::sqrt(x_i * x_i + x_j * x_j);
                        cos_a = x_i * coeff;
                        sin_a = -x_j * coeff;

                        // Rotate rows
                        for (int k = j; k < R.cols; ++k) {
                            x_i_new = R(iNext, k) * cos_a - R(i, k) * sin_a;
                            x_j_new = R(iNext, k) * sin_a + R(i, k) * cos_a;
                            R(iNext, k) = x_i_new;
                            R(i, k) = x_j_new;
                        }

                        // Rotate rows of Q
                        for (int k = 0; k < Q.cols; ++k) {
                            x_i_new = Q(iNext, k) * cos_a - Q(i, k) * sin_a;
                            x_j_new = Q(iNext, k) * sin_a + Q(i, k) * cos_a;
                            Q(iNext, k) = x_i_new;
                            Q(i, k) = x_j_new;
                        }
                    }
                }

//                if (j <= 100) std::cout << R.col(j) << std::endl << std::endl;

            } else {

                for (int i = A.rows - 1; i > j; --i) {

                    // If the element is zero, then there's no need to rotate
                    // Otherwise:
                    real x_i, x_j, coeff, cos_a, sin_a, x_i_new, x_j_new;

                    if (std::abs(R(i, j)) > 5e-7) {
                        // Prepare the rotation
                        x_i = R(i - 1, j);
                        x_j = R(i, j);

                        coeff = real(1.) / std::sqrt(x_i * x_i + x_j * x_j);
                        cos_a = x_i * coeff;
                        sin_a = -x_j * coeff;

                        // Rotate rows
                        for (int k = j; k < R.cols; ++k) {
                            x_i_new = R(i - 1, k) * cos_a - R(i, k) * sin_a;
                            x_j_new = R(i - 1, k) * sin_a + R(i, k) * cos_a;
                            R(i - 1, k) = x_i_new;
                            R(i, k) = x_j_new;
                        }

                        // Rotate rows of Q
                        for (int k = 0; k < Q.cols; ++k) {
                            x_i_new = Q(i - 1, k) * cos_a - Q(i, k) * sin_a;
                            x_j_new = Q(i - 1, k) * sin_a + Q(i, k) * cos_a;
                            Q(i - 1, k) = x_i_new;
                            Q(i, k) = x_j_new;
                        }
                    }
                }
            }
        }

        cv::transpose(Q, Q);
    }

    void QR(cv::Mat_<real> & A, cv::Mat_<real> & Q, cv::Mat_<real> & R,
            int matStructure, int method) {

        assert(A.rows >= A.cols);

        if ((matStructure & DIAGONAL) == matStructure) {
            QR_diagonal(A, Q, R);
        }

        else if ((matStructure & UPPER_TRI) == matStructure) {
            QR_upperTri(A, Q, R);
        }

        else if ((matStructure | HESSENBERG) == HESSENBERG) {
            // Regardless of choice, do Givens QR.
            // QR_Givens is already optimized for many zero elements below the diagonal
            QR_Givens(A, Q, R);
        }

        else {
            switch (method) {
                case QRMethod::GS: {
                    QR_GS_Plain(A, Q, R);
                    break;
                }
                case QRMethod::GS_MODIFIED: {
                    QR_GS_Modified(A, Q, R);
                    break;
                }
                case QRMethod::HOUSEHOLDER: {
                    QR_Householder(A, Q, R);
                    break;
                }
                case QRMethod::GIVENS: {
                    QR_Givens(A, Q, R);
                    break;
                }
                default: {
                    assert(false && "Wrong 'QR method' parameter");
                }
            }
        }
    }

};