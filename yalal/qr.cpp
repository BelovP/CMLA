#include "qr.hpp"
#include "norms.hpp"
#include <iostream>

namespace yalal {

    void QR_diagonal(cv::Mat_<real> A, cv::Mat_<real> &Q, cv::Mat_<real> &R) {
        R = cv::Mat_<real>::eye(A.cols, A.cols);
        for (int i = 0; i < R.cols; ++i) {
            R.at<real>(i, i) = A.at<real>(i, i);
        }
        Q = cv::Mat_<real>::eye(A.rows, A.cols);
    }

    void QR_upperTri(cv::Mat_<real> A, cv::Mat_<real> &Q, cv::Mat_<real> &R) {
        R = A;
        Q = cv::Mat_<real>::eye(A.rows, A.cols);
    }

    void QR_GS(cv::Mat_<real> A, cv::Mat_<real> &Q, cv::Mat_<real> &R) {
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

                R.at<real>(j, i) = dotProduct;
            }

            real Q_i_norm = vectorNorm(Q_i);
            Q_i /= Q_i_norm;
            R.row(i) *= Q_i_norm;
        }

        cv::transpose(Q, Q);
    }

    void QR_GS_Modified(cv::Mat_<real> A, cv::Mat_<real> &Q, cv::Mat_<real> &R) {
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

                R.at<real>(j, i) = dotProduct;
            }

            real Q_i_norm = vectorNorm(Q_i);
            Q_i /= Q_i_norm;
            R.row(i) *= Q_i_norm;
        }

        cv::transpose(Q, Q);
    }

    void QR_Householder(cv::Mat_<real> A, cv::Mat_<real> &Q, cv::Mat_<real> &R) {
        // TODO
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

                R.at<real>(j, i) = dotProduct;
            }

            real Q_i_norm = vectorNorm(Q_i);
            Q_i /= Q_i_norm;
            R.row(i) *= Q_i_norm;
        }

        cv::transpose(Q, Q);
    }

    void QR_Givens(cv::Mat_<real> A, cv::Mat_<real> &Q, cv::Mat_<real> &R) {
        // it's actually Q*, will be transposed back at the end of the function
        Q = cv::Mat_<real>::eye(A.rows, A.rows);
        A.copyTo(R);

        for (int j = 0; j < A.cols; ++j) {
            for (int i = A.rows - 1; i > j; --i) {

                // If the element is zero, then there's no need to rotate
                // Otherwise:
                real x_i, x_j, coeff, cos_a, sin_a, x_i_new, x_j_new;

                if (std::abs(R.at<real>(i,j)) > 5e-7) {
                    // Prepare the rotation
                    x_i = R.at<real>(i-1,j);
                    x_j = R.at<real>(i,j);

                    coeff = real(1.) / std::sqrt(x_i*x_i + x_j*x_j);
                    cos_a =  x_i * coeff;
                    sin_a = -x_j * coeff;

                    // Rotate rows
                    for (int k = j; k < R.cols; ++k) {
                        x_i_new = R.at<real>(i-1,k) * cos_a - R.at<real>(i,k) * sin_a;
                        x_j_new = R.at<real>(i-1,k) * sin_a + R.at<real>(i,k) * cos_a;
                        R.at<real>(i-1,k) = x_i_new;
                        R.at<real>(i  ,k) = x_j_new;
                    }

                    // Rotate rows of Q
                    for (int k = 0; k < Q.cols; ++k) {
                        x_i_new = Q.at<real>(i-1,k) * cos_a - Q.at<real>(i,k) * sin_a;
                        x_j_new = Q.at<real>(i-1,k) * sin_a + Q.at<real>(i,k) * cos_a;
                        Q.at<real>(i-1,k) = x_i_new;
                        Q.at<real>(i  ,k) = x_j_new;
                    }
                }
            }
        }

        cv::transpose(Q, Q);
    }

    void QR_Givens_Hess(cv::Mat_<real> A, cv::Mat_<real> &Q, cv::Mat_<real> &R) {
        // TODO
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

                R.at<real>(j, i) = dotProduct;
            }

            real Q_i_norm = vectorNorm(Q_i);
            Q_i /= Q_i_norm;
            R.row(i) *= Q_i_norm;
        }

        cv::transpose(Q, Q);
    }

    void QR(cv::Mat_<real> A, cv::Mat_<real> & Q, cv::Mat_<real> & R,
            int method, int matStructure) {

        assert(A.rows >= A.cols);

        if ((matStructure & DIAGONAL) == matStructure) {
            QR_diagonal(A, Q, R);
        }

        else if ((matStructure & UPPER_TRI) == matStructure) {
            QR_upperTri(A, Q, R);
        }

        else if ((matStructure | HESSENBERG) == HESSENBERG) {
            // Regardless of choice
            QR_Givens_Hess(A, Q, R);
        }

        else {
            switch (method) {
                case QRMethod::GS: {
                    QR_GS(A, Q, R);
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
            }
        }
    }

};