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

        if (matStructure & DIAGONAL == matStructure) {
            QR_diagonal(A, Q, R);
        }

        else if (matStructure & UPPER_TRI == matStructure) {
            QR_upperTri(A, Q, R);
        }

        else if (matStructure & HESSENBERG == matStructure) {
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