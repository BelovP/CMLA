#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>

#include <Eigen/QR>
#include <opencv2/core/eigen.hpp>

#include <armadillo>

#include "qr.hpp"

typedef std::chrono::seconds Seconds;
typedef std::chrono::milliseconds Milliseconds;
typedef std::chrono::steady_clock Clock;
typedef Clock::time_point Time;

int main() {

    std::vector<int> sizes = {5, 50, 100, 250, 450, 700, 900, 1200, 1400, 1600};
    const int k = sizes.size();
    std::vector<float> timeYalalMGS(k), timeYalalHouseholder(k),
            timeYalalGivens(k), timeEigen(k), timeArma(k);

    Time start_global = Clock::now(), end_global = Clock::now();

    for (int i = 0; i < k; ++i) {
        std::cout << sizes[i] << ", " <<
                std::chrono::duration_cast<Seconds>(end_global - start_global).count() << " sec" << std::endl;
        start_global = Clock::now();

        const int numRepeats = 3;
        std::vector<cv::Mat_<real>> mats(numRepeats);
        cv::Mat_<real> Q(sizes[i], sizes[i]), R(sizes[i], sizes[i]);
//
//        for (int j = 0; j < numRepeats; ++j) {
//            mats[j].create(sizes[i], sizes[i]);
//            cv::randu(mats[j], -1., 1.);
//        }

        for (int j = 0; j < numRepeats; ++j) {
            // upper triangular
            mats[j] = cv::Mat_<real>::zeros(sizes[i], sizes[i]);
            for (int i = 0; i < mats[j].rows; ++i) {
                for (int p = i; p < mats[j].cols; ++p) {
                    mats[j](i, p) = (cv::randu<real>() - 0.5) * 2.;
                }
            }

            for (int i = 0; i < mats[j].rows - 5; ++i) {
                mats[j](i+1, i) = (cv::randu<real>() - 0.5) * 2.;
            }
        }

//        for (int j = 0; j < numRepeats; ++j) {
//            Time start = Clock::now();
//            yalal::QR(mats[j], Q, R,
//                      yalal::MatStructure::ARBITRARY, yalal::QRMethod::GS_MODIFIED);
//            Time end = Clock::now();
//            timeYalalMGS[i] += std::chrono::duration_cast<Milliseconds>(end - start).count();
//        }
//        timeYalalMGS[i] /= numRepeats;
//
//        for (int j = 0; j < numRepeats; ++j) {
//            Time start = Clock::now();
//            yalal::QR(mats[j], Q, R,
//                      yalal::MatStructure::ARBITRARY, yalal::QRMethod::HOUSEHOLDER);
//            Time end = Clock::now();
//            timeYalalHouseholder[i] += std::chrono::duration_cast<Milliseconds>(end - start).count();
//        }
//        timeYalalHouseholder[i] /= numRepeats;

        for (int j = 0; j < numRepeats; ++j) {
            Time start = Clock::now();
            yalal::QR(mats[j], Q, R,
                      yalal::MatStructure::ARBITRARY, yalal::QRMethod::GIVENS);
            Time end = Clock::now();
            timeYalalGivens[i] += std::chrono::duration_cast<Milliseconds>(end - start).count();
        }
        timeYalalGivens[i] /= numRepeats;

        Eigen::MatrixXf A, Q_Eig, R_Eig;

        for (int j = 0; j < numRepeats; ++j) {
            cv::cv2eigen(mats[j], A);

            Time start = Clock::now();
            Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);
            Q_Eig = qr.householderQ();
            R_Eig = qr.matrixQR();
            Time end = Clock::now();

            timeEigen[i] += std::chrono::duration_cast<Milliseconds>(end - start).count();
        }
        timeEigen[i] /= numRepeats;

        arma::Mat<real> A_Arma, Q_Arma, R_Arma;

        for (int j = 0; j < numRepeats; ++j) {
            cv::Mat_<real> transposed;
            cv::transpose(mats[j], transposed);
            A_Arma = arma::Mat<real>(
                    reinterpret_cast<real*>(transposed.data), transposed.rows, transposed.cols);

            Time start = Clock::now();
            arma::qr(Q_Arma, R_Arma, A_Arma);
            Time end = Clock::now();

            timeArma[i] += std::chrono::duration_cast<Milliseconds>(end - start).count();
        }
        timeArma[i] /= numRepeats;

        end_global = Clock::now();
    }

    std::ofstream out("time_qr_sparse.txt");
    
    out << "n = [";
    for (int i = 0; i < k - 1; ++i) {
        out << sizes[i] << ",";
    }
    out << sizes[k-1] << "]" << std::endl;

    out << "timeYalalMGS = [";
    for (int i = 0; i < k - 1; ++i) {
        out << timeYalalMGS[i] << ",";
    }
    out << timeYalalMGS[k-1] << "]" << std::endl;

    out << "timeYalalHouseholder = [";
    for (int i = 0; i < k - 1; ++i) {
        out << timeYalalHouseholder[i] << ",";
    }
    out << timeYalalHouseholder[k-1] << "]" << std::endl;

    out << "timeYalalGivens = [";
    for (int i = 0; i < k - 1; ++i) {
        out << timeYalalGivens[i] << ",";
    }
    out << timeYalalGivens[k-1] << "]" << std::endl;

    out << "timeEigen = [";
    for (int i = 0; i < k - 1; ++i) {
        out << timeEigen[i] << ",";
    }
    out << timeEigen[k-1] << "]" << std::endl;

    out << "timeArma = [";
    for (int i = 0; i < k - 1; ++i) {
        out << timeArma[i] << ",";
    }
    out << timeArma[k-1] << "]" << std::endl;
    
    return 0;
}