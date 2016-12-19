#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>

#include <Eigen/Cholesky>
#include <opencv2/core/eigen.hpp>

#include <armadillo>

#include "cholesky.hpp"

typedef std::chrono::seconds Seconds;
typedef std::chrono::milliseconds Milliseconds;
typedef std::chrono::steady_clock Clock;
typedef Clock::time_point Time;

int main() {

    std::vector<int> sizes = {5, 50, 100, 250, 450, 700, 900, 1200, 1400};
    const int k = sizes.size();
    std::vector<float> timeYalal(k), timeEigen(k), timeArma(k);

    Time start_global = Clock::now(), end_global = Clock::now();

    for (int i = 0; i < k; ++i) {
        std::cout << sizes[i] << ", " <<
                std::chrono::duration_cast<Seconds>(end_global - start_global).count() << " sec" << std::endl;
        start_global = Clock::now();

        const int numRepeats = 3;
        std::vector<cv::Mat_<real>> mats(numRepeats);
        cv::Mat_<real> L(sizes[i], sizes[i]);

        for (int j = 0; j < numRepeats; ++j) {
            // large random symmetric positive definite
            mats[j].create(sizes[i], sizes[i]);
            cv::randu(mats[j], -1., 1.);
            mats[j] = mats[j].t() * mats[j];
        }

        for (int j = 0; j < numRepeats; ++j) {
            Time start = Clock::now();
            yalal::Cholesky(mats[j], L, yalal::MatStructure::ARBITRARY);
            Time end = Clock::now();
            timeYalal[i] += std::chrono::duration_cast<Milliseconds>(end - start).count();
        }
        timeYalal[i] /= numRepeats;

        Eigen::MatrixXf A, L_Eig;

        for (int j = 0; j < numRepeats; ++j) {
            cv::cv2eigen(mats[j], A);

            Time start = Clock::now();
            Eigen::LLT<Eigen::MatrixXf> chol(A);
            L_Eig = chol.matrixL();
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
            arma::Mat<real> chol_out;

            Time start = Clock::now();
            arma::chol(chol_out, A_Arma);
            Time end = Clock::now();

            timeArma[i] += std::chrono::duration_cast<Milliseconds>(end - start).count();
        }
        timeArma[i] /= numRepeats;

        end_global = Clock::now();
    }

    std::ofstream out("time_cholesky.txt");
    
    out << "n = [";
    for (int i = 0; i < k - 1; ++i) {
        out << sizes[i] << ",";
    }
    out << sizes[k-1] << "]" << std::endl;

    out << "timeYalal = [";
    for (int i = 0; i < k - 1; ++i) {
        out << timeYalal[i] << ",";
    }
    out << timeYalal[k-1] << "]" << std::endl;

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