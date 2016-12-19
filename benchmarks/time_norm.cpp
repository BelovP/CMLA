#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>

#include <Eigen/SVD>
#include <opencv2/core/eigen.hpp>

#include <armadillo>

#include "qr.hpp"
#include "norms.hpp"

typedef std::chrono::seconds Seconds;
typedef std::chrono::milliseconds Milliseconds;
typedef std::chrono::steady_clock Clock;
typedef Clock::time_point Time;

int main() {

    std::vector<int> sizes = {5, 50, 100, 250, 380, 490, 600};//, 750, 930};
    const int k = sizes.size();
    std::vector<float> timeYalal(k), timeOpenCV(k), timeEigen(k), timeArma(k);

    Time start_global = Clock::now(), end_global = Clock::now();

    for (int i = 0; i < k; ++i) {
        std::cout << sizes[i] << ", " <<
                std::chrono::duration_cast<Seconds>(end_global - start_global).count() << " sec" << std::endl;
        start_global = Clock::now();

        const int numRepeats = 3;
        std::vector<cv::Mat_<real>> mats(numRepeats);
        cv::Mat_<real> Q(sizes[i], sizes[i]), R(sizes[i], sizes[i]);

        for (int j = 0; j < numRepeats; ++j) {
            mats[j].create(sizes[i], sizes[i]);
            cv::randu(mats[j], -1., 1.);
        }

        std::vector<real> answers(numRepeats);

        for (int j = 0; j < numRepeats; ++j) {
            Time start = Clock::now();
            answers[j] = yalal::matrixNorm(mats[j]);
            Time end = Clock::now();
            timeYalal[i] += std::chrono::duration_cast<Milliseconds>(end - start).count();
        }
        timeYalal[i] /= numRepeats;

        for (int j = 0; j < numRepeats; ++j) {
            
            Time start = Clock::now();
            cv::SVD svd(mats[j], cv::SVD::NO_UV);
            real norm = svd.w.at<real>(0);
            if (std::abs(norm - answers[j]) > 1e-3) {
                std::cout << "OpenCV is wrong: " << norm << " vs " << answers[j] << std::endl;
            }
            Time end = Clock::now();

            timeOpenCV[i] += std::chrono::duration_cast<Milliseconds>(end - start).count();
        }
        timeOpenCV[i] /= numRepeats;

        Eigen::MatrixXf A, Q_Eig, R_Eig;

        for (int j = 0; j < numRepeats; ++j) {
            cv::cv2eigen(mats[j], A);

            Time start = Clock::now();
            Eigen::BDCSVD<Eigen::MatrixXf> svd(A);
            real norm = svd.singularValues()(0);
            if (std::abs(norm - answers[j]) > 1e-3) {
                std::cout << "Eigen is wrong: " << norm << " vs " << answers[j] << std::endl;
            }
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
            real norm = arma::norm(A_Arma, 2);
            if (std::abs(norm - answers[j]) > 1e-3) {
                std::cout << "Armadillo is wrong" << std::endl;
            }
            Time end = Clock::now();

            timeArma[i] += std::chrono::duration_cast<Milliseconds>(end - start).count();
        }
        timeArma[i] /= numRepeats;

        end_global = Clock::now();
    }

    std::ofstream out("time_norm.txt");
    
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

    out << "timeOpenCV = [";
    for (int i = 0; i < k - 1; ++i) {
        out << timeOpenCV[i] << ",";
    }
    out << timeOpenCV[k-1] << "]" << std::endl;

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
