#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>

#include "Eigen/LU"
#include <opencv2/core/eigen.hpp>

#include <armadillo>

#include "lu.hpp"
#include "cholesky.hpp"

typedef std::chrono::seconds Seconds;
typedef std::chrono::milliseconds Milliseconds;
typedef std::chrono::steady_clock Clock;
typedef Clock::time_point Time;

int main() {

    std::vector<int> sizes = {5, 50, 100, 250, 450, 700, 900};//, 1200, 1400, 1600, 1800};
    const int k = sizes.size();
    std::vector<float> timeYalalLu(k), timeYalalLuPivoting(k), 
            timeYalalCholeskyOuter(k), timeYalalCholeskyBanachiewicz(k), timeYalalCholeskyCrout(k), timeEigen(k), timeArma(k);

    Time start_global = Clock::now(), end_global = Clock::now();

    for (int i = 0; i < k; ++i) {
        std::cout << sizes[i] << ", " <<
                std::chrono::duration_cast<Seconds>(end_global - start_global).count() << " sec" << std::endl;
        start_global = Clock::now();

        const int numRepeats = 5;
        std::vector<cv::Mat_<real>> mats(numRepeats);
        cv::Mat_<real> A(sizes[i], sizes[i]), L(sizes[i], sizes[i]), U(sizes[i], sizes[i]), P(sizes[i], sizes[i]);

        for (int j = 0; j < numRepeats; ++j) {
            mats[j].create(sizes[i], sizes[i]);
            cv::randu(mats[j], 1., 3.);
        }

        for (int j = 0; j < numRepeats; ++j) {
            mats[j].copyTo(A);
            Time start = Clock::now();
            yalal::LU(A,  
                      yalal::MatStructure::ARBITRARY);
            Time end = Clock::now();
            timeYalalLu[i] += std::chrono::duration_cast<Milliseconds>(end - start).count();
        }
        timeYalalLu[i] /= numRepeats;

        for (int j = 0; j < numRepeats; ++j) {
            mats[j].copyTo(A);
            Time start = Clock::now();
            yalal::LU(A, P, 
                      yalal::MatStructure::ARBITRARY);
            Time end = Clock::now();
            timeYalalLuPivoting[i] += std::chrono::duration_cast<Milliseconds>(end - start).count();
        }
        timeYalalLuPivoting[i] /= numRepeats;

        Eigen::MatrixXf A_Eig, L_Eig, U_Eig, P_Eig;

        for (int j = 0; j < numRepeats; ++j) {
            cv::cv2eigen(mats[j], A_Eig);

            Time start = Clock::now();
            Eigen::PartialPivLU<Eigen::MatrixXf> lu(A_Eig);
            //Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);
            L_Eig = lu.matrixLU();
            U_Eig = lu.matrixLU();
            P_Eig = lu.permutationP();
            Time end = Clock::now();

            timeEigen[i] += std::chrono::duration_cast<Milliseconds>(end - start).count();
        }

        timeEigen[i] /= numRepeats;

        /*arma::Mat<real> A_Arma, Q_Arma, R_Arma;

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
        timeArma[i] /= numRepeats; */

        end_global = Clock::now();
    }

    std::ofstream out("time_lu.txt");
    
    out << "n = [";
    for (int i = 0; i < k - 1; ++i) {
        out << sizes[i] << ",";
    }
    out << sizes[k-1] << "]" << std::endl;

    out << "timeYalalLu = [";
    for (int i = 0; i < k - 1; ++i) {
        out << timeYalalLu[i] << ",";
    }
    out << timeYalalLu[k-1] << "]" << std::endl;

    out << "timeYalalLuPivoting = [";
    for (int i = 0; i < k - 1; ++i) {
        out << timeYalalLuPivoting[i] << ",";
    }
    out << timeYalalLuPivoting[k-1] << "]" << std::endl;

    /*out << "timeYalalCholeskyOuter = [";
    for (int i = 0; i < k - 1; ++i) {
        out << timeYalalCholeskyOuter[i] << ",";
    }
    out << timeYalalCholeskyOuter[k-1] << "]" << std::endl;

    out << "timeYalalCholeskyBanachiewicz = [";
    for (int i = 0; i < k - 1; ++i) {
        out << timeYalalCholeskyBanachiewicz[i] << ",";
    }
    out << timeYalalCholeskyBanachiewicz[k-1] << "]" << std::endl;

    out << "timeYalalCholeskyCrout = [";
    for (int i = 0; i < k - 1; ++i) {
        out << timeYalalCholeskyCrout[i] << ",";
    }
    out << timeYalalCholeskyCrout[k-1] << "]" << std::endl; */

    out << "timeEigen = [";
    for (int i = 0; i < k - 1; ++i) {
        out << timeEigen[i] << ",";
    }
    out << timeEigen[k-1] << "]" << std::endl;

    /*out << "timeArma = [";
    for (int i = 0; i < k - 1; ++i) {
        out << timeArma[i] << ",";
    }
    out << timeArma[k-1] << "]" << std::endl; */
    
    return 0;
}