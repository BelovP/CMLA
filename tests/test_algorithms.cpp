#include <iostream>
#include "algorithms.hpp"

const real EPS = 1e-4;

#ifdef DOUBLE_PRECISION
const real allowedError = 1e-3;
#else
const real allowedError = 20.;
#endif

using std::string;
using namespace yalal::MatStructure;

int main() {
    real dummy_data[9] = { 1, 7, 9, 0, 5, 7, 9, 7, 1};
    real dummy_constraints[3] = { 1, 7, 9};
    real dummy_answer[3] = {-48, 70, -49};
    cv::Mat_<real> A = cv::Mat(3, 3, CV_32F, dummy_data);
    real det = yalal::Det(A);
    if (std::abs(det - (-8)) > EPS) {
        std::cout << "Algorithms test FAIL for determinant computation. Value received: " << det << ", value expected: " << 8 << std::endl;
        return 1;
    }

    A = cv::Mat(3, 3, CV_32F, dummy_data);
    cv::Mat_<real> f = cv::Mat(3, 1, CV_32F, dummy_constraints);
    cv::Mat_<real> real_x = cv::Mat(3, 1, CV_32F, dummy_answer);
    cv::Mat_<real> x = yalal::SolveSystem(A, f);

    real error = cv::norm(x - real_x, cv::NORM_L2);
    if (error > EPS) {
        std::cout << "Algorithms test fail for solving system. Solution received: " << x << ", correct solution: " << real_x << std::endl;
        return 1;
    }

    return 0;
}