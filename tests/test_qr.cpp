#include <iostream>
#include "qr.hpp"

const real EPS = 1e-4;

using yalal::QR;
using yalal::QRMethod;

int main() {

    cv::Mat_<real> A(1023, 768);
    cv::randu(A, -1., 1.);

    cv::Mat_<real> Q, R;
    // TODO
    // QR(A, Q, R); ...

    return 0;
}