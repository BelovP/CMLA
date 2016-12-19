#include "common.hpp"

namespace yalal {

    cv::Mat_<real> SolveSystem(cv::Mat_<real> & L, cv::Mat_<real> & U,
							   cv::Mat_<real> & f, int matStructure = MatStructure::ARBITRARY);

    real Det(cv::Mat_<real> A, int matStructure = MatStructure::ARBITRARY);
}