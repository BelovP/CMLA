#include "common.hpp"

namespace yalal {

	// LU algorithm without pivoting. 
    bool LU(cv::Mat_<real> A, cv::Mat_<real> & L, cv::Mat_<real> & U,
            int matStructure = ARBITRARY);

    // LU algorithm with partial pivoting. PA = LU
    bool LU(cv::Mat_<real> A, cv::Mat_<real> & L, cv::Mat_<real> & U,cv::Mat_<real> & P,  
    		int matStructure = ARBITRARY);
}