#include "common.hpp"

namespace yalal {

	// LU algorithm without pivoting. LU is done in-place. 
	// Elements below main diagonal are from L matrix, on and above 
	// - from U matrix. We assume that L matrix has 1 on the main diagonal.
    bool LU(cv::Mat_<real> A, 
            int matStructure = MatStructure::ARBITRARY);

    // LU algorithm with partial pivoting. PA = LU
    bool LU(cv::Mat_<real> A, cv::Mat_<real> & P,  
    		int matStructure = MatStructure::ARBITRARY);

    // Decompose LU, stored in matrix A, to separate L and U matrices. 
    void RecoverLU(cv::Mat_<real>& A, cv::Mat_<real>& L, cv::Mat_<real>& U);
}