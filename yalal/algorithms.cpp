#include "lu.hpp"
#include "norms.hpp"
#include <vector>

namespace yalal {
	// Solves systems of the form Ax = f, where A - square matrix. LUx = f, => Ly = f, Ux = y
    cv::Mat_<real> SolveSystem(cv::Mat_<real> A, cv::Mat_<real> & f, int matStructure) {
    	assert(A.rows == A.cols);
    	assert(A.rows == f.rows);
    	int n = A.rows; 

    	cv::Mat_<real> L, U; 
    	LU(A, matStructure);
    	RecoverLU(A, L, U);
    	
    	// Forward step Ly = f.
    	cv::Mat_<real> y;  
    	y.create(f.rows, 1);
    	for (int i = 0; i < n; ++i) {
    		real acc = 0;
    		for (int j = 0; j < i; ++j)
    			acc += y(j, 0) * L(i, j);
    		y(i, 0) = f(i, 0) - acc;
    	}
    	
    	// Backward step Ux = y.
    	cv::Mat_<real> x;
    	x.create(y.rows, 1);
    	for (int i = n - 1; i >= 0; --i) {
    		real acc = 0;
    		for (int j = n - 1; j > i; --j) {
    			acc += x(j, 0) * U(i, j);
    		}
    		x(i, 0) = (y(i, 0) - acc) / U(i, i);
    	}

    	return x;
    }

    real Det(cv::Mat_<real> A, int matStructure) {
    	int n = A.rows;
    	assert(A.rows == A.cols);

    	cv::Mat_<real> A_dec;
		A.copyTo(A_dec);
    	LU(A_dec, matStructure);
    	
    	real det = 1;
    	for (int i = 0; i < n; ++i) {
    		det *= A_dec(i, i);
    	}

    	return det;
    }
}