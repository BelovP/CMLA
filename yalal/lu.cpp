#include "lu.hpp"
#include "norms.hpp"
#include <vector>

const real close_to_zero_eps = 1e-8;

using std::vector;

namespace yalal {
 
    // Returns false if there is no good pivoting (A_{k, k} close to 0).
	bool Pivot(cv::Mat_<real> & A, cv::Mat_<real> & L, cv::Mat_<real> & U, cv::Mat_<real> & P, vector<int> & row_permutations, int k) {
        int n = A.rows;
        int row_to_swap = k;
        int i;

        for (i = k + 1; i < n; ++i) {
            if (std::abs(L(i, k)) > std::abs(L(row_to_swap, k))) {
                row_to_swap = i;
            }
        }

        // Return false if we divide on a very small number.
        if (std::abs(L(row_to_swap, k)) < close_to_zero_eps) {
            return false;
        }
        
        P(k, row_permutations[row_to_swap]) = 1;
        
        if (row_to_swap == k) {
            return true;
        }

        std::swap(row_permutations[row_to_swap], row_permutations[k]);

        for (int i = 0; i < n; ++i) {
            if (i < k) {
                std::swap(L(k, i), L(row_to_swap, i));
            } else if(i < row_to_swap) {
                std::swap(U(k, i), L(row_to_swap, i));
            } else {
                std::swap(U(k, i), U(row_to_swap, i));   
            }
        }
        
        return true;
	}

// Returns false if LU is not possible. PA = LU
    bool _LU_compute(cv::Mat_<real> & A, cv::Mat_<real> & L, cv::Mat_<real> & U, cv::Mat_<real> & P, bool pivoting = false) {
        int n = A.rows;
        int col, i, j;
        real mul1, mul2;
        // Stores the indices of rows in the initial matrix A before permutations. 
        vector<int> row_permutations;

        if (pivoting) {
            row_permutations.resize(n);
            for (i = 0; i < row_permutations.size(); ++i) {
                row_permutations[i] = i;
            }
        }

        A.copyTo(L);
    	A.copyTo(U);


    	for (i = 0; i < n; ++i) {
    		for (j = 0; j < n; ++j) {
    			if (j < i) {
    				U(i, j) = 0;
    			}
    			else {
    				L(i, j) = 0;
    				if (i == j) {
    					L(i, j) = 1;
    				}	
    			}
    		}
    	}

        if (pivoting) {
        	P.create(n, n);
        }

        for (col = 0; col < n; ++col) {
            if (pivoting && !Pivot(A, L, U, P, row_permutations, col)) {
                return false;
            }
            
            for (i = col + 1; i < n; ++i) {
            	if (col < i) {
            		L(i, col) /= U(col, col);
            	} else {
            		U(i, col) /= U(col, col);
            	}
            }
            
            for (i = col + 1; i < n; ++i) {
                for (j = col + 1; j < n; ++j) {
                    if (col < i) {
                        mul1 = L(i, col);
                    } else {
                        mul1 = U(i, col);
                    }
                    if (j < col) {
                        mul2 = L(col, j);
                    } else {
                        mul2 = U(col, j);
                    }
                    if (j < i) {
                        L(i, j) = L(i, j) - mul1 * mul2;
                    }
                    else {
                        U(i, j) = U(i, j) - mul1 * mul2;
                    }
                }
            }
        }
        
        return true;
    }

	bool LU_internal(cv::Mat_<real> & A, cv::Mat_<real> & L, cv::Mat_<real> & U, 
    		   cv::Mat_<real> & P, int matStructure, bool pivoting) {
		// Non-empty square matrices
		assert(A.rows && A.rows == A.cols); 
		
		auto n = A.rows;

		if (matStructure == MatStructure::DIAGONAL) {
			if (pivoting) P = cv::Mat_<real>::eye(n, n);
			L = cv::Mat_<real>::eye(n, n);
			U = A;

			return true;
		} else if (matStructure == MatStructure::LOWER_TRI) {
			if (pivoting) P = cv::Mat_<real>::eye(n, n);
			L = A;
			U = cv::Mat_<real>::eye(n, n);

			return true;
		} else if (matStructure == MatStructure::UPPER_TRI) {
			if (pivoting) P = cv::Mat_<real>::eye(n, n);
			L = cv::Mat_<real>::eye(n, n);
			U = A;

			return true;
		} else {
			return _LU_compute(A, L, U, P, pivoting);
		}
	}

	// LU algorithm without pivoting. 
    bool LU(cv::Mat_<real> A, cv::Mat_<real> & L, cv::Mat_<real> & U,
               int matStructure) {
    	cv::Mat_<real> P;
    	return LU_internal(A, L, U, P, matStructure, false);
    }

    // LU algorithm with partial pivoting. PA = LU
    bool LU(cv::Mat_<real> A, cv::Mat_<real> & L, cv::Mat_<real> & U, 
    		   cv::Mat_<real> & P, int matStructure) {
    	return LU_internal(A, L, U, P, matStructure, true);
    }
}