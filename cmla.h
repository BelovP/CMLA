#include <vector>
#include <algorithm>
#include <iostream>

using std::vector;
using std::string;

namespace cmla {

  class Matrix {
   public: 
    Matrix(const int r, const int c, const double& init = 0.0): rows(r), cols(c), data(vector<vector<double> > (r, vector<double> (init))) {}
    
    Matrix(const int r, const double& init = 0.0) : Matrix(r, 1, init) {}
    
    int rows() const { return rows; }
    
    int cols() const { return cols; }
    
    double operator[](int i) const;
    
    double& operator[](int i);
    
    // Multiplies the matrix by the given value.
    Matrix& operator *(const double value);
    
    // Divides the matrix by the given value.
    Matrix& operator /(const double value);
    
    // Sums two matrices.
    Matrix& operator +(const Matrix& other);
    
    // Subtract one matrix from another.
    Matrix& operator -(const Matrix& other);
    
    // Reshapes matrix. The fill-in in new matrix happens row-by-row.
    void reshape(const int new_r, const int new_c);
    
    // Returns vector of matrix dimensions.
    vector<int> shape() { return {rows, cols}; }

    // Multiplies matrix on other matrix or vector.
    Matrix& dot(const Matrix & other);
    
    void DebugString();

   private:
    int rows;
    int cols;
    vector<vector<double> > data; 
};
} // namespace cmla