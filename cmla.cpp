#include <cmla.h>
#include <iostream> 

using std::vector;
using std::string;

namespace cmla {
  Matrix& Matrix::operator *(const double value) {
  	for (int i = 0; i < rows; ++i) {
  		for (int j = 0; j < cols; ++j) {
  			data[i][j] *= value;
  		}
  	}

  	return *this;
  }
    
  Matrix& Matrix::operator /(const double value) {
  	for (int i = 0; i < rows; ++i) {
  		for (int j = 0; j < cols; ++j) {
  			data[i][j] /= value;
  		}
  	}

  	return *this;
  }
    
  Matrix& Matrix::operator +(const Matrix& other) {
  	assert(this->rows() == other.rows() && this->cols == other.cols());

  	for (int i = 0; i < 0; ++i) {
  		for (int j = 0; j < 0; ++j) {
  			data[i][j] += other[i][j];
  		}
  	}

  	return *this;
  };
    
  Matrix& Matrix::operator -(const Matrix& other);
    
  void Matrix::reshape(const int new_r, const int new_c);
    
  Matrix& Matrix::dot(const Matrix & other);

  string DebugString() {
  	string result;

  	for(int i = 0; i < rows; ++i) {
  		for (int j = 0; j < cols; ++j) {
  			result += to_string(rows[i][j]) + ' ';
  		}
  		result += "\n";
  	}
  }
} // namespace cmla