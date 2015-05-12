/*
 * Matrix.h
 *
 *  Created on: May 12, 2015
 *      Author: Mikael
 */

#ifndef ATML_MATH_MATRIX_H_
#define ATML_MATH_MATRIX_H_

#include <memory>

using namespace std;

namespace ATML
{
namespace Math
{

template<class T>
class Matrix
{
public:
	T* Data;
private:
	unique_ptr<T[]> managedData;
	int rows;
	int columns;

public:
	Matrix();
	Matrix(int rows, int columns);
	Matrix(int rows, int columns, T initialValue);
	Matrix(int rows, int columns, const T* data);
	Matrix(const Matrix<T>& other);
	Matrix(Matrix<T>&& other);
	~Matrix();

	int ColumnCount() const;
	int RowCount() const;
	Matrix<T> Transpose() const;
	T Norm2() const;
	T At(int row, int column) const;

	Matrix<T>& operator=(const Matrix<T>& other);
	Matrix<T>& operator=(Matrix<T>&& other);

	bool operator==(const Matrix<T>& other);
	bool operator!=(const Matrix<T>& other);

	Matrix<T> operator+(const Matrix<T>& other) const;
	Matrix<T> operator*(const Matrix<T>& other) const;
	Matrix<T> operator-(const Matrix<T>& other) const;
	Matrix<T> operator%(const Matrix<T>& other) const;

	Matrix<T>& operator++();
	Matrix<T> operator++(int);

	Matrix<T>& operator--();
	Matrix<T> operator--(int);

	Matrix<T>& operator+=(const Matrix<T>& other);
	Matrix<T>& operator+=(const T& scalar);

	Matrix<T>& operator*=(const Matrix<T>& other);
	Matrix<T>& operator*=(const T& scalar);

	Matrix<T>& operator-=(const Matrix<T>& other);
	Matrix<T>& operator-=(const T& scalar);

	Matrix<T>& operator%=(const Matrix<T>& other);
};

template<typename T>
Matrix<T> operator*(const T& scalar, Matrix<T> right)
{
    return right *= scalar;
}

template<typename T>
Matrix<T> operator-(const T& scalar, Matrix<T> right)
{
    return right -= scalar;
}

template<typename T>
Matrix<T> operator+(const T& scalar, Matrix<T> right)
{
    return right += scalar;
}

template<typename T>
Matrix<T> operator*(Matrix<T> left, const T& scalar)
{
    return left *= scalar;
}

template<typename T>
Matrix<T> operator-(Matrix<T> left, const T& scalar)
{
    return left -= scalar;
}

template<typename T>
Matrix<T> operator+(Matrix<T> left, const T& scalar)
{
    return left += scalar;
}

} /* namespace Math */
} /* namespace ATML */

#endif /* ATML_MATH_MATRIX_H_ */
