/*
 * Matrix.h
 *
 *  Created on: May 12, 2015
 *      Author: Mikael
 */

#ifndef ATML_MATH_MATRIX_H_
#define ATML_MATH_MATRIX_H_

#include <memory>
#include <string>
#include <functional>

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
	int elementCount;

public:
	Matrix();
	Matrix(int rows, int columns);
	Matrix(int rows, int columns, T initialValue);
	Matrix(int rows, int columns, const T* data);
	Matrix(const Matrix<T>& other);
	Matrix(Matrix<T>&& other);
	~Matrix();

	static Matrix<T> Zeros(int rows, int columns);
	static Matrix<T> Ones(int rows, int columns);
	static Matrix<T> Identity(int dimension);
	static Matrix<T> RandomUniform(int rows, int columns, T min = 0, T max = 1);
	static Matrix<T> RandomNormal(int rows, int columns, T mean = 0, T deviation = 1);

	int ColumnCount() const;
	int RowCount() const;
	int ElementCount() const;
	Matrix<T> Transpose() const;
	Matrix<T> GetSubMatrix(int startRow, int startColumn, int rowLength, int columnlength) const;
	Matrix<T> Convolve(const Matrix<T>& kernel) const;
	Matrix<T> AddZeroBorder(int size) const;
	Matrix<T> AddZeroBorder(int leftSize, int rightSize, int upSize, int downSize) const;
	Matrix<T> AddBorder(int leftSize, int rightSize, int upSize, int downSize, T value) const;
	Matrix<T> AddBorder(int size, T value) const;
	Matrix<T> Rotate90() const;
	Matrix<T> Rotate180() const;
	Matrix<T> Rotate270() const;
	void Transform(function<T(T)> function);
	void SetSubMatrix(int startRow, int startColumn, const Matrix<T>& subMatrix);
	T Norm2() const;
	T Norm2Square() const;
	T Sum() const;
	T At(int row, int column) const;
	T& At(int row, int column);


	string GetString();

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

typedef Matrix<float> Matrixf;
typedef Matrix<double> Matrixd;
typedef Matrix<long double> Matrixld;

} /* namespace Math */
} /* namespace ATML */

#endif /* ATML_MATH_MATRIX_H_ */
