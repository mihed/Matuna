/*
 * Matrix.cpp
 *
 *  Created on: May 12, 2015
 *      Author: Mikael
 */

#include "Matrix.h"
#include <math.h>
#include <stdexcept>

namespace ATML
{
namespace Math
{

template class Matrix<float> ;
template class Matrix<double> ;
template class Matrix<long double> ;

template<class T>
Matrix<T>::Matrix()
{
	unique_ptr<T[]> managedData = nullptr;
	Data = nullptr;
	rows = 0;
	columns = 0;
}
template<class T>
Matrix<T>::Matrix(int rows, int columns) :
		rows(rows), columns(columns)
{
	if (rows <= 0)
		throw invalid_argument(
				"The dimensions has to be valid");
	else if (columns <= 0)
		throw invalid_argument(
				"The dimensions has to be valid");


	managedData = unique_ptr<T[]>(new T[rows * columns]);
	Data = managedData.get();
}

template<class T>
Matrix<T>::Matrix(int rows, int columns, const T* data) :
		rows(rows), columns(columns)
{
	if (rows <= 0)
		throw invalid_argument(
				"The dimensions has to be valid");
	else if (columns <= 0)
		throw invalid_argument(
				"The dimensions has to be valid");

	managedData = unique_ptr<T[]>(new T[rows * columns]);
	Data = managedData.get();
	memcpy(Data, data, sizeof(T) * rows * columns);
}

template<class T>
Matrix<T>::Matrix(int rows, int columns, T initialValue) :
		rows(rows), columns(columns)
{
	if (rows <= 0)
		throw invalid_argument(
				"The dimensions has to be valid");
	else if (columns <= 0)
		throw invalid_argument(
				"The dimensions has to be valid");

	managedData = unique_ptr<T[]>(new T[rows * columns]);
	Data = managedData.get();
	int cachedIndex;
	for (int i = 0; i < rows; i++)
	{
		cachedIndex = i * columns;
		for (int j = 0; j < columns; j++)
			Data[cachedIndex + j] = initialValue;
	}
}

template<class T>
Matrix<T>::Matrix(const Matrix<T>& matrix)
{
	rows = matrix.rows;
	columns = matrix.columns;
	managedData = unique_ptr<T[]>(new T[rows * columns]);
	Data = managedData.get();
	memcpy(Data, matrix.Data, sizeof(T) * rows * columns);
}

template<class T>
Matrix<T>::Matrix(Matrix<T>&& other)
{
	rows = other.rows;
	columns = other.columns;
	managedData = move(other.managedData);
	Data = managedData.get();
}

template<class T>
Matrix<T>::~Matrix()
{

}

template<class T>
int Matrix<T>::ColumnCount() const
{
	return columns;
}

template<class T>
int Matrix<T>::RowCount() const
{
	return rows;
}

template<class T>
Matrix<T> Matrix<T>::Transpose() const
{
	Matrix<T> result(columns, rows);
	auto resultData = result.Data;
	int cachedIndex;
	for (int i = 0; i < rows; i++)
	{
		cachedIndex = i * columns;
		for (int j = 0; j < columns; j++)
			resultData[j * rows + i] = Data[cachedIndex + j];
	}

	return result;
}

template<class T>
T Matrix<T>::Norm2() const
{
	T result = 0;
	int cachedIndex;
	for (int i = 0; i < rows; i++)
	{
		cachedIndex = i * columns;
		for (int j = 0; j < columns; j++)
		{
			T cache = Data[cachedIndex + j];
			result = sqrt(cache * cache);
		}
	}

	return result;
}

template<class T>
T Matrix<T>::At(int row, int column) const
{
	return Data[row * columns + column];
}

template<class T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& matrix)
{
	if (this == &matrix)
		return *this;

	managedData.reset();

	rows = matrix.rows;
	columns = matrix.columns;
	managedData = unique_ptr<T[]>(new T[rows * columns]);
	Data = managedData.get();
	memcpy(Data, matrix.Data, sizeof(T) * rows * columns);

	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& other)
{
	managedData.reset();

	rows = other.rows;
	columns = other.columns;
	managedData = move(other.managedData);
	Data = managedData.get();

	return *this;
}

template<class T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& rightMatrix) const
{
	return Matrix<T>(*this) += rightMatrix;
}

template<class T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& rightMatrix) const
{
	return Matrix<T>(*this) *= rightMatrix;
}

template<class T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& rightMatrix) const
{
	return Matrix<T>(*this) -= rightMatrix;
}

template<class T>
Matrix<T> Matrix<T>::operator%(const Matrix<T>& rightMatrix) const
{
	return Matrix<T>(*this) %= rightMatrix;
}

template<class T>
bool Matrix<T>::operator==(const Matrix<T>& other)
{
	if (other.rows != rows)
		throw invalid_argument(
				"The dimensions must agree when comparing a matrix");
	else if (other.columns != columns)
		throw invalid_argument(
				"The dimensions must agree when comparing a matrix");

	auto otherData = other.Data;
	int cachedIndex1;
	int cachedIndex2;
	for (int i = 0; i < rows; i++)
	{
		cachedIndex1 = i * columns;
		for (int j = 0; j < columns; j++)
		{
			cachedIndex2 = cachedIndex1 + j;
			if (Data[cachedIndex2] != otherData[cachedIndex2])
				return false;
		}
	}

	return true;
}

template<class T>
bool Matrix<T>::operator!=(const Matrix<T>& other)
{
	return !(*this == other);
}

template<class T>
Matrix<T>& Matrix<T>::operator++()
{
	if (rows <= 0)
		throw invalid_argument(
				"The dimensions has to be valid");
	else if (columns <= 0)
		throw invalid_argument(
				"The dimensions has to be valid");

	int cachedIndex;
	for (int i = 0; i < rows; i++)
	{
		cachedIndex = i * columns;
		for (int j = 0; j < columns; j++)
			++Data[cachedIndex + j];
	}

	return *this;
}

template<class T>
Matrix<T> Matrix<T>::operator++(int)
{
	Matrix<T> result(*this);
	++(*this);
	return result;
}

template<class T>
Matrix<T>& Matrix<T>::operator--()
{
	if (rows <= 0)
		throw invalid_argument(
				"The dimensions has to be valid");
	else if (columns <= 0)
		throw invalid_argument(
				"The dimensions has to be valid");

	int cachedIndex;
	for (int i = 0; i < rows; i++)
	{
		cachedIndex = i * columns;
		for (int j = 0; j < columns; j++)
			--Data[cachedIndex + j];
	}

	return *this;
}

template<class T>
Matrix<T> Matrix<T>::operator--(int)
{
	Matrix<T> result(*this);
	--(*this);
	return result;
}

template<class T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& other)
{
	if (other.rows != rows)
		throw invalid_argument(
				"The dimensions must agree when adding a matrix");
	else if (other.columns != columns)
		throw invalid_argument(
				"The dimensions must agree when adding a matrix");

	auto otherData = other.Data;
	int cachedIndex;
	int index;
	for (int i = 0; i < rows; i++)
	{
		cachedIndex = i * columns;
		for (int j = 0; j < columns; j++)
		{
			index = cachedIndex + j;
			Data[index] += otherData[index];
		}
	}

	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator+=(const T& scalar)
{
	if (rows <= 0)
		throw invalid_argument(
				"The dimensions has to be valid");
	else if (columns <= 0)
		throw invalid_argument(
				"The dimensions has to be valid");

	int cachedIndex;
	for (int i = 0; i < rows; i++)
	{
		cachedIndex = i * columns;
		for (int j = 0; j < columns; j++)
			Data[cachedIndex + j] += scalar;
	}

	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& other)
{
	if (other.rows != columns)
		throw invalid_argument(
				"The dimensions must agree when multiplying a matrix");

	int resultRows = rows;
	int resultColumns = other.columns;
	unique_ptr<T[]> resultBuffer(new T[resultRows * resultColumns]);
	auto rawResultBuffer = resultBuffer.get();
	auto otherBuffer = other.Data;
	T sum = 0;
	int cachedIndex1;
	int cachedIndex2;
	for (int i = 0; i < resultRows; i++)
	{
		cachedIndex1 = i * resultColumns;
		cachedIndex2 = i * columns;
		for (int j = 0; j < resultColumns; j++)
		{
			sum = 0;
			for (int k = 0; k < rows; k++)
				sum += otherBuffer[k * resultColumns + j]
						* Data[cachedIndex2 + k];

			rawResultBuffer[cachedIndex1 + j] = sum;
		}
	}

	managedData = move(resultBuffer);
	Data = managedData.get();
	rows = resultRows;
	columns = resultColumns;

	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator*=(const T& scalar)
{
	if (rows <= 0)
		throw invalid_argument(
				"The dimensions has to be valid");
	else if (columns <= 0)
		throw invalid_argument(
				"The dimensions has to be valid");

	int cachedIndex;
	for (int i = 0; i < rows; i++)
	{
		cachedIndex = i * columns;
		for (int j = 0; j < columns; j++)
			Data[cachedIndex + j] *= scalar;
	}

	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& other)
{
	if (other.rows != rows)
		throw invalid_argument(
				"The dimensions must agree when subtracting a matrix");
	else if (other.columns != columns)
		throw invalid_argument(
				"The dimensions must agree when subtracting a matrix");

	auto otherData = other.Data;
	int cachedIndex;
	int index;
	for (int i = 0; i < rows; i++)
	{
		cachedIndex = i * columns;
		for (int j = 0; j < columns; j++)
		{
			index = cachedIndex + j;
			Data[index] -= otherData[index];
		}
	}

	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator-=(const T& scalar)
{
	if (rows <= 0)
		throw invalid_argument(
				"The dimensions has to be valid");
	else if (columns <= 0)
		throw invalid_argument(
				"The dimensions has to be valid");

	int cachedIndex;
	for (int i = 0; i < rows; i++)
	{
		cachedIndex = i * columns;
		for (int j = 0; j < columns; j++)
			Data[cachedIndex + j] -= scalar;
	}

	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator%=(const Matrix<T>& other)
{
	if (other.rows != rows)
		throw invalid_argument(
				"The dimensions must agree when using a Hadamard product on a matrix");
	else if (other.columns != columns)
		throw invalid_argument(
				"The dimensions must agree when using a Hadamard product on a matrix");
	auto otherData = other.Data;
	int cachedIndex;
	int index;
	for (int i = 0; i < rows; i++)
	{
		cachedIndex = i * columns;
		for (int j = 0; j < columns; j++)
		{
			index = cachedIndex + j;
			Data[index] *= otherData[index];
		}
	}

	return *this;
}

} /* namespace Math */
} /* namespace ATML */
