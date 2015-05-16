/*
 * Matrix.cpp
 *
 *  Created on: May 12, 2015
 *      Author: Mikael
 */

#include "Matrix.h"
#include <stdio.h>
#include <math.h>
#include <sstream>
#include <stdexcept>
#include <random>

namespace ATML
{
	namespace Math
	{

		template class Matrix < float > ;
		template class Matrix < double > ;
		template class Matrix < long double > ;

		template<class T>
		Matrix<T>::Matrix()
		{
			unique_ptr<T[]> managedData = nullptr;
			Data = nullptr;
			elementCount = 0;
			rows = 0;
			columns = 0;
		}
		template<class T>
		Matrix<T>::Matrix(int rows, int columns) :
			rows(rows), columns(columns), elementCount(rows * columns)
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
			rows(rows), columns(columns), elementCount(rows * columns)
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
			rows(rows), columns(columns), elementCount(rows * columns)
		{
			if (rows <= 0)
				throw invalid_argument(
				"The dimensions has to be valid");
			else if (columns <= 0)
				throw invalid_argument(
				"The dimensions has to be valid");

			managedData = unique_ptr<T[]>(new T[rows * columns]);
			Data = managedData.get();
			for (int i = 0; i < elementCount; i++)
				Data[i] = initialValue;
		}

		template<class T>
		Matrix<T>::Matrix(const Matrix<T>& matrix)
		{
			rows = matrix.rows;
			columns = matrix.columns;
			elementCount = rows * columns;
			managedData = unique_ptr<T[]>(new T[elementCount]);
			Data = managedData.get();
			memcpy(Data, matrix.Data, sizeof(T) * elementCount);
		}

		template<class T>
		Matrix<T>::Matrix(Matrix<T>&& other)
		{
			rows = other.rows;
			columns = other.columns;
			elementCount = rows * columns;
			managedData = move(other.managedData);
			Data = managedData.get();
		}

		template<class T>
		Matrix<T>::~Matrix()
		{

		}

		template<class T>
		Matrix<T> Matrix<T>::Zeros(int rows, int columns)
		{
			return Matrix<T>(rows, columns, T(0));
		}

		template<class T>
		Matrix<T> Matrix<T>::Ones(int rows, int columns)
		{
			return Matrix<T>(rows, columns, 1);
		}

		template<class T>
		Matrix<T> Matrix<T>::Identity(int dimension)
		{
			Matrix<T> result = Zeros(dimension, dimension);
			auto buffer = result.Data;
			for (int i = 0; i < dimension; i++)
				buffer[i * dimension + i] = 1;

			return result;
		}

		template<class T>
		Matrix<T> Matrix<T>::RandomUniform(int rows, int columns, T min, T max)
		{
			random_device device;
			mt19937 mersienne(device());
			uniform_real_distribution<T> distribution(min, max);

			Matrix<T> result(rows, columns);
			auto buffer = result.Data;
			int count = rows * columns;
			for (int i = 0; i < count; i++)
				buffer[i] = distribution(mersienne);

			return result;
		}

		template<class T>
		Matrix<T> Matrix<T>::RandomNormal(int rows, int columns, T mean, T deviation)
		{
			random_device device;
			mt19937 mersienne(device());
			normal_distribution<T> distribution(mean, deviation);

			Matrix<T> result(rows, columns);
			auto buffer = result.Data;
			int count = rows * columns;
			for (int i = 0; i < count; i++)
				buffer[i] = distribution(mersienne);

			return result;
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
		int Matrix<T>::ElementCount() const
		{
			return elementCount;
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
			for (int i = 0; i < elementCount; i++)
			{
				T cache = Data[i];
				result += cache * cache;
			}

			return sqrt(result);
		}

		template<class T>
		T Matrix<T>::Norm2Square() const
		{
			T result = 0;
			for (int i = 0; i < elementCount; i++)
			{
				T cache = Data[i];
				result += cache * cache;
			}

			return result;
		}

		template<class T>
		void Matrix<T>::Transform(function<T(T)> function)
		{
			for (int i = 0; i < elementCount; i++)
				Data[i] = function(Data[i]);
		}

		template<class T>
		T Matrix<T>::At(int row, int column) const
		{
			return Data[row * columns + column];
		}

		template<class T>
		T& Matrix<T>::At(int row, int column)
		{
			return Data[row * columns + column];
		}

		template<class T>
		string Matrix<T>::GetString()
		{
			stringstream stream;
			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < columns; j++)
				{
					stream << Data[i * columns + j] << " ";
				}
				stream << "\n";
			}

			return stream.str();
		}

		template<class T>
		Matrix<T>& Matrix<T>::operator=(const Matrix<T>& matrix)
		{
			if (this == &matrix)
				return *this;

			managedData.reset();

			rows = matrix.rows;
			columns = matrix.columns;
			elementCount = rows * columns;
			managedData = unique_ptr<T[]>(new T[elementCount]);
			Data = managedData.get();
			memcpy(Data, matrix.Data, sizeof(T) * elementCount);

			return *this;
		}

		template<class T>
		Matrix<T>& Matrix<T>::operator=(Matrix<T>&& other)
		{
			rows = other.rows;
			columns = other.columns;
			elementCount = rows * columns;
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
			for (int i = 0; i < elementCount; i++)
				if (Data[i] != otherData[i])
					return false;

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

			for (int i = 0; i < elementCount; i++)
				++Data[i];

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

			for (int i = 0; i < elementCount; i++)
				--Data[i];

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
			for (int i = 0; i < elementCount; i++)
				Data[i] += otherData[i];

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


			for (int i = 0; i < elementCount; i++)
				Data[i] += scalar;

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
					for (int k = 0; k < columns; k++)
						sum += otherBuffer[k * resultColumns + j]
						* Data[cachedIndex2 + k];

					rawResultBuffer[cachedIndex1 + j] = sum;
				}
			}

			managedData = move(resultBuffer);
			Data = managedData.get();
			rows = resultRows;
			columns = resultColumns;
			elementCount = rows * columns;

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

			for (int i = 0; i < elementCount; i++)
				Data[i] *= scalar;

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
			for (int i = 0; i < elementCount; i++)
				Data[i] -= otherData[i];

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

			for (int i = 0; i < elementCount; i++)
				Data[i] -= scalar;

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
			for (int i = 0; i < elementCount; i++)
				Data[i] *= otherData[i];

			return *this;
		}

	} /* namespace Math */
} /* namespace ATML */
