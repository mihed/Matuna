/*
* Matrix.cpp
*
*  Created on: May 12, 2015
*      Author: Mikael
*/

#include "Matrix.h"
#include <stdio.h>
#include <math.h>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <random>

namespace Matuna
{
	namespace Math
	{

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
				throw invalid_argument("The dimensions has to be valid");
			else if (columns <= 0)
				throw invalid_argument("The dimensions has to be valid");

			managedData = unique_ptr<T[]>(new T[rows * columns]);
			Data = managedData.get();
		}

		template<class T>
		Matrix<T>::Matrix(int rows, int columns, const T* data) :
			rows(rows), columns(columns), elementCount(rows * columns)
		{
			if (rows <= 0)
				throw invalid_argument("The dimensions has to be valid");
			else if (columns <= 0)
				throw invalid_argument("The dimensions has to be valid");

			managedData = unique_ptr<T[]>(new T[rows * columns]);
			Data = managedData.get();
			memcpy(Data, data, sizeof(T) * rows * columns);
		}

		template<class T>
		Matrix<T>::Matrix(int rows, int columns, T initialValue) :
			rows(rows), columns(columns), elementCount(rows * columns)
		{
			if (rows <= 0)
				throw invalid_argument("The dimensions has to be valid");
			else if (columns <= 0)
				throw invalid_argument("The dimensions has to be valid");

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
		Matrix<T>::Matrix(Matrix<T> && other)
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
		Matrix<T> Matrix<T>::Convolve(const Matrix<T>& kernel) const
		{
			int kernelWidth = kernel.ColumnCount();
			int kernelHeight = kernel.RowCount();
			int resultWidth = columns - kernelWidth + 1;
			int resultHeight = rows - kernelHeight + 1;
			if (resultWidth <= 0)
				throw invalid_argument("The kernel width is too big");

			if (resultHeight <= 0)
				throw invalid_argument("The kernel height is too big");

			Matrix<T> result(resultHeight, resultWidth);
			auto resultBuffer = result.Data;
			auto kernelBuffer = kernel.Data;

			int temp1, temp2, temp3, temp4, temp5;
			T sum;
			for (int y = 0; y < resultHeight; y++)
			{
				temp1 = columns * y;
				temp4 = resultWidth * y;
				for (int x = 0; x < resultWidth; x++)
				{
					temp2 = x + temp1;
					sum = 0;
					for (int v = 0; v < kernelHeight; v++)
					{
						temp3 = temp2 + columns * v;
						temp5 = v * kernelWidth;
						for (int u = 0; u < kernelWidth; u++)
							sum += Data[temp3 + u] * kernelBuffer[temp5 + u];
					}

					resultBuffer[x + temp4] = sum;
				}
			}

			return result;
		}

		template<class T>
		Matrix<T> Matrix<T>::GetSubMatrix(int startRow, int startColumn, int rowLength,
			int columnLength) const
		{
			if (startRow + rowLength > rows)
				throw invalid_argument("Index out of range");

			if (startColumn + columnLength > columns)
				throw invalid_argument("Index out of range");

			Matrix<T> result(rowLength, columnLength);
			T* temp = Data + startColumn + columns * startRow;
			T* rowPointer;
			T* resultPointer;
			for (int i = 0; i < rowLength; i++)
			{
				rowPointer = temp + columns * i;
				resultPointer = result.Data + i * columnLength;
				memcpy(resultPointer, rowPointer, columnLength * sizeof(T));
			}

			return result;
		}
		template<class T>
		void Matrix<T>::SetSubMatrix(int startRow, int startColumn,
			const Matrix<T>& subMatrix)
		{

			auto rowLength = subMatrix.RowCount();
			auto columnLength = subMatrix.ColumnCount();

			if (startRow + rowLength > rows)
				throw invalid_argument("Index out of range");

			if (startColumn + columnLength > columns)
				throw invalid_argument("Index out of range");

			T* temp = Data + startColumn + columns * startRow;
			T* rowPointer;
			T* rowPointer2;
			for (int i = 0; i < rowLength; i++)
			{
				rowPointer = temp + columns * i;
				rowPointer2 = subMatrix.Data + i * columnLength;
				memcpy(rowPointer, rowPointer2, columnLength * sizeof(T));
			}
		}

		template<class T>
		Matrix<T> Matrix<T>::AddZeroBorder(int leftSize, int rightSize, int upSize,
			int downSize) const
		{
			Matrix<T> result = Zeros(rows + upSize + downSize,
				columns + rightSize + leftSize);
			result.SetSubMatrix(upSize, leftSize, *this);
			return result;
		}

		template<class T>
		Matrix<T> Matrix<T>::AddBorder(int leftSize, int rightSize, int upSize,
			int downSize, T value) const
		{
			Matrix<T> result(rows + upSize + downSize, columns + rightSize + leftSize,
				value);
			result.SetSubMatrix(upSize, leftSize, *this);
			return result;
		}

		template<class T>
		Matrix<T> Matrix<T>::AddZeroBorder(int size) const
		{
			Matrix<T> result = Zeros(rows + 2 * size, columns + 2 * size);
			result.SetSubMatrix(size, size, *this);
			return result;
		}

		template<class T>
		Matrix<T> Matrix<T>::AddBorder(int size, T value) const
		{
			Matrix<T> result(rows + 2 * size, columns + 2 * size, value);
			result.SetSubMatrix(size, size, *this);
			return result;
		}

		template<class T>
		Matrix<T> Matrix<T>::AppendUp(Matrix<T> matrix) const
		{
			if (matrix.ColumnCount() != columns)
				throw invalid_argument(
				"Cannot append a matrix that has different column count");

			Matrix<T> result(rows + matrix.RowCount(), columns);
			memcpy(result.Data, matrix.Data, matrix.ElementCount() * sizeof(T));
			memcpy(result.Data + matrix.ElementCount(), Data, elementCount * sizeof(T));
			return result;
		}

		template<class T>
		Matrix<T> Matrix<T>::AppendDown(Matrix<T> matrix) const
		{
			if (matrix.ColumnCount() != columns)
				throw invalid_argument(
				"Cannot append a matrix that has different column count");

			Matrix<T> result(rows + matrix.RowCount(), columns);
			memcpy(result.Data, Data, elementCount * sizeof(T));
			memcpy(result.Data + elementCount, matrix.Data, matrix.ElementCount() * sizeof(T));
			return result;
		}

		template<class T>
		Matrix<T> Matrix<T>::AppendRight(Matrix<T> matrix) const
		{
			if (matrix.RowCount() != rows)
				throw invalid_argument(
				"Cannot append a matrix that has different row count");

			auto inputColumns = matrix.ColumnCount();
			auto resultColumns = columns + inputColumns;
			Matrix<T> result(rows, resultColumns);
			T* startRowPosition = result.Data + columns;
			for (int i = 0; i < rows; i++)
			{
				auto temp = i * resultColumns;
				memcpy(result.Data + temp,  Data + i * columns, columns * sizeof(T));
				memcpy(startRowPosition + temp, matrix.Data + i * inputColumns, inputColumns * sizeof(T));
			}
			return result;
		}

		template<class T>
		Matrix<T> Matrix<T>::AppendLeft(Matrix<T> matrix) const
		{
			if (matrix.RowCount() != rows)
				throw invalid_argument(
				"Cannot append a matrix that has different row count");

			auto inputColumns = matrix.ColumnCount();
			auto resultColumns = columns + inputColumns;
			Matrix<T> result(rows, resultColumns);
			T* startRowPosition = result.Data + inputColumns;
			for (int i = 0; i < rows; i++)
			{
				auto temp = i * resultColumns;
				memcpy(result.Data + temp, matrix.Data + i * inputColumns, inputColumns * sizeof(T));
				memcpy(startRowPosition + temp, Data + i * columns, columns * sizeof(T));
			}
			return result;
		}

		template<class T>
		Matrix<T> Matrix<T>::Reshape(int height, int width) const
		{
			if ((height * width) != elementCount)
				throw invalid_argument(
				"You cannot reshape the matrix into something that doesn't contains the elements");
			return Matrix<T>(height, width, Data);
		}

		template<class T>
		Matrix<T> Matrix<T>::Rotate90() const
		{
			Matrix<T> result(columns, rows);
			int temp = rows - 1;
			for (int i = 0; i < columns; i++)
				for (int j = 0; j < rows; j++)
					result.Data[i * rows + temp - j] = Data[j * columns + i];

			return result;
		}

		template<class T>
		Matrix<T> Matrix<T>::Rotate180() const
		{
			Matrix<T> result(rows, columns);
			int temp = rows - 1;
			int temp2 = columns - 1;
			for (int j = 0; j < rows; j++)
				for (int i = 0; i < columns; i++)
					result.Data[(temp - j) * columns + temp2 - i] =
					Data[j * columns + i];

			return result;
		}

		template<class T>
		Matrix<T> Matrix<T>::Rotate270() const
		{
			Matrix<T> result(columns, rows);
			int temp2 = columns - 1;
			for (int i = 0; i < columns; i++)
				for (int j = 0; j < rows; j++)
					result.Data[(temp2 - i) * rows + j] = Data[j * columns + i];

			return result;
		}


		template<class T>
		Matrix<T> Matrix<T>::VanillaDownSample(int widthSamplingSize, int heightSamplingSize) const
		{
			int resultRows = static_cast<int>(floor(double(rows) / heightSamplingSize));
			int resultColumns  = static_cast<int>(floor(double(columns) / widthSamplingSize));

			resultRows = resultRows == 0 ? 1 : resultRows;
			resultColumns = resultColumns == 0 ? 1 : resultColumns;

			Matrix<T> result(resultRows, resultColumns);
			int tempIndex;
			int tempIndex2;
			for (int i = 0; i < resultRows; i++)
			{
				tempIndex = i * resultColumns;
				tempIndex2 = (i * heightSamplingSize) * columns;
				for (int j = 0; j < resultColumns; j++)
					result.Data[tempIndex + j] = Data[tempIndex2 + j * widthSamplingSize];
			}

			return result;
		}

		template<class T>
		Matrix<T> Matrix<T>::VanillaUpSample(int widthSamplingSize, int heightSamplingSize, int resultRows, int resultColumns) const
		{
			Matrix<T> result = Matrix<T>::Zeros(resultRows, resultColumns);
			int tempIndex;
			int tempIndex2;
			for (int i = 0; i < rows; i++)
			{
				tempIndex = (i * heightSamplingSize) * resultColumns;
				tempIndex2 = i * columns;
				for (int j = 0; j < columns; j++)
					result.Data[tempIndex + j * widthSamplingSize] = Data[tempIndex2 + j];
			}

			return result;
		}

		template<class T>
		Matrix<T> Matrix<T>::MaxDownSample(int widthSamplingSize, int heightSamplingSize) const
		{
			int resultRows = static_cast<int>(floor(double(rows) / heightSamplingSize));
			int resultColumns  = static_cast<int>(floor(double(columns) / widthSamplingSize));

			resultRows = resultRows == 0 ? 1 : resultRows;
			resultColumns = resultColumns == 0 ? 1 : resultColumns;

			Matrix<T> result(resultRows, resultColumns);

			int widthRemainder = columns % widthSamplingSize;
			int heightRemainder = rows % heightSamplingSize;

			if (widthRemainder != 0 || heightRemainder || 0)
			{
				throw invalid_argument("Not implemented at the moment");
			}
			else
			{
				T maxValue;
				T currentValue;
				int startRowIndex;
				int endRowIndex;
				int startColumnIndex;
				int endColumnIndex;
				int tempIndex;
				int tempIndex2;

				for (int i = 0; i < resultRows; i++)
				{
					tempIndex2 = i * resultColumns;
					startRowIndex = i * heightSamplingSize;
					endRowIndex = startRowIndex + heightSamplingSize; 
					for (int j = 0; j < resultColumns; j++)
					{
						maxValue = numeric_limits<float>::min();
						startColumnIndex = j * widthSamplingSize;
						endColumnIndex = startColumnIndex + widthSamplingSize;
						for (int k = startRowIndex; k < endRowIndex; k++)
						{
							tempIndex = k * columns;
							for (int l = startColumnIndex; l < endColumnIndex; l++)
							{
								currentValue = Data[tempIndex + l];
								if (currentValue > maxValue)
									maxValue = currentValue;
							}
						}

						result.Data[tempIndex2 + j] = maxValue;
					}
				}
			}

			return result;
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
		T Matrix<T>::Sum() const
		{
			T result = 0;
			for (int i = 0; i < elementCount; i++)
				result += Data[i];

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
		Matrix<T>& Matrix<T>::operator=(Matrix<T> && other)
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
				throw invalid_argument("The dimensions has to be valid");
			else if (columns <= 0)
				throw invalid_argument("The dimensions has to be valid");

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
				throw invalid_argument("The dimensions has to be valid");
			else if (columns <= 0)
				throw invalid_argument("The dimensions has to be valid");

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
				throw invalid_argument("The dimensions has to be valid");
			else if (columns <= 0)
				throw invalid_argument("The dimensions has to be valid");

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
				throw invalid_argument("The dimensions has to be valid");
			else if (columns <= 0)
				throw invalid_argument("The dimensions has to be valid");

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
				throw invalid_argument("The dimensions has to be valid");
			else if (columns <= 0)
				throw invalid_argument("The dimensions has to be valid");

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

		template class Matrix<float> ;
		template class Matrix<double> ;
		template class Matrix<long double> ;

	} /* namespace Math */
} /* namespace Matuna */
