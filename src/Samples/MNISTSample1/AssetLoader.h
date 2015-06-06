/*
 * AssetLoader.h
 *
 *  Created on: Jun 4, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_SAMPLES_MNISTSAMPLE1_ASSETLOADER_H_
#define MATUNA_SAMPLES_MNISTSAMPLE1_ASSETLOADER_H_

#include "Math/Matrix.h"
#include "Helper/Path.h"
#include "Helper/FileHelper.h"
#include <vector>
#include <ios>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace Matuna::Math;
using namespace Matuna::Helper;
using namespace std;

inline int ReverseInt(int i)
{
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

template<class T>
class AssetLoader
{
private:
	static vector<Matrix<T>> ReadMNIST(string path, int count)
	{
		vector<Matrix<T>> result;
		ifstream file(path, ios::in | ios::binary);
		if (file.is_open())
		{
			int magicNumber = 0;
			int numberImages = 0;
			int rowCount = 0;
			int columnCount = 0;
			file.read((char*) &magicNumber, sizeof(magicNumber));
			magicNumber = ReverseInt(magicNumber);

			if (magicNumber != 2051)
				throw runtime_error("The endianess is not correct or int size");

			file.read((char*) &numberImages, sizeof(numberImages));
			numberImages = ReverseInt(numberImages);
			file.read((char*) &rowCount, sizeof(rowCount));
			rowCount = ReverseInt(rowCount);
			file.read((char*) &columnCount, sizeof(columnCount));
			columnCount = ReverseInt(columnCount);

			if (rowCount != columnCount)
				throw runtime_error("The dimension does not match");

			if (count > numberImages)
				throw invalid_argument(
						"The requested labels are more than existing labels");

			unsigned char pixelValue = 0;
			auto char_size = sizeof(unsigned char);
			for (int i = 0; i < count; ++i)
			{
				Matrix<T> image(rowCount, columnCount);
				for (int r = 0; r < rowCount; ++r)
				{
					for (int c = 0; c < columnCount; ++c)
					{
						file.read(reinterpret_cast<char*>(&pixelValue),
								char_size);
						image.At(r, c) = (T) pixelValue / T(255);
					}
				}

				result.push_back(image);
			}

			file.close();
		}

		return result;
	}

	static vector<Matrix<T>> ReadMNISTLabels(string path, int count)
	{
		vector<Matrix<T>> result;
		ifstream file(path, ios::in | ios::binary);
		if (file.is_open())
		{
			int magicNumber = 0;
			int numberLabels = 0;
			file.read((char*) &magicNumber, sizeof(magicNumber));
			magicNumber = ReverseInt(magicNumber);

			if (magicNumber != 2049)
				throw runtime_error("The endianess is not correct or int size");

			file.read((char*) &numberLabels, sizeof(numberLabels));
			numberLabels = ReverseInt(numberLabels);

			if (count > numberLabels)
				throw invalid_argument(
						"The requested labels are more than existing labels");

			for (int i = 0; i < count; ++i)
			{
				Matrix<T> target = Matrix<T>::Zeros(10, 1);
				unsigned char temp = 0;
				file.read((char*) &temp, sizeof(temp));
				target.At(temp, 0) = 1;
				result.push_back(target);
			}
		}

		return result;
	}

public:
	AssetLoader();
	~AssetLoader();

	static vector<Matrix<T>> ReadTestImages(int count = 10000)
	{
		string path = Path::Combine(Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "assets");
		path = Path::Combine(path, "MNIST", "t10k-images.idx3-ubyte");
		return ReadMNIST(path, count);
	}
	static vector<Matrix<T>> ReadTestTargets(int count = 10000)
	{
		string path = Path::Combine(Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "assets");
		path = Path::Combine(path, "MNIST", "t10k-labels.idx1-ubyte");
		return ReadMNISTLabels(path, count);
	}

	static vector<Matrix<T>> ReadTrainingImages(int count = 60000)
	{
		string path = Path::Combine(Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "assets");
		path = Path::Combine(path, "MNIST", "train-images.idx3-ubyte");
		return ReadMNIST(path, count);
	}

	static vector<Matrix<T>> ReadTrainingTargets(int count = 60000)
	{
		string path = Path::Combine(Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "assets");
		path = Path::Combine(path, "MNIST", "train-labels.idx1-ubyte");
		return ReadMNISTLabels(path, count);
	}
};

#endif /* MATUNA_SAMPLES_MNISTSAMPLE1_ASSETLOADER_H_ */
