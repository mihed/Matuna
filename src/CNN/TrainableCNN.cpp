/*
 * TrainableCNN.cpp
 *
 *  Created on: May 9, 2015
 *      Author: Mikael
 */

#include "TrainableCNN.h"

namespace ATML
{
namespace MachineLearning
{

//Just add a type if the network is suppose to support more types.

template class TrainableCNN<float> ;
template class TrainableCNN<double> ;
template class TrainableCNN<long double> ;

template<class T>
TrainableCNN<T>::TrainableCNN(const CNNConfig& config) :
		CNN(config)
{

}

template<class T>
TrainableCNN<T>::~TrainableCNN()
{

}

template<class T>
bool TrainableCNN<T>::RequireInputAlignment(int formatIndex) const
{
	auto inputDataDesc = this->InputDataDescriptions()[formatIndex];
	auto inputMemDesc = this->InputMemoryDescriptions()[formatIndex];

	return !(inputDataDesc.Width == inputMemDesc.Width
			&& inputDataDesc.Height == inputMemDesc.Height
			&& inputDataDesc.Units == inputMemDesc.Units);
}

template<class T>
bool TrainableCNN<T>::RequireOutputAlignment(int formatIndex) const
{
	auto outputDataDesc = this->OutputDataDescriptions()[formatIndex];
	auto outputMemDesc = this->OutputMemoryDescriptions()[formatIndex];

	return !(outputDataDesc.Width == outputMemDesc.Width
			&& outputDataDesc.Height == outputMemDesc.Height
			&& outputDataDesc.Units == outputMemDesc.Units);
}

template<class T>
unique_ptr<T[]> AlignmentHelper(const LayerDataDescription& inputDataDesc,
		const LayerMemoryDescription& inputMemDesc, T* input)
{
	unique_ptr<T[]> result(new T[inputMemDesc.TotalMemory()]);
	T* rawBuffer = result.get();
	int const1 = inputDataDesc.Width * inputDataDesc.Height;
	int const2 = inputMemDesc.Width * inputMemDesc.Height;
	int temp1, temp2, temp3, temp4;
	for (int z = 0; z < inputDataDesc.Units; z++)
	{
		temp1 = const1 * z;
		temp3 = const2 * (z + inputMemDesc.UnitOffset);
		for (int y = 0; y < inputDataDesc.Height; y++)
		{
			temp2 = inputDataDesc.Width * y + temp1;
			temp4 = inputMemDesc.Width * (y + inputMemDesc.HeightOffset)
					+ temp3;
			for (int x = 0; x < inputDataDesc.Width; x++)
				rawBuffer[temp4 + x + inputMemDesc.WidthOffset] = input[temp2
						+ x];
		}
	}

	return move(result);
}

template<class T>
unique_ptr<T[]> UnalignmentHelper(const LayerDataDescription& inputDataDesc,
		const LayerMemoryDescription& inputMemDesc, T* input)
{
	unique_ptr<T[]> result(new T[inputDataDesc.TotalUnits()]);
	T* rawBuffer = result.get();
	int const1 = inputDataDesc.Width * inputDataDesc.Height;
	int const2 = inputMemDesc.Width * inputMemDesc.Height;
	int temp1, temp2, temp3, temp4;
	for (int z = 0; z < inputDataDesc.Units; z++)
	{
		temp1 = const1 * z;
		temp3 = const2 * (z + inputMemDesc.UnitOffset);
		for (int y = 0; y < inputDataDesc.Height; y++)
		{
			temp2 = inputDataDesc.Width * y + temp1;
			temp4 = inputMemDesc.Width * (y + inputMemDesc.HeightOffset)
					+ temp3;
			for (int x = 0; x < inputDataDesc.Width; x++)
				rawBuffer[temp2 + x] = input[temp4 + x
						+ inputMemDesc.WidthOffset];
		}
	}

	return move(result);
}

template<class T>
unique_ptr<T[]> TrainableCNN<T>::AlignToOutput(T* output, int formatIndex) const
{
	auto outputDataDesc = this->OutputDataDescriptions()[formatIndex];
	auto outputMemDesc = this->OutputMemoryDescriptions()[formatIndex];

	return move(AlignmentHelper<T>(outputDataDesc, outputMemDesc, output));
}

template<class T>
unique_ptr<T[]> TrainableCNN<T>::AlignToInput(T* input, int formatIndex) const
{
	auto inputDataDesc = this->InputDataDescriptions()[formatIndex];
	auto inputMemDesc = this->InputMemoryDescriptions()[formatIndex];

	return move(AlignmentHelper<T>(inputDataDesc, inputMemDesc, input));
}

template<class T>
unique_ptr<T[]> TrainableCNN<T>::UnalignFromOutput(T* output,
		int formatIndex) const
{
	auto outputDataDesc = this->OutputDataDescriptions()[formatIndex];
	auto outputMemDesc = this->OutputMemoryDescriptions()[formatIndex];

	return move(UnalignmentHelper<T>(outputDataDesc, outputMemDesc, output));
}

template<class T>
unique_ptr<T[]> TrainableCNN<T>::UnalignFromInput(T* input, int formatIndex) const
{
	auto inputDataDesc = this->InputDataDescriptions()[formatIndex];
	auto inputMemDesc = this->InputMemoryDescriptions()[formatIndex];

	return move(UnalignmentHelper<T>(inputDataDesc, inputMemDesc, input));
}

template<class T>
unique_ptr<T[]> TrainableCNN<T>::FeedForwardUnaligned(T* input, int formatIndex)
{
	if (RequireInputAlignment(formatIndex))
	{
		unique_ptr<T[]> alignedInput = move(AlignToInput(input, formatIndex));
		unique_ptr<T[]> alignedOutput = move(FeedForwardAligned(alignedInput.get(), formatIndex));

		if (RequireOutputAlignment(formatIndex))
			return move(UnalignFromOutput(alignedOutput.get(), formatIndex));
		else
			return move(alignedOutput);
	}
	else
	{
		unique_ptr<T[]> alignedOutput = move(FeedForwardAligned(input, formatIndex));
		if (RequireOutputAlignment(formatIndex))
			return move(UnalignFromOutput(alignedOutput.get(), formatIndex));
		else
			return move(alignedOutput);
	}
}

template<class T>
unique_ptr<T[]> TrainableCNN<T>::CalculateGradientUnaligned(T* input,
		int formatIndex)
{
	if (RequireInputAlignment(formatIndex))
	{
		unique_ptr<T[]> alignedInput = move(AlignToInput(input, formatIndex));
		return move(CalculateGradientAligned(alignedInput.get(), formatIndex));
	}
	else
		return move(CalculateGradientAligned(input, formatIndex));
}

} /* namespace MachineLearning */
} /* namespace ATML */
