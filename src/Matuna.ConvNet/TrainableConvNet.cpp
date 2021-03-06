/*
 * TrainableConvNet.cpp
 *
 *  Created on: May 9, 2015
 *      Author: Mikael
 */

#include "TrainableConvNet.h"

namespace Matuna
{
namespace MachineLearning
{

template<class T>
TrainableConvNet<T>::TrainableConvNet(const ConvNetConfig& config) :
		ConvNet(config)
{

}

template<class T>
TrainableConvNet<T>::~TrainableConvNet()
{

}

template<class T>
bool TrainableConvNet<T>::RequireForwardInputAlignment(int formatIndex) const
{
	auto inputDataDesc = this->InputForwardDataDescriptions()[formatIndex];
	auto inputMemDesc = this->InputForwardMemoryDescriptions()[formatIndex];

	return !(inputDataDesc.Width == inputMemDesc.Width
			&& inputDataDesc.Height == inputMemDesc.Height
			&& inputDataDesc.Units == inputMemDesc.Units);
}

template<class T>
bool TrainableConvNet<T>::RequireForwardOutputAlignment(int formatIndex) const
{
	auto outputDataDesc = this->OutputForwardDataDescriptions()[formatIndex];
	auto outputMemDesc = this->OutputForwardMemoryDescriptions()[formatIndex];

	return !(outputDataDesc.Width == outputMemDesc.Width
			&& outputDataDesc.Height == outputMemDesc.Height
			&& outputDataDesc.Units == outputMemDesc.Units);
}

template<class T>
bool TrainableConvNet<T>::RequireBackOutputAlignment(int formatIndex) const
{
	auto outBackDataDesc = this->OutputBackDataDescriptions()[formatIndex];
	auto outBackMemDesc = this->OutputBackMemoryDescriptions()[formatIndex];

	return !(outBackDataDesc.Width == outBackMemDesc.Width
		&& outBackDataDesc.Height == outBackMemDesc.Height
		&& outBackDataDesc.Units == outBackMemDesc.Units);
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
unique_ptr<T[]> TrainableConvNet<T>::AlignToForwardOutput(T* output, int formatIndex) const
{
	auto outputDataDesc = this->OutputForwardDataDescriptions()[formatIndex];
	auto outputMemDesc = this->OutputForwardMemoryDescriptions()[formatIndex];

	return move(AlignmentHelper<T>(outputDataDesc, outputMemDesc, output));
}

template<class T>
unique_ptr<T[]> TrainableConvNet<T>::AlignToForwardInput(T* input, int formatIndex) const
{
	auto inputDataDesc = this->InputForwardDataDescriptions()[formatIndex];
	auto inputMemDesc = this->InputForwardMemoryDescriptions()[formatIndex];

	return move(AlignmentHelper<T>(inputDataDesc, inputMemDesc, input));
}

template<class T>
unique_ptr<T[]> TrainableConvNet<T>::UnalignFromForwardOutput(T* output,
		int formatIndex) const
{
	auto outputDataDesc = this->OutputForwardDataDescriptions()[formatIndex];
	auto outputMemDesc = this->OutputForwardMemoryDescriptions()[formatIndex];

	return move(UnalignmentHelper<T>(outputDataDesc, outputMemDesc, output));
}

template<class T>
unique_ptr<T[]> TrainableConvNet<T>::UnalignFromForwardInput(T* input,
		int formatIndex) const
{
	auto inputDataDesc = this->InputForwardDataDescriptions()[formatIndex];
	auto inputMemDesc = this->InputForwardMemoryDescriptions()[formatIndex];

	return move(UnalignmentHelper<T>(inputDataDesc, inputMemDesc, input));
}

template<class T>
unique_ptr<T[]> TrainableConvNet<T>::AlignToBackOutput(T* input, int formatIndex) const
{
	auto outBackDataDesc = this->OutputBackDataDescriptions()[formatIndex];
	auto outBackMemDesc = this->OutputBackMemoryDescriptions()[formatIndex];

	return move(AlignmentHelper<T>(outBackDataDesc, outBackMemDesc, input));
}

template<class T>
unique_ptr<T[]> TrainableConvNet<T>::UnalignFromBackOutput(T* input, int formatIndex) const
{
	auto outBackDataDesc = this->OutputBackDataDescriptions()[formatIndex];
	auto outBackMemDesc = this->OutputBackMemoryDescriptions()[formatIndex];

	return move(UnalignmentHelper<T>(outBackDataDesc, outBackMemDesc, input));
}

template<class T>
unique_ptr<T[]> TrainableConvNet<T>::FeedForwardUnaligned(T* input, int formatIndex)
{
	if (RequireForwardInputAlignment(formatIndex))
	{
		unique_ptr<T[]> alignedInput = move(AlignToForwardInput(input, formatIndex));
		unique_ptr<T[]> alignedOutput = move(
				FeedForwardAligned(alignedInput.get(), formatIndex));

		if (RequireForwardOutputAlignment(formatIndex))
			return move(UnalignFromForwardOutput(alignedOutput.get(), formatIndex));
		else
			return move(alignedOutput);
	}
	else
	{
		unique_ptr<T[]> alignedOutput = move(
				FeedForwardAligned(input, formatIndex));
		if (RequireForwardOutputAlignment(formatIndex))
			return move(UnalignFromForwardOutput(alignedOutput.get(), formatIndex));
		else
			return move(alignedOutput);
	}
}

template<class T>
T TrainableConvNet<T>::CalculateErrorUnaligned(T* input, int formatIndex, T* target)
{
	if (RequireForwardInputAlignment(formatIndex))
	{
		unique_ptr<T[]> alignedInput = move(AlignToForwardInput(input, formatIndex));
		if (RequireForwardOutputAlignment(formatIndex))
		{
			unique_ptr<T[]> alignedTarget = move(AlignToForwardOutput(target, formatIndex));
			return CalculateErrorAligned(alignedInput.get(), formatIndex, alignedTarget.get());
		}
		else
			return CalculateErrorAligned(alignedInput.get(), formatIndex, target);
	}
	else
	{
		if (RequireForwardOutputAlignment(formatIndex))
		{
			unique_ptr<T[]> alignedTarget = move(AlignToForwardOutput(target, formatIndex));
			return CalculateErrorAligned(input, formatIndex, alignedTarget.get());
		}
		else
			return CalculateErrorAligned(input, formatIndex, target);
	}
}

template<class T>
unique_ptr<T[]> TrainableConvNet<T>::BackPropUnaligned(T* input, int formatIndex, T* target)
{
	if (RequireForwardInputAlignment(formatIndex))
	{
		unique_ptr<T[]> alignedInput = move(AlignToForwardInput(input, formatIndex));
		unique_ptr<T[]> alignedOutput;
		if (RequireForwardOutputAlignment(formatIndex))
		{
			unique_ptr<T[]> alignedTarget = move(AlignToForwardOutput(target, formatIndex));
			alignedOutput = move(BackPropAligned(alignedInput.get(), formatIndex, alignedTarget.get()));
		}
		else
			alignedOutput = move(BackPropAligned(alignedInput.get(), formatIndex, target));

		if (RequireBackOutputAlignment(formatIndex))
			return move(UnalignFromBackOutput(alignedOutput.get(), formatIndex));
		else
			return move(alignedOutput);
	}
	else
	{
		unique_ptr<T[]> alignedOutput;
		if (RequireForwardOutputAlignment(formatIndex))
		{
			unique_ptr<T[]> alignedTarget = move(AlignToForwardOutput(target, formatIndex));
			alignedOutput = move(BackPropAligned(input, formatIndex, alignedTarget.get()));
		}
		else
			alignedOutput = move(BackPropAligned(input, formatIndex, target));

		if (RequireBackOutputAlignment(formatIndex))
			return move(UnalignFromBackOutput(alignedOutput.get(), formatIndex));
		else
			return move(alignedOutput);
	}
}

template<class T>
unique_ptr<T[]> TrainableConvNet<T>::CalculateGradientUnaligned(T* input,
	int formatIndex, T* target)
{
	if (RequireForwardInputAlignment(formatIndex))
	{
		unique_ptr<T[]> alignedInput = move(AlignToForwardInput(input, formatIndex));
		if (RequireForwardOutputAlignment(formatIndex))
		{
			unique_ptr<T[]> alignedTarget = move(AlignToForwardOutput(target, formatIndex));
			return move(CalculateGradientAligned(alignedInput.get(), formatIndex, alignedTarget.get()));
		}
		else
			return move(CalculateGradientAligned(alignedInput.get(), formatIndex, target));
	}
	else
	{
		if (RequireForwardOutputAlignment(formatIndex))
		{
			unique_ptr<T[]> alignedTarget = move(AlignToForwardOutput(target, formatIndex));
			return move(CalculateGradientAligned(input, formatIndex, alignedTarget.get()));
		}
		else
			return move(CalculateGradientAligned(input, formatIndex, target));
	}
}

//Just add a type if the network is suppose to support more types.

template class TrainableConvNet<float> ;
template class TrainableConvNet<double> ;
template class TrainableConvNet<long double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
