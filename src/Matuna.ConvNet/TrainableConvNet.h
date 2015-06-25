/*
 * TrainableConvNet.h
 *
 *  Created on: May 9, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_CONVNET_TRAINABLECONVNET_H_
#define MATUNA_MATUNA_CONVNET_TRAINABLECONVNET_H_

#include "ConvNet.h"
#include "ConvNetConfig.h"
#include "IAlgorithmConfig.h"
#include "ConvNetTrainer.h"

#include <memory>

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class TrainableConvNet: public ConvNet
{
public:
	TrainableConvNet(const ConvNetConfig& config);
	virtual ~TrainableConvNet();

	unique_ptr<T[]> AlignToForwardOutput(T* input, int formatIndex) const;
	unique_ptr<T[]> AlignToForwardInput(T* input, int formatIndex) const;
	unique_ptr<T[]> UnalignFromForwardOutput(T* input, int formatIndex) const;
	unique_ptr<T[]> UnalignFromForwardInput(T* input, int formatIndex) const;

	//Since the input to the back propagation is the output from the forward propagation
	//we don't have any align to BackInput. (It's AlignToForwardOutput and UnalignFromForwardOutput)
	//This also implies that the targets are completely determined by the forward output alignment.

	unique_ptr<T[]> AlignToBackOutput(T* input, int formatIndex) const;
	unique_ptr<T[]> UnalignFromBackOutput(T* input, int formatIndex) const;

	bool RequireForwardInputAlignment(int formatIndex) const;
	bool RequireForwardOutputAlignment(int formatIndex) const;
	bool RequireBackOutputAlignment(int formatIndex) const;

	virtual unique_ptr<T[]> FeedForwardAligned(T* input, int formatIndex) = 0;
	unique_ptr<T[]> FeedForwardUnaligned(T* input, int formatIndex);

	virtual T CalculateErrorFromForwardAligned(T* propagatedValue, int formatIndex,
			T* target)= 0;

	virtual T CalculateErrorAligned(T* input, int formatIndex, T* target) = 0;
	T CalculateErrorUnaligned(T* input, int formatIndex, T* target);

	virtual unique_ptr<T[]> BackPropAligned(T* input, int formatIndex, T* target) = 0;

	//The target is also unaligned
	unique_ptr<T[]> BackPropUnaligned(T* input, int formatIndex, T* target);

	virtual unique_ptr<T[]> CalculateGradientAligned(T* input, int formatIndex, T* target) = 0;

	//The target is also unaligned
	unique_ptr<T[]> CalculateGradientUnaligned(T* input, int formatIndex, T* target);

	virtual unique_ptr<T[]> GetParameters()= 0;

	virtual void SetParameters(T* parameters) = 0;

	virtual size_t GetParameterCount()= 0;

	virtual void TrainNetwork(unique_ptr<ConvNetTrainer<T>> trainer,
			unique_ptr<IAlgorithmConfig> algorithm) = 0;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_CONVNET_TRAINABLECONVNET_H_ */
