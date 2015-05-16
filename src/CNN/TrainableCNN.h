/*
 * TrainableCNN.h
 *
 *  Created on: May 9, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_TRAINABLECNN_H_
#define ATML_CNN_TRAINABLECNN_H_

#include "CNN.h"
#include "CNNConfig.h"
#include "IAlgorithmConfig.h"
#include "CNNTrainer.h"

#include <memory>

namespace ATML
{
namespace MachineLearning
{

template<class T>
class TrainableCNN: public CNN
{
public:
	TrainableCNN(const CNNConfig& config);
	virtual ~TrainableCNN();

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

	virtual T CalculateErrorAligned(T* propagatedValue, int formatIndex,
			T* target)= 0;

	virtual unique_ptr<T[]> BackPropAligned(T* input, int formatIndex, T* target) = 0;

	//The target is also unaligned
	unique_ptr<T[]> BackPropUnaligned(T* input, int formatIndex, T* target);

	virtual unique_ptr<T[]> CalculateGradientAligned(T* input, int formatIndex, T* target) = 0;

	//The target is also unaligned
	unique_ptr<T[]> CalculateGradientUnaligned(T* input, int formatIndex, T* target);

	virtual unique_ptr<T[]> GetParameters()= 0;

	virtual void SetParameters(T* parameters) = 0;

	virtual size_t GetParameterCount()= 0;

	virtual void TrainNetwork(unique_ptr<CNNTrainer<T>> trainer,
			unique_ptr<IAlgorithmConfig> algorithm) = 0;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_TRAINABLECNN_H_ */
