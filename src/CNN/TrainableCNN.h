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

	unique_ptr<T[]> AlignToOutput(T* input, int formatIndex) const;
	unique_ptr<T[]> AlignToInput(T* input, int formatIndex) const;
	unique_ptr<T[]> UnalignFromOutput(T* input, int formatIndex) const;
	unique_ptr<T[]> UnalignFromInput(T* input, int formatIndex) const;

	bool RequireInputAlignment(int formatIndex) const;
	bool RequireOutputAlignment(int formatIndex) const;

	virtual unique_ptr<T[]> FeedForwardAligned(T* input, int formatIndex) = 0;
	unique_ptr<T[]> FeedForwardUnaligned(T* input, int formatIndex);

	virtual T CalculateErrorAligned(T* propagatedValue, int formatIndex,
			T* target)= 0;

	virtual unique_ptr<T[]> CalculateGradientAligned(T* input, int formatIndex)= 0;
	unique_ptr<T[]> CalculateGradientUnaligned(T* input, int formatIndex);

	virtual unique_ptr<T[]> GetParameters()= 0;

	virtual void SetParameters(T* parameters) = 0;

	virtual size_t GetParameterCount()= 0;

	virtual void TrainNetwork(unique_ptr<CNNTrainer<T>> trainer,
			unique_ptr<IAlgorithmConfig> algorithm) = 0;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_TRAINABLECNN_H_ */
