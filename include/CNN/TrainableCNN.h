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

	virtual void FeedForward(const T* input, int formatIndex, T* output) = 0;

	virtual T CalculateError(const T* propagatedValue, int formatIndex,
			const T* target)= 0;

	virtual void CalculateGradient(const T* input, int formatIndex,
			T* output)= 0;

	virtual void GetParameters(T* parameters)= 0;

	virtual size_t GetParameterCount()= 0;

	virtual void TrainNetwork(unique_ptr<CNNTrainer<T>> trainer,
			unique_ptr<IAlgorithmConfig> algorithm) = 0;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_TRAINABLECNN_H_ */
