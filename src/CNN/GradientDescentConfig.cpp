/*
 * GradientDescentConfig.cpp
 *
 *  Created on: Jun 3, 2015
 *      Author: Mikael
 */

#include "GradientDescentConfig.h"
#include <stdexcept>

namespace ATML
{
namespace MachineLearning
{

template<class T>
GradientDescentConfig<T>::GradientDescentConfig()
{
	batchSize = 0;
	epochs = 0;
	momentum = 0;
	samplesPerEpoch = 0;
	stepSizeCallback = nullptr;
}

template<class T>
GradientDescentConfig<T>::~GradientDescentConfig()
{

}

template<class T>
void GradientDescentConfig<T>::SetSamplesPerEpoch(int samplesPerEpoch)
{
	this->samplesPerEpoch = samplesPerEpoch;
}

template<class T>
void GradientDescentConfig<T>::SetBatchSize(int batchSize)
{
	this->batchSize = batchSize;
}

template<class T>
void GradientDescentConfig<T>::SetEpochs(int epochs)
{
	this->epochs = epochs;
}

template<class T>
void GradientDescentConfig<T>::SetMomentum(T momentum)
{
	throw runtime_error("Not implemented");
	this->momentum = momentum;
}

template<class T>
void GradientDescentConfig<T>::SetStepSizeCallback(
		function<T(int)> stepSizeCallback)
{
	this->stepSizeCallback = stepSizeCallback;
}

template<class T>
function<T(int)> GradientDescentConfig<T>::GetStepSizeCallback()
{
	return stepSizeCallback;
}

template<class T>
int GradientDescentConfig<T>::GetSamplesPerEpoch()
{
	return samplesPerEpoch;
}

template<class T>
int GradientDescentConfig<T>::GetBatchSize()
{
	return batchSize;
}

template<class T>
int GradientDescentConfig<T>::GetEpochs()
{
	return epochs;
}

template<class T>
T GradientDescentConfig<T>::GetMomentum()
{
	throw runtime_error("Not implemented");
	return momentum;
}

template class GradientDescentConfig<float> ;
template class GradientDescentConfig<double> ;
template class GradientDescentConfig<long double> ;

} /* namespace MachineLearning */
} /* namespace ATML */
