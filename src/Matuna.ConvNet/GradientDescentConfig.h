/*
 * GradientDescentConfig.h
 *
 *  Created on: Jun 3, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_CONVNET_GRADIENTDESCENTCONFIG_H_
#define MATUNA_MATUNA_CONVNET_GRADIENTDESCENTCONFIG_H_

#include "IAlgorithmConfig.h"
#include <functional>
#include <vector>

using namespace std;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class GradientDescentConfig: public IAlgorithmConfig
{
private:
	int batchSize;
	int epochs;
	int samplesPerEpoch;
	T momentum;
	function<T(int)> stepSizeCallback;

public:
	GradientDescentConfig();
	~GradientDescentConfig();

	void SetBatchSize(int batchSize);
	void SetSamplesPerEpoch(int samplesPerEpoch);
	void SetEpochs(int epochs);
	void SetMomentum(T momentum);
	void SetStepSizeCallback(function<T(int)> stepSizeCallback);

	function<T(int)> GetStepSizeCallback();
	int GetBatchSize();
	int GetSamplesPerEpoch();
	int GetEpochs();
	T GetMomentum();
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_CONVNET_GRADIENTDESCENTCONFIG_H_ */
