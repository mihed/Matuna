/*
 * CNNOCL.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNNOCL_CNNOCL_H_
#define MATUNA_CNNOCL_CNNOCL_H_

#include "CNN/CNNConfig.h"
#include "CNN/CNN.h"
#include "CNN/TrainableCNN.h"
#include "CNN/CNNTrainer.h"
#include "CNN/IAlgorithmConfig.h"

#include "OCLHelper/OCLDeviceInfo.h"
#include "OCLHelper/OCLContext.h"
#include "OCLHelper/OCLDevice.h"

#include "OCLForwardBackPropLayer.h"
#include "StandardOutputLayer.h"

#include <memory>
#include <vector>

using namespace std;
using namespace Matuna::Helper;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class CNNOCL final: public TrainableCNN<T>
{
private:
	vector<shared_ptr<OCLContext>> contexts;
	vector<unique_ptr<OCLForwardBackPropLayer<T>>> layers;
	unique_ptr<StandardOutputLayer<T>> outputLayer;
public:
	CNNOCL(const vector<OCLDeviceInfo>& devices, unique_ptr<CNNConfig> config);
	virtual ~CNNOCL();

	virtual unique_ptr<T[]> FeedForwardAligned(T* input, int formatIndex)
			override;

	virtual T CalculateErrorFromForwardAligned(T* propagatedValue, int formatIndex,
			T* target) override;

	virtual T CalculateErrorAligned(T* input, int formatIndex, T* target) override;

	virtual unique_ptr<T[]> BackPropAligned(T* input, int formatIndex, T* target) override;

	virtual unique_ptr<T[]> CalculateGradientAligned(T* input, int formatIndex, T* target)
			override;

	virtual unique_ptr<T[]> GetParameters() override;

	virtual void SetParameters(T* parameters) override;

	virtual size_t GetParameterCount() override;

	virtual void TrainNetwork(unique_ptr<CNNTrainer<T>> trainer,
			unique_ptr<IAlgorithmConfig> algorithm) override;

	vector<OCLForwardBackPropLayer<T>*> GetLayers() const;
	StandardOutputLayer<T>* GetOutputLayer() const;
	vector<OCLContext*> GetOCLContexts() const;

private:
	void InitializeContexts(const vector<OCLDeviceInfo>& devices);
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_CNNOCL_CNNOCL_H_ */
