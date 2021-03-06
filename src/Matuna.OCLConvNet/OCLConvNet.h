/*
 * OCLConvNet.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_OCLCONVNET_OCLCONVNET_H_
#define MATUNA_MATUNA_OCLCONVNET_OCLCONVNET_H_

#include "Matuna.ConvNet/ConvNetConfig.h"
#include "Matuna.ConvNet/ConvNet.h"
#include "Matuna.ConvNet/TrainableConvNet.h"
#include "Matuna.ConvNet/ConvNetTrainer.h"
#include "Matuna.ConvNet/IAlgorithmConfig.h"

#include "Matuna.OCLHelper/OCLDeviceInfo.h"
#include "Matuna.OCLHelper/OCLContext.h"
#include "Matuna.OCLHelper/OCLDevice.h"

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
class OCLConvNet final: public TrainableConvNet<T>
{
private:
	vector<shared_ptr<OCLContext>> contexts;
	vector<unique_ptr<OCLForwardBackPropLayer<T>>> layers;
	unique_ptr<StandardOutputLayer<T>> outputLayer;
public:
	OCLConvNet(const vector<OCLDeviceInfo>& devices, unique_ptr<ConvNetConfig> config);
	virtual ~OCLConvNet();

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

	virtual void TrainNetwork(unique_ptr<ConvNetTrainer<T>> trainer,
			unique_ptr<IAlgorithmConfig> algorithm) override;

	vector<OCLForwardBackPropLayer<T>*> GetLayers() const;
	StandardOutputLayer<T>* GetOutputLayer() const;
	vector<OCLContext*> GetOCLContexts() const;

private:
	void InitializeContexts(const vector<OCLDeviceInfo>& devices);
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_OCLCONVNET_OCLCONVNET_H_ */
