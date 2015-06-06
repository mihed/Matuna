/*
 * OCLConvNet.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_OCLConvNet_OCLConvNet_H_
#define MATUNA_OCLConvNet_OCLConvNet_H_

#include "ConvNet/ConvNetConfig.h"
#include "ConvNet/ConvNet.h"
#include "ConvNet/TrainableConvNet.h"
#include "ConvNet/ConvNetTrainer.h"
#include "ConvNet/IAlgorithmConfig.h"

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

#endif /* MATUNA_OCLConvNet_OCLConvNet_H_ */
