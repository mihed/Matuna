/*
 * CNNOpenCL.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNNOPENCL_CNNOPENCL_H_
#define ATML_CNNOPENCL_CNNOPENCL_H_

#include "CNN/CNNConfig.h"
#include "CNN/CNN.h"
#include "CNN/TrainableCNN.h"
#include "CNN/CNNTrainer.h"
#include "CNN/IAlgorithmConfig.h"

#include "OpenCLHelper/OpenCLContext.h"
#include "OpenCLForwardBackPropLayer.h"
#include "StandardOutputLayer.h"

#include <memory>

using namespace std;
using namespace ATML::Helper;

namespace ATML
{
namespace MachineLearning
{

template<class T>
class CNNOpenCL final: public TrainableCNN<T>
{
private:
	shared_ptr<OpenCLContext> context;
	vector<unique_ptr<OpenCLForwardBackPropLayer<T>>> layers;
	unique_ptr<StandardOutputLayer<T>> outputLayer;
public:
	//TODO: Add a vector of contexts
	CNNOpenCL(unique_ptr<OpenCLContext> context, unique_ptr<CNNConfig> config);
	virtual ~CNNOpenCL();

	virtual unique_ptr<T[]> FeedForwardAligned(T* input, int formatIndex)
			override;

	virtual T CalculateErrorAligned(T* propagatedValue, int formatIndex,
			T* target) override;

	virtual unique_ptr<T[]> CalculateGradientAligned(T* input, int formatIndex)
			override;

	virtual unique_ptr<T[]> GetParameters() override;

	virtual void SetParameters(T* parameters) override;

	virtual size_t GetParameterCount() override;

	virtual void TrainNetwork(unique_ptr<CNNTrainer<T>> trainer,
			unique_ptr<IAlgorithmConfig> algorithm) override;

	vector<OpenCLForwardBackPropLayer<T>*> GetLayers();
	StandardOutputLayer<T>* GetOutputLayer();
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNNOPENCL_CNNOPENCL_H_ */
