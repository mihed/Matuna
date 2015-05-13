/*
 * PerceptronLayer.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef CNNOPENCL_PERCEPTRONLAYER_H_
#define CNNOPENCL_PERCEPTRONLAYER_H_

#include "OpenCLForwardBackPropLayer.h"
#include "ForwardPerceptronKernel.h"
#include "CNN/PerceptronLayerConfig.h"
#include "Math/Matrix.h"
#include "OpenCLHelper/OpenCLContext.h"

#include <unordered_map>
#include <vector>
#include <memory>

using namespace std;
using namespace ATML::Helper;
using namespace ATML::Math;

namespace ATML
{
namespace MachineLearning
{

template<class T>
class PerceptronLayer: public OpenCLForwardBackPropLayer<T>
{

private:
	unordered_map<OpenCLDevice*, unique_ptr<ForwardPerceptronKernel<T>>> deviceAndKernels;
	unique_ptr<OpenCLMemory> weights;
	unique_ptr<OpenCLMemory> biases;
	PerceptronLayerConfig config;
	LayerDataDescription inputDescription;

public:
	PerceptronLayer(shared_ptr<OpenCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			const PerceptronLayerConfig* config);
	virtual ~PerceptronLayer();

	virtual void InterlockFinalized() override;

	virtual void EnqueueForwardPropagation(OpenCLDevice* device, int queueIndex,
			OpenCLMemory* previousInput, OpenCLMemory* output, bool blocking =
			true) override;

	virtual void EnqueueBackPropagation(OpenCLDevice* device, int queueIndex,
			OpenCLMemory* previousInput, OpenCLMemory* delta,
			OpenCLMemory* deltaOutput, bool blocking = true) override;

	virtual void GetParameters(T* parameters, OpenCLDevice* device,
			int queueIndex, bool blocking = true) override;

	virtual void SetParameters(T* parameters, OpenCLDevice* device,
			int queueIndex, bool blocking = true) override;

	virtual size_t GetParameterCount() override;

	PerceptronLayerConfig GetConfig() const;

	Matrix<T> GetWeights();
	Matrix<T> GetBias();

	//TODO: Add some read / write parameters. Now it's all random

private:
	void InitializeNormalPerceptron();
	void InitializeImagePerceptron();
	void InitializeParameters();
};

}
/* namespace MachineLearning */
} /* namespace ATML */

#endif /* CNNOPENCL_PERCEPTRONLAYER_H_ */
