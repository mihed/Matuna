/*
 * PerceptronLayer.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNNOPENCL_PERCEPTRONLAYER_H_
#define ATML_CNNOPENCL_PERCEPTRONLAYER_H_

#include "OpenCLForwardBackPropLayer.h"
#include "BackPerceptronKernel.h"
#include "ForwardPerceptronKernel.h"
#include "GradientPerceptronKernel.h"
#include "SimpleSumKernel.h"
#include "ImageForwardPerceptronKernel.h"
#include "ImageBackPerceptronKernel.h"
#include "DivideByScalarKernel.h"
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
	unordered_map<OpenCLDevice*, unique_ptr<ForwardPerceptronKernel<T>>> deviceAndForwardKernels;
	unordered_map<OpenCLDevice*, unique_ptr<ImageForwardPerceptronKernel<T>>> deviceAndImageForwardKernels;
	unordered_map<OpenCLDevice*, unique_ptr<BackPerceptronKernel<T>>> deviceAndBackKernels;
	unordered_map<OpenCLDevice*, unique_ptr<ImageBackPerceptronKernel<T>>> deviceAndImageBackKernels;
	unordered_map<OpenCLDevice*, unique_ptr<GradientPerceptronKernel<T>>> deviceAndGradientKernels;

	unique_ptr<OpenCLMemory> scalarCache;
	unordered_map<OpenCLDevice*, unique_ptr<DivideByScalarKernel<T>>> deviceAndDivideByScalarKernels;
	unordered_map<OpenCLDevice*, unique_ptr<SimpleSumKernel<T>>> deviceAndSimpleSumKernels;

	unique_ptr<OpenCLMemory> weights;
	unique_ptr<OpenCLMemory> biases;
	PerceptronLayerConfig config;
	LayerDataDescription inputDescription;

public:
	PerceptronLayer(shared_ptr<OpenCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			ATMLActivationFunction backPropActivation,
			const PerceptronLayerConfig* config);
	virtual ~PerceptronLayer();

	virtual void InterlockFinalized() override;

	virtual void EnqueueForwardPropagation(OpenCLDevice* device, int queueIndex,
			OpenCLMemory* previousInput, OpenCLMemory* output, bool blocking =
			true) override;

	virtual void EnqueueBackPropagation(OpenCLDevice* device, int queueIndex,
			OpenCLMemory* previousInput, OpenCLMemory* delta,
			OpenCLMemory* deltaOutput, bool blocking = true) override;

	virtual void EnqueueCalculateGradient(OpenCLDevice* device, int queueIndex,
		OpenCLMemory* previousInput, OpenCLMemory* delta, OpenCLMemory* gradient, bool blocking = true) override;

	virtual vector<tuple<OpenCLMemory*, int>> GetParameters() override;

	virtual void SetParameters(T* parameters, OpenCLDevice* device,
		int queueIndex, bool blocking = true) override;

	virtual void GetParameters(T* parameters, OpenCLDevice* device,
		int queueIndex, bool blocking = true) override;

	virtual size_t GetParameterCount() override;

	PerceptronLayerConfig GetConfig() const;

	Matrix<T> GetWeights();
	Matrix<T> GetBias();

	//TODO: Add some read / write parameters. Now it's all random

private:
	void InitializeNormalForwardPerceptron();
	void InitializeImageForwardPerceptron();
	void InitializeNormalBackPerceptron();
	void InitializeImageBackPerceptron();
	void InitializeNormalGradientKernel();
	void InitializeImageGradientKernel();
	void InitializeParameters();
};

}
/* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNNOPENCL_PERCEPTRONLAYER_H_ */
