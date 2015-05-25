/*
 * ConvolutionLayer.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef CNNOPENCL_CONVOLUTIONLAYER_H_
#define CNNOPENCL_CONVOLUTIONLAYER_H_

#include "OpenCLForwardBackPropLayer.h"
#include "CNN/ConvolutionLayerConfig.h"
#include "OpenCLHelper/OpenCLContext.h"

#include "ConvolutionKernel.h"
#include "BackConvolutionKernel.h"
#include "MultiplyAllUnitsKernel.h"
#include "ZeroBorderKenel.h"
#include "SumAllUnitsKernel.h"

#include <unordered_map>
#include <vector>
#include <memory>

using namespace std;
using namespace ATML::Helper;

namespace ATML
{
namespace MachineLearning
{

template<class T>
class ConvolutionLayer: public OpenCLForwardBackPropLayer<T>
{
private:

	unordered_map<OpenCLDevice*, unique_ptr<ConvolutionKernel<T>>> deviceAndConvolutionKernels;
	unordered_map<OpenCLDevice*, unique_ptr<SumAllUnitsKernel<T>>> deviceAndSumKernels;

	ConvolutionLayerConfig convolutionConfig;
	unique_ptr<OpenCLMemory> filters;
	unique_ptr<OpenCLMemory> biases;
	unique_ptr<OpenCLMemory> summaryCache;

public:
	ConvolutionLayer(shared_ptr<OpenCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			ATMLActivationFunction backPropActivation,
			const ConvolutionLayerConfig* config);
	virtual ~ConvolutionLayer();

	ConvolutionLayerConfig GetConfig() const;

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

	virtual void GetParameters(T* parameters, OpenCLDevice* device,
			int queueIndex, bool blocking = true) override;

	virtual void SetParameters(T* parameters, OpenCLDevice* device,
			int queueIndex, bool blocking = true) override;

	virtual size_t GetParameterCount() override;

private:
	void InitializeParameters();
	void InitializeConvolutionKernel();
	void InitializeSumAllKernel();
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* CNNOPENCL_CONVOLUTIONLAYER_H_ */
