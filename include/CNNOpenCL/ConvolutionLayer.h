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

using namespace ATML::Helper;

namespace ATML
{
namespace MachineLearning
{

template<class T>
class ConvolutionLayer: public OpenCLForwardBackPropLayer<T>
{
public:
	ConvolutionLayer(shared_ptr<OpenCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			const ConvolutionLayerConfig* config);
	virtual ~ConvolutionLayer();

	virtual void InterlockFinalized() override;

	virtual void EnqueueForwardPropagation(OpenCLDevice* device, int queueIndex,
			OpenCLMemory* previousInput, OpenCLMemory* output, bool blocking =
					true) override;

	virtual void EnqueueBackPropagation(OpenCLDevice* device, int queueIndex,
			OpenCLMemory* previousInput, OpenCLMemory* delta,
			OpenCLMemory* deltaOutput, bool blocking = true) override;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* CNNOPENCL_CONVOLUTIONLAYER_H_ */
