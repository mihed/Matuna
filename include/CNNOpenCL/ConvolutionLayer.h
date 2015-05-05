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

namespace ATML
{
namespace MachineLearning
{

class ConvolutionLayer: public OpenCLForwardBackPropLayer
{
public:
	ConvolutionLayer(const LayerDataDescription& inputLayerDescription,
			const ConvolutionLayerConfig* config);
	~ConvolutionLayer();

	virtual void EnqueueForwardPropagation(
			shared_ptr<OpenCLMemory> previousInput,
			shared_ptr<OpenCLMemory> output) override;

	virtual void EnqueueBackPropagation(shared_ptr<OpenCLMemory> previousInput,
			shared_ptr<OpenCLMemory> delta,
			shared_ptr<OpenCLMemory> deltaOutput) override;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* CNNOPENCL_CONVOLUTIONLAYER_H_ */
